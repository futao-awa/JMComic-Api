"""PDF requests finish before the download and isolate client cancellation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from jmcomic_api.concurrency.tasks import BusyError, TaskManager
from jmcomic_api.services.album import PdfArtifact
from jmcomic_api.services.jm_client import JmRuntime
from jmcomic_api.services.pdf_jobs import PdfJobs
from jmcomic_api.settings import Settings


async def test_submit_deduplicates_and_completes_after_request_left(tmp_path):
    gate = asyncio.Event()
    path = tmp_path / "complete.pdf"
    path.write_bytes(b"%PDF-complete")

    async def build(*args, **kwargs):
        await gate.wait()
        return PdfArtifact(path, "test.pdf")

    manager = TaskManager()
    service = SimpleNamespace(get_or_build=AsyncMock(side_effect=build))
    registry = PdfJobs(manager, service, Settings(min_free_disk_mb=0))
    runtime = JmRuntime(None, None)
    params = dict(album_id="123", shard_index=0, encrypted=True, quality=45)
    job = registry.submit(runtime, **params)
    assert registry.submit(runtime, **params) is job
    await asyncio.sleep(0)
    assert job.state == "running"
    assert runtime._users == 1
    with pytest.raises(HTTPException) as error:
        registry.get(job.id, "different-account")
    assert error.value.status_code == 404
    gate.set()
    await asyncio.gather(*manager.tasks)
    assert job.state == "ready"
    assert registry.status(job)["data"]["size_bytes"] == len(b"%PDF-complete")
    assert runtime._users == 0
    assert service.get_or_build.await_count == 1
    assert service.get_or_build.await_args.kwargs["enable_pwd"] is True


async def test_range_job_passes_explicit_page_bounds(tmp_path):
    path = tmp_path / "range.pdf"
    path.write_bytes(b"%PDF-range")

    service = SimpleNamespace(
        build_range=AsyncMock(
            return_value=(
                PdfArtifact(path, "range_11_20.pdf"),
                100,
                11,
                20,
                "Test Album",
            )
        )
    )
    registry = PdfJobs(TaskManager(), service, Settings(min_free_disk_mb=0))
    runtime = JmRuntime(None, None)
    job = registry.submit(
        runtime,
        album_id="123",
        shard_index=0,
        start_page=11,
        end_page=20,
        encrypted=True,
        quality=45,
    )
    await asyncio.gather(*registry.manager.tasks)

    assert job.state == "ready"
    service.build_range.assert_awaited_once_with(
        "123",
        11,
        20,
        registry.settings.shard_cache_path,
        enable_pwd=True,
        jpeg_quality=45,
    )


async def test_queue_limit_failure_retry_and_expiration(tmp_path):
    gate = asyncio.Event()

    async def build(*args, **kwargs):
        await gate.wait()
        raise TimeoutError()

    manager = TaskManager()
    service = SimpleNamespace(get_or_build=AsyncMock(side_effect=build))
    registry = PdfJobs(manager, service, Settings(queue_capacity=0, max_concurrent_downloads=1))
    runtime = JmRuntime(None, None)
    params = dict(album_id="123", shard_index=0, encrypted=True, quality=45)
    job = registry.submit(runtime, **params)
    with pytest.raises(BusyError):
        registry.submit(runtime, **{**params, "album_id": "456"})
    gate.set()
    await asyncio.gather(*manager.tasks)
    assert job.state == "failed"
    assert "timed out" in job.message
    retry = registry.submit(runtime, **params)
    assert retry.id != job.id
    await asyncio.gather(*manager.tasks)
    registry.ttl = 0
    registry.prune()
    assert not registry.jobs


async def test_sent_job_is_not_reused_after_server_file_cleanup(tmp_path):
    path = tmp_path / "sent.pdf"
    path.write_bytes(b"%PDF-sent")
    service = SimpleNamespace(
        get_or_build=AsyncMock(return_value=PdfArtifact(path, "sent.pdf"))
    )
    registry = PdfJobs(TaskManager(), service, Settings(min_free_disk_mb=0))
    runtime = JmRuntime(None, None)
    params = dict(album_id="123", shard_index=0, encrypted=True, quality=45)

    first = registry.submit(runtime, **params)
    await asyncio.gather(*registry.manager.tasks)
    assert first.state == "ready"
    registry.mark_sent(first)
    assert first.state == "sent"

    second = registry.submit(runtime, **params)
    assert second.id != first.id
    await asyncio.gather(*registry.manager.tasks)
    assert service.get_or_build.await_count == 2


def test_task_http_status_and_binary_size_guard(client, tmp_path):
    tc, runtime = client
    runtime.identity = "anonymous"
    runtime.generation = "test"
    path = tmp_path / "result.pdf"
    path.write_bytes(b"%PDF-result")
    service = SimpleNamespace(get_or_build=AsyncMock(return_value=PdfArtifact(path, "result.pdf")))
    registry = PdfJobs(tc.app.state.manager, service, tc.app.state.settings)
    tc.app.state.pdf_jobs = registry
    response = tc.post("/pdf/jobs", json={"album_id": "123", "quality": "medium"})
    assert response.status_code == 202
    job_id = response.json()["data"]["job_id"]
    tc.portal.call(asyncio.sleep, 0.01)
    assert tc.get(f"/pdf/jobs/{job_id}").json()["data"]["state"] == "ready"
    assert tc.get(f"/pdf/jobs/{job_id}/file?max_bytes=1").status_code == 413
    result = tc.get(f"/pdf/jobs/{job_id}/file")
    assert result.content == b"%PDF-result"
    assert result.headers["content-type"].startswith("application/pdf")
    assert tc.get("/pdf/jobs/unknown").status_code == 404
    assert tc.post("/pdf/jobs", json={"album_id": "123", "shard_index": -1}).status_code == 422
