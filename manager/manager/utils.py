import asyncio
import signal
import time
from contextlib import asynccontextmanager
from uuid import UUID

from prefect import flow, get_client
from prefect.client.schemas.actions import WorkPoolCreate
from prefect.client.schemas.filters import (
    DeploymentFilter,
    DeploymentFilterId,
    DeploymentFilterTags,
)
from prefect.client.schemas.objects import WorkPool
from prefect.client.schemas.responses import DeploymentResponse
from prefect.exceptions import ObjectNotFound
from prefect.workers import ProcessWorker


async def create_work_pool(wp_name: str) -> WorkPool:
    async with get_client() as client:
        try:
            return await client.read_work_pool(wp_name)
        except ObjectNotFound:
            return await client.create_work_pool(
                WorkPoolCreate(
                    name=wp_name,
                    type='process',
                    base_job_template={},
                    is_paused=False,
                    concurrency_limit=5,
                )
            )


@asynccontextmanager
async def worker_manager(wp_name: str):
    # Create worker instance
    worker = ProcessWorker(work_pool_name=wp_name, prefetch_seconds=10, limit=5)

    # Start worker in background task
    worker_task = asyncio.create_task(worker.start())

    # Give worker time to initialize
    await asyncio.sleep(2)
    print('Worker started')

    try:
        yield worker
    finally:
        print('Worker stopping...')
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        print('Worker stopped')


async def wait_for_worker_registration(wp_name: str) -> None:
    """Wait until worker appears in work pool's worker list"""
    async with get_client() as client:
        start_time = time.time()
        while time.time() - start_time < 30:  # 30s timeout
            workers = await client.read_workers_for_work_pool(wp_name)
            if workers:
                print(f'Worker registered: {workers[0].name}')
                return True
            await asyncio.sleep(1)
        print('Worker failed to register within 30 seconds')
        return False


async def cleanup_session_deployments(
    session_id: UUID, additional_deployments: list[UUID] = []
) -> None:
    """Delete all deployments tagged with our session ID"""
    async with get_client() as client:
        # Find deployments with our session tag
        deployments: list[DeploymentResponse] = await client.read_deployments(
            deployment_filter=DeploymentFilter(
                tags=DeploymentFilterTags(all_=[str(session_id)])
            )
        )
        deployments += await client.read_deployments(
            deployment_filter=DeploymentFilter(
                id=DeploymentFilterId(any_=additional_deployments)
            )
        )

        if not deployments:
            print('No deployments found for cleanup')
            return

        print(f'Cleaning up {len(deployments)} deployments...')
        for deployment in deployments:
            try:
                await client.delete_deployment(deployment.id)
                print(f'Deleted deployment: {deployment.name} ({deployment.id})')
            except Exception as e:
                print(f'Error deleting {deployment.name}: {str(e)}')


def setup_stopsig() -> asyncio.Event:
    # Setup signal handling
    stop_event = asyncio.Event()

    def signal_handler() -> None:
        if not stop_event.is_set():
            stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    return stop_event


async def deploy_flow(
    name: str, tag: str, source: str, entrypoint: str, wp_name: str, parameters: dict | None = None
) -> UUID:
    sourced = await flow.from_source(source=source, entrypoint=entrypoint)
    deployment = await sourced.deploy(name=name, work_pool_name=wp_name, tags=[tag], parameters=parameters)

    message = f'Created deployment: {name} ({deployment})'
    if parameters is not None:
        message += f' with parameters {parameters}'
    print(message)

    return deployment
