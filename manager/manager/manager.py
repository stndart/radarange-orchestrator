import asyncio
from uuid import UUID, uuid4

from prefect import flow
from service import SPAWNER_FLOW_NAME, ManagedService
from utils import (
    cleanup_session_deployments,
    create_work_pool,
    deploy_flow,
    setup_stopsig,
    wait_for_worker_registration,
    worker_manager,
)


class ServiceManager:
    wp_name: str
    session_id: UUID
    default_deployments: list[UUID] = []

    def __init__(self, wp_name: str = 'services-lair') -> None:
        self.wp_name = wp_name
        self.session_id = f'session-{uuid4()}'

    def entrypoint(self) -> str:
        return f'{__file__}:ServiceManager.spawner_flow'

    @staticmethod
    @flow(log_prints=True)
    async def spawner_flow(name: str, session_id: str, wp_name: str):
        print(f'Spawner flow {name} running')
        try:
            service_instance = await ManagedService.aload(name)
        except ValueError:
            print(f'Service {name} is not registered')
        else:
            await deploy_flow(
                name=name,
                tag=session_id,
                source='./',
                entrypoint=service_instance.entrypoint(),
                wp_name=wp_name,
                parameters={'config': service_instance.config}
            )

    async def main(self) -> None:
        stop_event = setup_stopsig()

        try:
            # Create work pool
            work_pool = await create_work_pool(self.wp_name)
            print(f'Work pool ready: {work_pool.name}')

            # Deploy flows
            parent_deployment = await deploy_flow(
                name=SPAWNER_FLOW_NAME,
                tag=self.session_id,
                source='./',
                entrypoint=self.entrypoint(),
                wp_name=self.wp_name,
                parameters={'session_id': self.session_id, 'wp_name': self.wp_name},
            )
            self.default_deployments.append(parent_deployment)

            # Run worker and flow
            async with worker_manager(self.wp_name):
                # Ensure worker is properly registered
                if not await wait_for_worker_registration(self.wp_name):
                    print('Worker registration failed. Exiting.')
                    return

                print('Running... Press Ctrl+C to stop')
                await stop_event.wait()
                print('Shutdown signal received')

        finally:
            await cleanup_session_deployments(self.session_id)


# export PREFECT_API_URL="http://localhost:4200/api"
if __name__ == '__main__':
    manager = ServiceManager()
    asyncio.run(manager.main())
