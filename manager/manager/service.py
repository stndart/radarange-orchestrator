from __future__ import annotations

import time

from prefect import flow
from prefect.blocks.core import Block
from prefect.states import (
    Completed,
    # Pending,
    # Running,
    # Failed,
    # Cancelled,
    # Suspended,
    State,
)
from pydantic import BaseModel

SPAWNER_FLOW_NAME = 'spawner'


class ServiceConfig(BaseModel):
    name: str
    some_arg: int


class ServiceConfigBlock(Block):
    config: ServiceConfig


class ManagedService:
    config: ServiceConfig

    @staticmethod
    def load(name: str) -> ServiceInstance:
        config = ServiceConfigBlock.load(name).config
        return ServiceInstance(config=config)

    @staticmethod
    async def aload(name: str) -> ServiceInstance:
        blk = await ServiceConfigBlock.load(name)
        return ServiceInstance(config=blk.config)


class InstanceConfig(BaseModel):
    name: str
    some_arg: int

    @staticmethod
    def from_service_config(config: ServiceConfig) -> InstanceConfig:
        return InstanceConfig(name=config.name, some_arg=config.some_arg)


class ServiceInstance:
    config: InstanceConfig

    def __init__(self, config: ServiceConfig):
        self.config = InstanceConfig.from_service_config(config)

    def entrypoint(self) -> str:
        return f'{__file__}:ServiceInstance.run'

    @staticmethod
    @flow(log_prints=True)
    def run(config: InstanceConfig) -> State:
        print('Got config!', config.some_arg)
        time.sleep(10)
        return Completed(message="All's fine!")


# export PREFECT_API_URL="http://localhost:4200/api"
if __name__ == '__main__':
    some_service = ServiceConfig(name='ss', some_arg=42)
    ServiceConfigBlock(config=some_service).save(some_service.name)