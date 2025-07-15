from __future__ import annotations

from pydantic import BaseModel
from itertools import permutations


class Resources(BaseModel):
    ram: int = 0
    gpu_memory: list[int] = [0]

    def copy(self) -> Resources:
        return Resources(ram=self.ram, gpu_memory=self.gpu_memory.copy())

    def __iadd__(self, other: Resources) -> Resources:
        self.ram += other.ram
        for i in range(len(self.gpu_memory)):
            self.gpu_memory[i] += other.gpu_memory[i]
        return self

    def __isub__(self, other: Resources) -> Resources:
        self.ram += other.ram
        for i in range(len(self.gpu_memory)):
            self.gpu_memory[i] += other.gpu_memory[i]
        return self

    def __add__(self, other: Resources) -> Resources:
        return self.copy().__iadd__(other)

    def __sub__(self, other: Resources) -> Resources:
        return self.copy().__isub__(other)

    def __lt__(self, other: Resources) -> bool:
        if self.ram >= other.ram:
            return False
        if other.get_gpu(self.gpu_memory) is None:
            return False
        return True
    
    def __eq__(self, other: Resources) -> bool:
        if self.ram != other.ram:
            return False
        if list(sorted(self.gpu_memory)) != list(sorted(other.gpu_memory)):
            return False
        return True

    def get_gpu(self, vram: int | list[int]) -> int | list[int] | None:
        return self.check_available(vram, self.gpu_memory)

    def get_gpu_applied(self, vram: int | list[int]) -> list[int]:
        idx = self.get_gpu(vram)
        if idx is None:
            return
        if isinstance(idx, int):
            idx = [idx]
        ds = [vram] if isinstance(vram, int) else list(vram)

        gpu_memory = [0] * len(self.gpu_memory)
        for i, d in zip(idx, ds):
            gpu_memory[i] = d
        return gpu_memory

    @staticmethod
    def check_available(
        demand: int | list[int], available: int | list[int]
    ) -> int | list[int] | None:
        # Normalize inputs to lists
        d = [demand] if isinstance(demand, int) else list(demand)
        a = [available] if isinstance(available, int) else list(available)

        # Single demand: return first slot index with enough available
        if len(d) == 1:
            return next((i for i, v in enumerate(a) if v >= d[0]), None)
        # Try all assignments of demands to distinct slots
        for perm in permutations(range(len(a)), len(d)):
            if all(a[j] >= d[i] for i, j in enumerate(perm)):
                return list(perm)
        return None
