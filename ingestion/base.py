"""Abstract adapter interface all ingestion sources implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from core.schemas import Job


class IngestionAdapter(ABC):
    """Yield canonical Job objects; never leak source-specific types upward."""

    source_name: str

    @abstractmethod
    async def fetch_jobs(self, **kwargs) -> AsyncIterator[Job]:
        """Stream canonical jobs. Implementations handle pagination internally."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the source is reachable."""
        ...
