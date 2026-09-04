"""Typed token-cache boundary failures."""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import override


@dataclass(frozen=True, slots=True)
class CacheValidationError(ValueError):
    """Report an invalid cache payload at the filesystem boundary."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class CacheExistsError(FileExistsError):
    """Report a destination protected by the fail-if-exists policy."""

    path: str

    @override
    def __str__(self) -> str:
        return f"token cache already exists: {self.path}"


@dataclass(frozen=True, slots=True)
class CacheWriteError(OSError):
    """Report an atomic cache publication failure."""

    path: str

    @override
    def __str__(self) -> str:
        return f"could not publish token cache: {self.path}"
