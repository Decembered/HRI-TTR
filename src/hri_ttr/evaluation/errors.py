"""Typed failures shared by metric and diagnostic boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

from typing_extensions import override


@dataclass(frozen=True, slots=True)
class EvaluationError(ValueError):
    """Report malformed tensors supplied to an evaluation boundary."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


def reject_evaluation(detail: str) -> NoReturn:
    """Raise the typed evaluation boundary error."""
    raise EvaluationError(detail)
