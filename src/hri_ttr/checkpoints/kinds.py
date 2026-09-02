"""Shared identities that do not import training implementation."""

from enum import StrEnum


class ModelKind(StrEnum):
    """Independent tokenizer domains."""

    HUMAN = "human"
    G1 = "g1"
