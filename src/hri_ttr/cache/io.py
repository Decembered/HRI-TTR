"""Atomic filesystem persistence for token caches."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import shutil
import struct
import tempfile
from array import array
from enum import StrEnum
from pathlib import Path
from re import fullmatch
from typing import Literal, assert_never

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError

from hri_ttr.cache.errors import CacheExistsError, CacheValidationError, CacheWriteError
from hri_ttr.cache.models import CacheManifest, SequenceIds, TokenCache

NPY_MAGIC = b"\x93NUMPY\x01\x00"
NPY_PREAMBLE_BYTES = 10


class CacheWritePolicy(StrEnum):
    """Select safe refusal or transactional cache replacement."""

    FAIL = "fail"
    REPLACE = "replace"


def _parse_policy(policy: CacheWritePolicy | str) -> CacheWritePolicy:
    try:
        parsed = CacheWritePolicy(policy)
    except ValueError:
        detail = f"unknown cache write policy: {policy}"
        raise CacheValidationError(detail) from None
    match parsed:
        case CacheWritePolicy.FAIL:
            return CacheWritePolicy.FAIL
        case CacheWritePolicy.REPLACE:
            return CacheWritePolicy.REPLACE
    assert_never(parsed)


def _write_npy(
    path: Path, values: npt.NDArray[np.uint16] | npt.NDArray[np.int64], descr: str
) -> None:
    header_text = (
        f"{{'descr': '{descr}', 'fortran_order': False, 'shape': ({len(values)},), }}"
    )
    padding = -(len(NPY_MAGIC) + 2 + len(header_text) + 1) % 64
    header = (header_text + " " * padding + "\n").encode("ascii")
    with path.open("wb") as stream:
        _ = stream.write(NPY_MAGIC)
        _ = stream.write(struct.pack("<H", len(header)))
        _ = stream.write(header)
        _ = stream.write(values.tobytes(order="C"))


def _read_npy(path: Path, descr: Literal["<u2", "<i8"]) -> bytes:
    raw = path.read_bytes()
    if len(raw) < NPY_PREAMBLE_BYTES or raw[:8] != NPY_MAGIC:
        detail = f"invalid npy preamble: {path}"
        raise CacheValidationError(detail)
    header_length = struct.unpack("<H", raw[8:10])[0]
    header_end = 10 + header_length
    header = raw[10:header_end].decode("ascii")
    pattern = (
        rf"\{{'descr': '{descr}', 'fortran_order': False, 'shape': \((\d+),\), \}} *\n"
    )
    matched = fullmatch(pattern, header)
    if matched is None:
        detail = f"invalid npy header: {path}"
        raise CacheValidationError(detail)
    count = int(matched.group(1))
    item_size = 2 if descr == "<u2" else 8
    payload = raw[header_end:]
    if len(payload) != count * item_size:
        detail = f"invalid npy payload length: {path}"
        raise CacheValidationError(detail)
    return payload


def write_token_cache(
    destination: Path,
    cache: TokenCache,
    *,
    policy: CacheWritePolicy | str = CacheWritePolicy.FAIL,
) -> Path:
    """Publish all four cache files as one directory transaction."""
    selected_policy = _parse_policy(policy)
    parent = destination.parent
    _ = parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and selected_policy is CacheWritePolicy.FAIL:
        raise CacheExistsError(str(destination))
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    backup = parent / f"{temporary.name}.backup"
    try:
        _write_npy(temporary / "tokens.npy", cache.tokens, "<u2")
        _write_npy(temporary / "offsets.npy", cache.offsets, "<i8")
        _ = (temporary / "sequence_ids.json").write_text(
            SequenceIds(cache.sequence_ids).model_dump_json(indent=2),
            encoding="utf-8",
        )
        _ = (temporary / "manifest.json").write_text(
            cache.manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        if destination.exists():
            _ = destination.replace(backup)
        _ = temporary.replace(destination)
        if backup.exists():
            shutil.rmtree(backup)
    except OSError as error:
        if backup.exists() and destination.exists():
            _ = destination.replace(temporary)
        if backup.exists():
            _ = backup.replace(destination)
        raise CacheWriteError(str(destination)) from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def read_token_cache(
    source: Path, *, expected_manifest: CacheManifest | None = None
) -> TokenCache:
    """Parse a complete cache and reject missing, extra, or malformed content."""
    expected = {"tokens.npy", "offsets.npy", "sequence_ids.json", "manifest.json"}
    try:
        actual = {path.name for path in source.iterdir()}
        if actual != expected:
            detail = "cache directory must contain exactly four files"
            raise CacheValidationError(detail)  # noqa: TRY301
        token_values: array[int] = array("H")
        token_values.frombytes(_read_npy(source / "tokens.npy", "<u2"))
        offset_values: array[int] = array("q")
        offset_values.frombytes(_read_npy(source / "offsets.npy", "<i8"))
        tokens = np.asarray(token_values, dtype=np.uint16)
        offsets = np.asarray(offset_values, dtype=np.int64)
        sequence_ids = SequenceIds.model_validate_json(
            (source / "sequence_ids.json").read_bytes()
        ).root
        manifest = CacheManifest.model_validate_json(
            (source / "manifest.json").read_bytes()
        )
        if expected_manifest is not None and manifest != expected_manifest:
            detail = "cache manifest identity does not match the expected artifacts"
            raise CacheValidationError(detail)  # noqa: TRY301
        return TokenCache(tokens, offsets, sequence_ids, manifest)
    except CacheValidationError:
        raise
    except (OSError, ValueError, ValidationError) as error:
        detail = f"invalid token cache at {source}"
        raise CacheValidationError(detail) from error
