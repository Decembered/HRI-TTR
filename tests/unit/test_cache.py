from __future__ import annotations

# pyright: reportAny=false
import json
import shutil
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

from hri_ttr.cache import (
    CacheExistsError,
    CacheManifest,
    CacheValidationError,
    CacheWriteError,
    CacheWritePolicy,
    TokenCache,
    read_token_cache,
    write_token_cache,
)

SHA = "a" * 64


def _manifest() -> CacheManifest:
    return CacheManifest(
        tokenizer_sha256=SHA,
        checkpoint_sha256="b" * 64,
        normalizer_sha256="c" * 64,
        schema_sha256="d" * 64,
        split_sha256="e" * 64,
        valid_frame_lengths=(5, 8),
        valid_token_lengths=(2, 2),
        padded_frame_counts=(3, 0),
    )


def _cache() -> TokenCache:
    return TokenCache(
        tokens=np.array([1, 2, 3, 255], dtype=np.uint16),
        offsets=np.array([0, 2, 4], dtype=np.int64),
        sequence_ids=("alpha", "beta"),
        manifest=_manifest(),
    )


def test_cache_roundtrip_when_payload_is_valid(tmp_path: Path) -> None:
    # Given
    destination = tmp_path / "cache"

    # When
    _ = write_token_cache(destination, _cache())
    loaded = read_token_cache(destination)

    # Then
    np.testing.assert_array_equal(loaded.tokens, _cache().tokens)
    np.testing.assert_array_equal(loaded.offsets, _cache().offsets)
    assert loaded.sequence_ids == ("alpha", "beta")
    assert loaded.manifest == _manifest()


def test_cache_write_fails_safely_when_destination_exists(tmp_path: Path) -> None:
    # Given
    destination = tmp_path / "cache"
    _ = write_token_cache(destination, _cache())
    before = (destination / "tokens.npy").read_bytes()

    # When / Then
    with pytest.raises(CacheExistsError):
        _ = write_token_cache(destination, _cache())
    assert (destination / "tokens.npy").read_bytes() == before


@pytest.mark.parametrize(
    ("tokens", "offsets"),
    [
        (np.array([256], dtype=np.uint16), np.array([0, 1], dtype=np.int64)),
        (np.array([1], dtype=np.uint16), np.array([1, 1], dtype=np.int64)),
    ],
)
def test_cache_rejects_malformed_arrays(
    tokens: np.ndarray[tuple[int], np.dtype[np.uint16]],
    offsets: np.ndarray[tuple[int], np.dtype[np.int64]],
) -> None:
    # Given
    manifest = CacheManifest(
        tokenizer_sha256=SHA,
        checkpoint_sha256="b" * 64,
        normalizer_sha256="c" * 64,
        schema_sha256="d" * 64,
        split_sha256="e" * 64,
        valid_frame_lengths=(4,),
        valid_token_lengths=(1,),
        padded_frame_counts=(0,),
    )

    # When / Then
    with pytest.raises(CacheValidationError):
        _ = TokenCache(tokens, offsets, ("alpha",), manifest)


def test_cache_reader_rejects_unknown_manifest_field(tmp_path: Path) -> None:
    # Given
    destination = tmp_path / "cache"
    _ = write_token_cache(destination, _cache())
    manifest_path = destination / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    _ = manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    # When / Then
    with pytest.raises(CacheValidationError):
        _ = read_token_cache(destination)


def test_manifest_rejects_malformed_hash() -> None:
    # Given
    malformed = _manifest().model_dump()
    malformed["checkpoint_sha256"] = "not-a-hash"

    # When / Then
    with pytest.raises(ValidationError):
        _ = CacheManifest.model_validate(malformed)


def test_cache_replace_publishes_only_new_payload(tmp_path: Path) -> None:
    # Given
    destination = tmp_path / "cache"
    _ = write_token_cache(destination, _cache())
    replacement = TokenCache(
        np.array([9, 8, 7, 6], dtype=np.uint16),
        _cache().offsets,
        _cache().sequence_ids,
        _cache().manifest,
    )

    # When
    _ = write_token_cache(destination, replacement, policy="replace")

    # Then
    np.testing.assert_array_equal(
        read_token_cache(destination).tokens, replacement.tokens
    )
    assert {path.name for path in destination.iterdir()} == {
        "tokens.npy",
        "offsets.npy",
        "sequence_ids.json",
        "manifest.json",
    }


def test_cache_replace_rolls_back_when_publish_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    destination = tmp_path / "cache"
    original = _cache()
    _ = write_token_cache(destination, original)
    path_type = type(destination)
    real_replace = path_type.replace
    call_count = 0

    def interrupted_replace(source: Path, target: Path) -> Path:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError
        return real_replace(source, target)

    monkeypatch.setattr(path_type, "replace", interrupted_replace)

    # When / Then
    with pytest.raises(CacheWriteError):
        _ = write_token_cache(destination, original, policy="replace")
    np.testing.assert_array_equal(read_token_cache(destination).tokens, original.tokens)
    assert [
        path for path in tmp_path.iterdir() if path.name.startswith(".cache.")
    ] == []


def test_cache_policy_is_rejected_before_destination_is_touched(tmp_path: Path) -> None:
    # Given
    destination = tmp_path / "new-parent" / "cache"

    # When / Then
    with pytest.raises(CacheValidationError):
        _ = write_token_cache(destination, _cache(), policy="overwrite")
    assert not destination.parent.exists()


def test_cache_replace_restores_original_when_backup_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    destination = tmp_path / "cache"
    original = _cache()
    _ = write_token_cache(destination, original)
    replacement = TokenCache(
        np.array([9, 8, 7, 6], dtype=np.uint16),
        original.offsets,
        original.sequence_ids,
        original.manifest,
    )
    real_remove = shutil.rmtree
    call_count = 0

    def fail_first_remove(path: Path) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError
        real_remove(path)

    monkeypatch.setattr(shutil, "rmtree", fail_first_remove)

    # When / Then
    with pytest.raises(CacheWriteError):
        _ = write_token_cache(destination, replacement, policy=CacheWritePolicy.REPLACE)
    np.testing.assert_array_equal(read_token_cache(destination).tokens, original.tokens)
    assert tuple(tmp_path.iterdir()) == (destination,)


def test_cache_owns_read_only_array_snapshots() -> None:
    # Given
    tokens = np.array([1, 2, 3, 255], dtype=np.uint16)
    offsets = np.array([0, 2, 4], dtype=np.int64)

    # When
    cache = TokenCache(tokens, offsets, ("alpha", "beta"), _manifest())
    tokens[0] = 99
    offsets[1] = 1

    # Then
    assert cache.tokens[0] == 1
    assert cache.offsets[1] == 2
    assert not cache.tokens.flags.writeable
    assert not cache.offsets.flags.writeable
    assert cache.tokens.flags.owndata
    assert cache.offsets.flags.owndata


def test_cache_reader_rejects_unexpected_manifest_identity(tmp_path: Path) -> None:
    # Given
    destination = tmp_path / "cache"
    _ = write_token_cache(destination, _cache())
    expected = _manifest().model_copy(update={"checkpoint_sha256": "f" * 64})

    # When / Then
    with pytest.raises(CacheValidationError):
        _ = read_token_cache(destination, expected_manifest=expected)
