"""Tests for the Stage 3/4 token timeline boundary."""

from __future__ import annotations

import numpy as np
import pytest

from hri_ttr.contracts import (
    FramesPerToken,
    ModelId,
    SchemaId,
    TokenBatch,
    TokenizerSpec,
)
from hri_ttr.language import (
    LanguageContractError,
    OfflineTeacherExample,
    OnlineStudentExample,
    SemanticCondition,
)


def _spec(kind: str, model: str) -> TokenizerSpec:
    return TokenizerSpec(
        kind=kind,
        schema_id=SchemaId(f"{kind}-schema"),
        model_id=ModelId(model),
        input_fps=20.0,
        frames_per_token=FramesPerToken(4),
        codebook_size=256,
    )


def _tokens(kind: str, values: list[int], model: str) -> TokenBatch:
    ids = np.asarray([values], dtype=np.int64)
    return TokenBatch(ids, np.ones_like(ids, dtype=np.bool_), _spec(kind, model))


def test_offline_teacher_accepts_same_index_target() -> None:
    # Given: H[0:3], G[0:2], and target G[2].
    human = _tokens("human", [1, 2, 3], "human-a")
    g1 = _tokens("g1", [8, 9], "g1-a")
    target = _tokens("g1", [10], "g1-a")

    # When: the offline teacher example is constructed for k=2.
    example = OfflineTeacherExample(
        human_prefix=human,
        g1_prefix=g1,
        target_g1=target,
        target_index=2,
        semantic=SemanticCondition("approach the person"),
    )

    # Then: the target index remains the same Human prefix index.
    assert example.target_index == 2


def test_offline_teacher_rejects_current_g1_in_history() -> None:
    # Given: an invalid G1 prefix that already contains G[k].
    human = _tokens("human", [1, 2, 3], "human-a")
    g1 = _tokens("g1", [8, 9, 10], "g1-a")

    # When / Then: constructing H[0:k] + G[0:k] -> G[k] is rejected.
    with pytest.raises(LanguageContractError, match="offline teacher timeline"):
        _ = OfflineTeacherExample(
            human_prefix=human,
            g1_prefix=g1,
            target_g1=_tokens("g1", [10], "g1-a"),
            target_index=2,
            semantic=SemanticCondition("turn left"),
        )


def test_online_student_accepts_next_index_target() -> None:
    # Given: paired Human and G1 observations through k=1.
    human = _tokens("human", [1, 2], "human-a")
    g1 = _tokens("g1", [8, 9], "g1-a")

    # When: the future online student example is constructed.
    example = OnlineStudentExample(
        human_prefix=human,
        g1_prefix=g1,
        target_index=2,
        semantic=SemanticCondition("hold position"),
    )

    # Then: it predicts the next G1 token.
    assert example.target_index == 2


def test_language_boundary_rejects_swapped_token_domains() -> None:
    # Given: a G1 token batch placed in the Human slot.
    swapped = _tokens("g1", [7], "g1-a")

    # When / Then: the semantic token domains cannot be interchanged.
    with pytest.raises(LanguageContractError, match="human token domain"):
        _ = OnlineStudentExample(
            human_prefix=swapped,
            g1_prefix=_tokens("g1", [8], "g1-a"),
            target_index=1,
            semantic=SemanticCondition("wait"),
        )


def test_language_boundary_requires_final_tokenizer_timing() -> None:
    # Given: a Human tokenizer with the wrong frame-to-token ratio.
    ids = np.asarray([[1]], dtype=np.int64)
    wrong = TokenBatch(
        ids,
        np.ones_like(ids, dtype=np.bool_),
        TokenizerSpec(
            kind="human",
            schema_id=SchemaId("human-schema"),
            model_id=ModelId("human-a"),
            input_fps=20.0,
            frames_per_token=FramesPerToken(2),
            codebook_size=256,
        ),
    )

    # When / Then: a Stage 3/4 example refuses the incompatible timeline.
    with pytest.raises(LanguageContractError, match="tokenizer timing"):
        _ = OnlineStudentExample(
            human_prefix=wrong,
            g1_prefix=_tokens("g1", [8], "g1-a"),
            target_index=1,
            semantic=SemanticCondition("wait"),
        )


def test_language_contract_owns_prefix_arrays() -> None:
    # Given: caller-owned arrays used to construct a Human token batch.
    ids = np.asarray([[4, 5]], dtype=np.int64)
    mask = np.ones_like(ids, dtype=np.bool_)
    human = TokenBatch(ids, mask, _spec("human", "human-a"))

    # When: the caller mutates its original array.
    ids[0, 0] = 200

    # Then: the language example still sees the owned immutable copy.
    example = OnlineStudentExample(
        human_prefix=human,
        g1_prefix=_tokens("g1", [6, 7], "g1-a"),
        target_index=2,
        semantic=SemanticCondition("wait"),
    )
    assert example.human_prefix.token_ids.tolist() == [[4, 5]]
