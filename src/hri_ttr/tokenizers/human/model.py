"""Independent Human 262D causal tokenizer."""

from __future__ import annotations

from typing import Final

from hri_ttr.tokenizers.common.contracts import TokenizerArchitecture
from hri_ttr.tokenizers.common.layers import CausalDecoder, CausalEncoder
from hri_ttr.tokenizers.common.model import CausalMotionTokenizer


class HumanEncoder(CausalEncoder):
    """Human-specific encoder instance for the official-compatible 262D schema."""


class HumanDecoder(CausalDecoder):
    """Human-specific decoder instance for the official-compatible 262D schema."""


class HumanTokenizer(CausalMotionTokenizer):
    """Own all Human encoder, decoder, codebook, and EMA state."""

    human_feature_dim: Final[int] = 262

    def __init__(self, architecture: TokenizerArchitecture | None = None) -> None:
        """Create an independent Human tokenizer."""
        selected = architecture or TokenizerArchitecture()
        super().__init__(
            self.human_feature_dim,
            selected,
            HumanEncoder(
                self.human_feature_dim,
                selected.width,
                selected.code_dim,
                selected.residual_depth,
            ),
            HumanDecoder(
                self.human_feature_dim,
                selected.width,
                selected.code_dim,
                selected.residual_depth,
            ),
        )
