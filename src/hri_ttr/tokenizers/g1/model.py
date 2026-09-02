"""Independent G1 75D causal tokenizer."""

from __future__ import annotations

from typing import Final

from hri_ttr.tokenizers.common.contracts import TokenizerArchitecture
from hri_ttr.tokenizers.common.layers import CausalDecoder, CausalEncoder
from hri_ttr.tokenizers.common.model import CausalMotionTokenizer


class G1Encoder(CausalEncoder):
    """G1-specific encoder instance for the 75D V2 schema."""


class G1Decoder(CausalDecoder):
    """G1-specific decoder instance for the 75D V2 schema."""


class G1Tokenizer(CausalMotionTokenizer):
    """Own all G1 encoder, decoder, codebook, and EMA state."""

    g1_feature_dim: Final[int] = 75

    def __init__(self, architecture: TokenizerArchitecture | None = None) -> None:
        """Create an independent G1 tokenizer."""
        selected = architecture or TokenizerArchitecture()
        super().__init__(
            self.g1_feature_dim,
            selected,
            G1Encoder(
                self.g1_feature_dim,
                selected.width,
                selected.code_dim,
                selected.residual_depth,
            ),
            G1Decoder(
                self.g1_feature_dim,
                selected.width,
                selected.code_dim,
                selected.residual_depth,
            ),
        )
