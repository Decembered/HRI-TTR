"""Public G1 75D representation API."""

from hri_ttr.representations.g1.constants import (
    G1_DOF_AXES,
    G1_DOF_LIMITS_RAD,
    G1_DOF_NAMES,
)
from hri_ttr.representations.g1.contacts import (
    DEFAULT_G1_FOOT_CONTACT_THRESHOLDS,
    G1FootContactThresholds,
    compute_g1_foot_contacts,
)
from hri_ttr.representations.g1.episode import EpisodeFrame
from hri_ttr.representations.g1.features import (
    DecodedG1Motion,
    EncodedG1Motion,
    G1MotionInput,
    decode_g1_features,
    encode_g1_features,
)
from hri_ttr.representations.g1.normalizer import G1FeatureNormalizer
from hri_ttr.representations.g1.schema import (
    G1_FEATURE_DIM,
    G1_FEATURE_FIELDS,
    G1_FEATURE_SLICES,
    G1_SCHEMA_VERSION,
)

__all__ = [
    "DEFAULT_G1_FOOT_CONTACT_THRESHOLDS",
    "G1_DOF_AXES",
    "G1_DOF_LIMITS_RAD",
    "G1_DOF_NAMES",
    "G1_FEATURE_DIM",
    "G1_FEATURE_FIELDS",
    "G1_FEATURE_SLICES",
    "G1_SCHEMA_VERSION",
    "DecodedG1Motion",
    "EncodedG1Motion",
    "EpisodeFrame",
    "G1FeatureNormalizer",
    "G1FootContactThresholds",
    "G1MotionInput",
    "compute_g1_foot_contacts",
    "decode_g1_features",
    "encode_g1_features",
]
