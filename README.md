# HRI-TTR

HRI-TTR is an independent Human-to-G1 causal motion-token project. It does
not import or modify Think-Then-React at runtime. The original TTR repository
and its existing HRI adapters remain frozen references for behaviour,
measurements, and optional baseline weights.

This first iteration establishes code, contracts, tests, and training entry
points only. It does not claim full VQ training, reconstruction quality, or
Stage 3/4 language-model results.

Human and G1 tokenizers are separate models with separate normalizers,
encoders, decoders, and codebooks. Their tokens are only aligned later by a
language model.

## Setup

From the project root, run:

```sh
source scripts/use-nonhidden-uv-env.sh
uv sync --frozen
uv run hri-ttr --help
```

Source the script once in every new shell before using `uv`. On macOS, a hidden
file flag (`UF_HIDDEN`) can make Python skip the editable-install path file inside
the usual `.venv`. The script selects the visible `venv` directory instead. It
does not set `PYTHONPATH` or change file flags with `chflags`.

## Pair input safety

`data audit` and `data prepare` use paired files named
`<ID>_actor.npz` and `<ID>_reactor.npz` by default. NPZ loading disables object
arrays and requires the exact standalone actor/reactor fields.

Legacy Stage 0 `.pkl` pairs are still supported, but pickle can execute code
while loading. They are refused unless the caller explicitly passes
`--allow-trusted-pickle`. Use that flag only for the trusted local Stage 0
corpus; never use it for downloaded or unreviewed files. Prepared output records
`source_format=safe_npz` or `source_format=trusted_pickle` as provenance.

## Same-motion corpus

The formal Human/G1 tokenizer input is a prepared corpus, not the raw pairing
manifest. Build and verify it on the training server with:

```sh
hri-ttr data prepare-same-motion \
  --manifest /data/users/autovla/datasets/hri_ttr_same_motion_v1/manifest/same_motion.jsonl \
  --output /data/users/autovla/datasets/hri_ttr_same_motion_20hz_v1 \
  --g1-mjcf /data/users/autovla/sonic_repro/gmr/assets/unitree_g1/g1_mocap_29dof.xml \
  --target-fps 20 \
  --workers 16

hri-ttr data audit-corpus \
  --corpus /data/users/autovla/datasets/hri_ttr_same_motion_20hz_v1

hri-ttr data smoke-load \
  --corpus /data/users/autovla/datasets/hri_ttr_same_motion_20hz_v1 \
  --split train \
  --domain human \
  --batches 10
```

The corpus stores unnormalized `float32` Human `[T,262]` and G1 `[T,75]`
arrays at 20 FPS. The training loader applies the train-only normalizer at read
time, reads shards with NumPy memory mapping, never crosses a sequence boundary,
and masks repeated tail padding from the loss.

Formal tokenizer training selects the corpus directly and keeps train and
validation data separate:

```sh
torchrun --nproc_per_node=8 -m hri_ttr.cli train human-vq \
  --config configs/human_vq/causal_scratch_8x3090.json \
  --corpus /data/users/autovla/datasets/hri_ttr_same_motion_20hz_v1

torchrun --nproc_per_node=8 -m hri_ttr.cli train g1-vq \
  --config configs/g1_vq/causal_scratch_8x3090.json \
  --corpus /data/users/autovla/datasets/hri_ttr_same_motion_20hz_v1
```

These commands are documented for the later training stage; corpus preparation
and audit do not start either model.
