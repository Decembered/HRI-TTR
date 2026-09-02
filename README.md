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
