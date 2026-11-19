# Provenance register

Every ported file, dataset and external source is recorded here with licence and version.

| Item | Source | Licence | Used for | Notes |
|---|---|---|---|---|
| `spikes/gate4a/field_notes_results.jsonl` | https://github.com/sergioamsilva/dgx-spark-field-notes | MIT (c) 2026 Sergio Silva | Gate 4a readings | copied unmodified; measurements are the author's |
| PTX ISA 9.5 documentation | https://docs.nvidia.com/cuda/parallel-thread-execution/ | NVIDIA documentation terms | Gate 0b inventory | **HTML is never committed**; only extracted facts and citations |
| `oracles/gen_elements.py` inputs | ml_dtypes 0.7.0 (https://github.com/jax-ml/ml_dtypes) | Apache-2.0 | element-format oracle (library 1) | pinned in `oracles/requirements.txt`; not vendored |
| `oracles/gen_elements.py` inputs | gfloat 0.5.2 (https://github.com/graphcore-research/gfloat) | MIT | element-format oracle (library 2) | pinned; not vendored |
| `crates/spark-num/src/mma.rs` `lane_a_coords`/`lane_b_coords`/`lane_cd_coords` | ounce-core `src/gpu/tensor.rs` (commit 0f4f231, https://github.com/hffgzktbcc-blip/ounce-core) | MIT (c) 2026 hffgzktbcc-blip | per-lane fragment layouts for m16n8k8/m16n8k16 f16 | re-expressed shape-generically; ounce-core's synthetic opcodes NOT ported; validated against the PTX formulas in `tests/mma_oracle.rs` |
| `spikes/ptoxide-mma/mma.patch` | ptoxide, https://github.com/gvilums/ptoxide, commit 522cde29551cdb66a8a8456a36da21bae57d72fe | MIT OR Apache-2.0 (LICENSE-MIT and LICENSE-APACHE present) | Gate 2 spike: a fork adding `mma.sync` m16n8k16 | only our patch is committed (with context lines); the clone is git-ignored; ptoxide's bundled `nvcc`-generated kernels/tests are NOT used (decision D1, see docs/decisions/0001-ptx-executor.md) |
