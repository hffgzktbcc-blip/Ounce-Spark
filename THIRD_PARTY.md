# Third-party material

Each entry names the project, licence, what was taken, and where it is recorded in `docs/PROVENANCE.md`.

- **dgx-spark-field-notes** (Sergio Silva, MIT): benchmark readings in `spikes/gate4a/field_notes_results.jsonl`.
  Source: <https://github.com/sergioamsilva/dgx-spark-field-notes>. Attribution kept in `data/SOURCES.md`.
- **ml_dtypes** 0.7.0 (Apache-2.0) and **gfloat** 0.5.2 (MIT): used at generation time by `oracles/gen_elements.py`; their outputs (decode tables, encode cases) are committed in `oracles/data/elements.json`. Neither is vendored.
- **ounce-core** (MIT, Copyright (c) 2026 hffgzktbcc-blip): the `mma` per-lane fragment layout functions in `crates/spark-num/src/mma.rs` are ported from `src/gpu/tensor.rs` (commit 0f4f231). Recorded in `docs/PROVENANCE.md`.
- **ptoxide** (MIT OR Apache-2.0, https://github.com/gvilums/ptoxide): `spikes/ptoxide-mma/mma.patch` is a patch against commit 522cde2; the fork is the proposed basis of the Gate 2 executor (docs/decisions/0001-ptx-executor.md).
