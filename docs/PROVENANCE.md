# Provenance register

Every ported file, dataset and external source is recorded here with licence and version.

| Item | Source | Licence | Used for | Notes |
|---|---|---|---|---|
| `spikes/gate4a/field_notes_results.jsonl` | https://github.com/sergioamsilva/dgx-spark-field-notes | MIT (c) 2026 Sergio Silva | Gate 4a readings | copied unmodified; measurements are the author's |
| PTX ISA 9.5 documentation | https://docs.nvidia.com/cuda/parallel-thread-execution/ | NVIDIA documentation terms | Gate 0b inventory | **HTML is never committed**; only extracted facts and citations |
| `oracles/gen_elements.py` inputs | ml_dtypes 0.7.0 (https://github.com/jax-ml/ml_dtypes) | Apache-2.0 | element-format oracle (library 1) | pinned in `oracles/requirements.txt`; not vendored |
| `oracles/gen_elements.py` inputs | gfloat 0.5.2 (https://github.com/graphcore-research/gfloat) | MIT | element-format oracle (library 2) | pinned; not vendored |
