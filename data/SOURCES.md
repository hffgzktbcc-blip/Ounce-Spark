# Dataset sources and attribution

`data/spark_decode.csv` holds the 15 single-stream decode readings behind Gate 4a (spec section 15).

| Source | Kind | Licence / terms | Rows |
|---|---|---|---|
| dgx-spark-field-notes, https://github.com/sergioamsilva/dgx-spark-field-notes, results in `spikes/gate4a/field_notes_results.jsonl` | `verified_raw` (read from the raw file) | MIT, (c) 2026 Sergio Silva | 8 |
| llama.cpp discussion #16578, https://github.com/ggml-org/llama.cpp/discussions/16578 | `secondary` (read through a page-summary tool; the raw page returned HTTP 403) | no licence stated for user-posted numbers; attribution and URL kept | 2 |
| verdictbits review, NVIDIA developer blog, LMSYS review | `secondary` (search snippets) | not checked | 5 |

Not ingested yet: Spark Arena (results live on `spark-arena.com`, a host this environment does not allow), the JetsonHacks
table (its URL returns 404), and the other field-notes rows (speculative decoding, models with unknown active parameters).
The `active_params_b` and `bits_per_weight` columns come from model memory or model names, not from the sources.
