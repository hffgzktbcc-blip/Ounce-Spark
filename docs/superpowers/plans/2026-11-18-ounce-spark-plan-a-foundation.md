# ounce-spark Plan A: Foundation, PTX inventory, numeric oracles, dataset tooling, Gate 2 spike

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `ounce-spark` repository with its governance guards, then deliver Gate 0b (PTX inventory), Gate 1 (spec-first numeric oracles with a Rust reference), Gate 3a/3d (calibration dataset and noise-floor tool) and the Gate 2 spike (decide fork-vs-fresh-vs-numerics-only for the PTX executor).

**Architecture:** A Rust workspace (`crates/spark-num` first) holds the numeric reference. Python tools under `tools/`, `inventory/` and `oracles/` produce committed, hash-checked data (PTX inventory, oracle JSON, dataset CSV) *before* any Rust code consumes it, so oracles stay independent of the emulator. Rust tests read the committed JSON.

**Tech Stack:** Rust (latest stable at execution time, pinned in `rust-toolchain.toml`), `serde_json` (dev-dependency), `cargo-deny`; Python 3.12 + `pytest`, `jsonschema`, `numpy`, `ml_dtypes`, `gfloat` (exact versions pinned in `oracles/requirements.txt`); GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-18-ounce-spark-design.md` (v3.2, approved 2026-11-18). Read sections 4, 7, 9.3, 12, 15 and 16 before starting.

**Scope of this plan (Plan A of 3).** In: repo scaffold, Gate 0b, Gate 1 (non-block-scaled `mma` for FP16, BF16, FP8 E4M3/E5M2, INT8, and all element formats), Gate 3a/3d, Gate 2 spike. **Out, deferred to later plans, with reasons:** Gate 2 executor (its shape depends on the spike's decision); Gate 3b outside-tester harness and `ptxas`-accepts check; Gate 4b calibrated model; block-scaled FP4/FP6 `mma` and TF32 oracles (need the Gate 0b inventory plus CUTLASS scale-factor reading first). Local repo only: **do not create a GitHub remote** without the owner's confirmation.

## Global Constraints

Copied from the spec; every task includes these.

- "**No money.** Free tools and free access only."
- "**MIT.** Port only from `ounce-core` (owner's MIT code), Mesa NAK and envytools (MIT, notices in `THIRD_PARTY.md`), and other MIT/Apache-2.0 sources with provenance recorded."
- "**Never used:** `ounce-runtime` and `oboromi` (GPL-3.0); anything from private `ounce-nx2` ...; retail content, firmware, keys."
- "v1 uses **no NVIDIA toolchain**: no `nvcc`, `ptxas` or `cuobjdump`, and no NVIDIA-built output is executed, disassembled or committed. All kernels are written by hand in PTX from NVIDIA's public documentation."
- "The HTML is **not committed** (NVIDIA documentation terms); only citations and extracted facts go in the repo."
- "Rust workspace, `#![forbid(unsafe_code)]` where possible."
- "Expected outputs and hashes committed before the emulator runs." (spec-first oracles)
- "an unmodelled instruction stops the run with its line or offset" (never guess).
- "A gate is done only when merged with CI green and its `PLAN.md` box ticked."
- Evidence labels: research-only, bounded-verified, independently reproduced, publicly supported, **spec-verified** (matches an oracle derived from NVIDIA's public specification, committed before the emulator ran; not hardware agreement), **inferred**.
- Not legal advice anywhere.

## Review Focus

Failure modes the spec implies but no gate's exit criteria name; each has its pinning test in the owning task.

1. **PTX page differs from expectation** (blocked/login page, new version, renamed sections): the extractor must fail loudly naming the missing section, never emit a partial inventory. (Task 2)
2. **Special values in 4/6/8-bit formats:** E2M1/E2M3/E3M2 have no Inf or NaN codes; `.satfinite` must clamp, never overflow; `-0` must survive; E4M3 NaN is only code `0x7F`/`0xFF`; midpoints round to even. (Tasks 3, 4)
3. **Packed `x2` field order:** which operand lands in the upper half; a wrong order silently swaps elements. (Task 4)
4. **`mma` with NaN, Inf, all-zero or empty-K inputs:** must propagate or return zero, never panic. (Task 6)
5. **Noise floor on tiny or dirty groups:** a group of one value, a zero or negative tok/s, or duplicate rows from one submitter counted as several submitters. (Task 7)

## File Structure

```
ounce-spark/                         (git init here; local only)
  Cargo.toml  rust-toolchain.toml  deny.toml  LICENSE  README.md  PLAN.md  THIRD_PARTY.md  .gitignore
  .github/workflows/ci.yml
  docs/{LIMITS,PROVENANCE,SOURCES}.md   docs/decisions/0001-ptx-executor.md
  docs/superpowers/{specs,plans}/       (exist)
  tools/{check_governance,noise_floor,import_field_notes}.py
  inventory/{fetch_ptx,extract_ptx}.py  inventory/{schema.json,ptx_sm12x.json,SOURCE.json,OPEN_QUESTIONS.md}
  oracles/{gen_elements,gen_mma}.py  oracles/{requirements.txt,MANIFEST.sha256,KNOWN_DISAGREEMENTS.md}
  oracles/data/{elements,mma}.json
  data/{spark_decode.csv,schema.json,SOURCES.md}
  crates/spark-num/{Cargo.toml,src/{lib,elem,mma}.rs,tests/{elements_oracle,mma_oracle}.rs}
  tests/py/test_*.py
  spikes/gate4a/ (exists)   spikes/ptoxide-mma/ (Task 8)
  .cache/                    (gitignored; fetched NVIDIA HTML)
```

Task order: 1, then 2 and 7 in parallel, then 3 and 5 (both need 2), then 4 (needs 3) and 6 (needs 5), then 8 (needs 5, 6), then 9.

---

### Task 1: Repository scaffold and governance guard

**Files:**
- Create: `Cargo.toml` (workspace, `members = ["crates/*"]`), `rust-toolchain.toml`, `deny.toml`, `LICENSE` (MIT, "Copyright (c) 2026 the ounce-spark contributors"), `README.md`, `PLAN.md`, `THIRD_PARTY.md`, `.gitignore` (`target/`, `.cache/`, `__pycache__/`), `docs/LIMITS.md`, `docs/PROVENANCE.md`, `docs/SOURCES.md`, `.github/workflows/ci.yml`, `tools/check_governance.py`
- Test: `tests/py/test_check_governance.py`

**Interfaces:**
- Produces: `tools/check_governance.py::check(root: pathlib.Path) -> list[str]` (violation strings; empty means clean). CLI `python tools/check_governance.py` exits 1 and prints violations.
- `PLAN.md` holds the gate checklist (Gate 0b, 1, 2 spike, 2, 3a, 3b/c, 4b) with unticked boxes and the rule "done only when merged, CI green, box ticked". `docs/LIMITS.md` holds the seven evidence labels above. `docs/SOURCES.md` copies the spec section 7 allowed/never-used lists.

- [ ] **Step 1: Write the failing tests** in `tests/py/test_check_governance.py`, building tiny trees under `tmp_path`:
  - `test_clean_tree_passes`: complete required-file tree gives `[]`.
  - `test_missing_required_file_reported`: no `LICENSE` gives a list containing `"missing: LICENSE"`.
  - `test_forbid_unsafe_required`: `crates/x/src/lib.rs` without `#![forbid(unsafe_code)]` gives `"forbid(unsafe_code) missing: crates/x/src/lib.rs"`.
  - `test_elf_binary_rejected`: file starting `b"\x7fELF"` gives `"binary rejected: <path>"`; `test_cubin_extension_rejected` (`a.cubin`, `a.fatbin`) likewise.
  - `test_nvidia_doc_html_rejected`: a file whose first 2048 bytes contain both `<html` and `PTX ISA` gives `"NVIDIA documentation HTML rejected: <path>"`.
  - `test_real_repo_passes`: `check(<repo root>) == []`.
  Required files: `LICENSE README.md PLAN.md THIRD_PARTY.md deny.toml rust-toolchain.toml docs/LIMITS.md docs/PROVENANCE.md docs/SOURCES.md`. The walk skips `.git`, `target`, `.cache`.
- [ ] **Step 2: Run** `python -m pytest tests/py/test_check_governance.py -v`. Expected: FAIL (module not found).
- [ ] **Step 3: Implement** `check` and the CLI in `tools/check_governance.py`, create the scaffold files above and a CI workflow with two jobs, `governance` (`python tools/check_governance.py`) and `pytest` (`python -m pytest tests/py`). Run `git init -b main` at the repo root.
- [ ] **Step 4: Run** `python -m pytest tests/py -v` and `python tools/check_governance.py`. Expected: all PASS, exit 0.
- [ ] **Step 5: Commit** (`git add -A && git commit -m "chore: scaffold ounce-spark with governance guard"`). The existing `spikes/gate4a/` and `docs/` are included in this first commit.

---

### Task 2: Gate 0b, PTX inventory extractor

**Files:**
- Create: `inventory/fetch_ptx.py`, `inventory/extract_ptx.py`, `inventory/schema.json`, `inventory/ptx_sm12x.json` (generated, committed), `inventory/SOURCE.json`, `inventory/OPEN_QUESTIONS.md`
- Modify: `.github/workflows/ci.yml` (add job `inventory`: fetch, extract, diff against committed JSON)
- Test: `tests/py/test_extract_ptx.py`

**Interfaces:**
- Produces: `fetch(dest: Path) -> Path` (downloads `https://docs.nvidia.com/cuda/parallel-thread-execution/` to `dest`, default `.cache/ptx.html`, returns the path; `SOURCE.json` records URL, UTC date and sha256); `extract(html: str) -> dict`; `main()` writes `inventory/ptx_sm12x.json`.
- Output shape (validated by `inventory/schema.json`): `{"ptx_version": str, "source_sha256": str, "targets": [str], "static_smem_kb": {target: int}, "instructions": [ {"name": str, "qualifiers": [str], "syntax": str, "archs": [str], "family_archs": [str], "clause": str, "anchor": str} ]}`. `archs` and `family_archs` are the architecture-specific and family-specific lists as printed. Cover `mma` (including `.kind::` and `.block_scale` forms), `cvt` (all `e2m1x2`/`e2m3x2`/`e3m2x2`/`ue8m0x2`/`e4m3x2`/`e5m2x2` forms), `cp.async`, `ldmatrix`, `shfl`, `bar`, atomics, and `.approx` error bounds.
- Excerpts of NVIDIA text in the JSON are limited to 200 characters per entry; the HTML itself is never written under a tracked path.

- [ ] **Step 1: Write the failing tests.** Unit tests use small **synthetic** HTML fixtures written by hand (own wording, mimicking the page's structure, not copied excerpts):
  - `test_extract_fails_on_missing_section`: html lacking the shared-memory section raises `ExtractionError` whose message names `static shared memory`.
  - `test_extract_rejects_non_ptx_page`: a login/blocked page raises `ExtractionError("not a PTX ISA page")`.
  - `test_schema_validates_output`: `jsonschema.validate(extract(fixture), schema)` passes; every entry has non-empty `clause`.
  - Integration `test_real_page_facts` (skipped with `pytest.skip` when `.cache/ptx.html` is absent): `static_smem_kb["sm_121a"] == 100`, `static_smem_kb["sm_90a"] == 228`; an `mma` entry with qualifier `.kind::mxf4nvf4` has `archs` containing both `sm_120a` and `sm_121a`; the `cvt` entry for destination `e2m1x2` from `.f32` has `archs` containing `sm_120a` and a derived field `sm_121a_in_archs` that the test asserts `False`, plus `family_lists_sm120f` asserted `True`.
  These last assertions encode the flattened-text read in spec section 16. **If the structured parse disagrees, stop and record the discrepancy in `inventory/OPEN_QUESTIONS.md`; do not edit the assertion to match.**
- [ ] **Step 2: Run** `python -m pytest tests/py/test_extract_ptx.py -v`. Expected: FAIL (module not found).
- [ ] **Step 3: Implement** `extract` with an HTML parser (`html.parser` or `lxml`, pin it), working from document structure (headings, tables, `<pre>` syntax blocks), not flattened text. Raise `ExtractionError` naming any expected section that is missing.
- [ ] **Step 4: Run** `python inventory/fetch_ptx.py && python inventory/extract_ptx.py`, then the tests. Expected: PASS including the integration test; `inventory/ptx_sm12x.json` and `SOURCE.json` written.
- [ ] **Step 5: Write `inventory/OPEN_QUESTIONS.md`:** the exact `cvt` arch lists as parsed, the PTX anchor, and a line asking the owner to confirm visually; then tick the Gate 0b box in `PLAN.md` **only after** the owner confirms. Run `python tools/check_governance.py` (must still pass: no HTML tracked). Commit.

---

### Task 3: Gate 1a, element-format oracles (Python, two libraries)

**Files:**
- Create: `oracles/gen_elements.py`, `oracles/requirements.txt` (exact pins), `oracles/KNOWN_DISAGREEMENTS.md`, `oracles/data/elements.json`, `oracles/MANIFEST.sha256`
- Modify: `.github/workflows/ci.yml` (job `oracles`: regenerate to a temp dir and diff against committed files)
- Test: `tests/py/test_gen_elements.py`

**Interfaces:**
- Consumes: `inventory/ptx_sm12x.json` (which formats and `cvt` forms exist for `sm_121a`).
- Produces: `build() -> dict` and `main(out_dir: Path)`. `elements.json` shape: `{"formats": {name: {"bits": int, "decode": [float-as-hex-string per code], "max": float, "has_inf": bool, "has_nan": bool}}, "encode_cases": [{"fmt": str, "x": hex-string, "sat": bool, "code": int}]}` for `e4m3`, `e5m2`, `e2m3`, `e3m2`, `e2m1`, `ue8m0`. `MANIFEST.sha256` lists the sha256 of every file under `oracles/data/`.
- Decode tables are **exhaustive** over all codes. `ml_dtypes` and `gfloat` are both used; any code where they disagree must be listed in `oracles/KNOWN_DISAGREEMENTS.md` or generation fails.
- `encode_cases` cover: `0.0`, `-0.0`, the smallest subnormal, the exact midpoint between each pair of adjacent codes at one exponent (ties to even), the maximum, one half-ulp above maximum, `1e30`, `inf`, `-inf`, `nan`, each with `sat` true and false. NaN/Inf under `.satfinite` are included **only if** the inventory's `cvt` clause defines them; otherwise omit and list the omission in `docs/LIMITS.md`.

- [ ] **Step 1: Write the failing tests** in `tests/py/test_gen_elements.py` (spec-first facts, asserted independent of the libraries): `e2m1` decode values are exactly `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` and their negatives; maxima `e4m3` 448, `e5m2` 57344, `e2m3` 7.5, `e3m2` 28, `e2m1` 6; `ue8m0` code 127 decodes to 1.0 and code 255 is NaN; `e2m1`/`e2m3`/`e3m2` report `has_inf == has_nan == False`; `test_two_libraries_agree_on_all_decodes` (every code, minus listed disagreements); `test_build_is_deterministic` (two `build()` calls produce equal JSON); `test_manifest_matches_files`.
- [ ] **Step 2: Run** `python -m pytest tests/py/test_gen_elements.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `oracles/gen_elements.py`. Install pins with `pip install -r oracles/requirements.txt`.
- [ ] **Step 4: Run** `python oracles/gen_elements.py oracles/data && python -m pytest tests/py/test_gen_elements.py -v`. Expected: PASS; `elements.json` and `MANIFEST.sha256` written.
- [ ] **Step 5: Record provenance** (ml_dtypes Apache-2.0, gfloat MIT, versions) in `docs/PROVENANCE.md` and `THIRD_PARTY.md`; commit **the oracle data before any Rust consumer exists**.

---

### Task 4: `spark-num` element codecs (Rust)

**Files:**
- Create: `crates/spark-num/Cargo.toml` (dev-dependency `serde_json`), `crates/spark-num/src/lib.rs` (`#![forbid(unsafe_code)]`), `crates/spark-num/src/elem.rs`
- Modify: `.github/workflows/ci.yml` (add `cargo test` and `cargo deny check`)
- Test: `crates/spark-num/tests/elements_oracle.rs`

**Interfaces:**
- Consumes: `oracles/data/elements.json` (Task 3), `inventory/ptx_sm12x.json` (packing order).
- Produces: `pub enum Fmt { E4M3, E5M2, E2M3, E3M2, E2M1, Ue8m0 }`; `pub fn decode(fmt: Fmt, code: u8) -> f64` (NaN for NaN codes); `pub fn encode_rne(fmt: Fmt, x: f64, sat: bool) -> u8`; `pub fn pack_x2(fmt: Fmt, first: u8, second: u8) -> u8` with the field order taken from the inventory's `cvt` clause.

- [ ] **Step 1: Write the failing tests** in `elements_oracle.rs`: `decode_matches_oracle_exhaustively` (every code of every format, bit-exact via `f64::to_bits`, NaNs compared as NaN); `encode_matches_oracle_cases` (every `encode_cases` entry); `e2m1_has_no_inf_or_nan_codes`; `negative_zero_survives` (`encode_rne(E2M1, -0.0, false)` decodes with the sign bit set); `satfinite_clamps_overflow` (`encode_rne(E4M3, 1e30, true)` decodes to 448); `pack_x2_order_matches_inventory` (reads the inventory clause and asserts which argument occupies the upper bits).
- [ ] **Step 2: Run** `cargo test -p spark-num`. Expected: FAIL (does not compile).
- [ ] **Step 3: Implement** `elem.rs` against the signatures above; ties-to-even implemented on the bit pattern, not by float formatting.
- [ ] **Step 4: Run** `cargo test -p spark-num && cargo deny check`. Expected: PASS.
- [ ] **Step 5: Tick** the element-formats line of the Gate 1 box in `PLAN.md`; commit.

---

### Task 5: Gate 1b, `mma` oracles (Python)

**Files:**
- Create: `oracles/gen_mma.py`, `oracles/data/mma.json`
- Modify: `oracles/MANIFEST.sha256`, `.github/workflows/ci.yml` (extend `oracles` job)
- Test: `tests/py/test_gen_mma.py`

**Interfaces:**
- Consumes: `inventory/ptx_sm12x.json`; `oracles/data/elements.json`.
- Produces: `build(seed: int = 20261118) -> dict`. `mma.json` shape: `{"cases": [{"id": str, "shape": [m,n,k], "a_fmt": str, "b_fmt": str, "acc": "f32"|"s32", "a": [codes], "b": [codes], "c": [values], "class": "exact"|"banded", "expected": [values], "band": [per-element abs tolerance]}]}`. Combinations: `m16n8k8` and `m16n8k16` f16, `m16n8k16` bf16, `m16n8k32` e4m3 and e5m2 (f32 accumulate), `m16n8k32` s8 (s32 accumulate). **The generator refuses any shape/type combination the inventory does not list for `sm_121a`.**
- `exact` cases use small power-of-two-valued inputs so every partial sum is exactly representable: any accumulation order gives the same bits, so comparison is bit-exact. `banded` cases use seeded random inputs; `expected` is the exact rational result (Python `fractions.Fraction`) rounded once to f32; `band[i] = K * 2^-24 * (sum_k |a_ik * b_kj| + |c_ij|)`. This band is **my derivation** from standard floating-point summation error analysis, used because PTX leaves accumulation order unspecified (cite the inventory clause); record it in `docs/LIMITS.md`. INT8 cases are always `exact`.

- [ ] **Step 1: Write the failing tests:** `test_deterministic_for_seed`; `test_exact_cases_are_order_independent` (recompute each `exact` case in float64 with three different summation orders and assert identical results); `test_refuses_unlisted_combination` (monkeypatch the inventory to drop `bf16` and assert `build()` raises); `test_case_counts` (at least 20 cases per combination, at least 5 of them `exact`); `test_band_formula` (for one hand-built case the band equals `K * 2**-24 * (sum|ab| + |c|)`).
- [ ] **Step 2: Run** `python -m pytest tests/py/test_gen_mma.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `oracles/gen_mma.py`.
- [ ] **Step 4: Run** `python oracles/gen_mma.py oracles/data && python -m pytest tests/py -v`. Expected: PASS.
- [ ] **Step 5: Commit** the data and updated manifest before Task 6 begins.

---

### Task 6: `spark-num` `mma` reference (Rust)

**Files:**
- Create: `crates/spark-num/src/mma.rs`
- Modify: `crates/spark-num/src/lib.rs` (`pub mod mma;`), `docs/PROVENANCE.md` (record the port from `ounce-core/src/gpu/tensor.rs`, MIT), `THIRD_PARTY.md`
- Test: `crates/spark-num/tests/mma_oracle.rs`

**Interfaces:**
- Consumes: `elem::decode`; `oracles/data/mma.json`.
- Produces: `pub struct MmaShape { pub m: usize, pub n: usize, pub k: usize }`; `pub fn mma_ref_f32(shape: MmaShape, a: &[f64], b: &[f64], c: &[f32]) -> Vec<f32>` (inputs already decoded, products exact, f32 accumulate in increasing `k`); `pub fn mma_ref_s32(shape: MmaShape, a: &[i8], b: &[i8], c: &[i32]) -> Vec<i32>` (wrapping unless the inventory clause says otherwise); `pub fn lane_a_coords(shape: MmaShape, lane: usize) -> Vec<(usize, usize)>`, `lane_b_coords`, `lane_cd_coords`, ported for the two f16 shapes only. Lane layouts for the `k32` types are **not** implemented here (Plan B); record that in `docs/LIMITS.md`.

- [ ] **Step 1: Write the failing tests** in `mma_oracle.rs`: `exact_cases_bit_exact` (every `exact` case, `f32::to_bits` equality); `banded_cases_within_band` (every `banded` case, `|got - expected| <= band`); `nan_input_propagates` (a NaN in `a` makes the affected outputs NaN, no panic); `inf_minus_inf_is_nan`; `all_zero_inputs_return_c`; `lane_coords_cover_matrix_once` (union over 32 lanes of `lane_a_coords` for `m16n8k16` covers every `(row, col)` of A exactly once).
- [ ] **Step 2: Run** `cargo test -p spark-num --test mma_oracle`. Expected: FAIL.
- [ ] **Step 3: Implement** `mma.rs`. Port the f16 lane-layout functions from `ounce-core` with the MIT notice kept; do not copy `tensor.rs`'s synthetic opcodes.
- [ ] **Step 4: Run** `cargo test -p spark-num && python tools/check_governance.py`. Expected: PASS.
- [ ] **Step 5: Tick** the Gate 1 box in `PLAN.md` for the formats covered; label in `docs/LIMITS.md`: **spec-verified**, bit-exact for `exact` cases, banded otherwise. Commit.

---

### Task 7: Gate 3a/3d, calibration dataset and noise-floor tool

**Files:**
- Create: `data/spark_decode.csv`, `data/schema.json`, `data/SOURCES.md`, `tools/import_field_notes.py`, `tools/noise_floor.py`
- Test: `tests/py/test_noise_floor.py`, `tests/py/test_dataset.py`

**Interfaces:**
- Produces: CSV columns `id, source_url, source_kind (verified_raw|secondary), operator, model, active_params_b, params_total_b, fmt, bits_per_weight, stack, stack_version, build, kernel, depth, metric (tg|pp), batch, spec_decoding (true|false), value_tps, value_pm, date, notes`.
- `tools/noise_floor.py`: `sigma_between(values: list[float], submitters: list[str]) -> float | None`; `interval(x_med: float, sigma: float) -> tuple[float, float]` = `(x_med*exp(-1.96*sigma), x_med*exp(1.96*sigma))`; `interval_error(pred: float, lo: float, hi: float) -> float` (0 inside; else `(lo-pred)/lo` below, `(pred-hi)/hi` above); `group_rows(rows) -> dict` keyed by `(model, fmt, stack, depth, metric)`.
- `sigma_between` is the **unscaled** median absolute deviation of `ln(x)` across **distinct** submitters (as the spec states; no 1.4826 factor). Returns `None` when fewer than 2 distinct submitters.
- `tools/import_field_notes.py` reads `spikes/gate4a/field_notes_results.jsonl` (MIT-licensed source: attribution in `data/SOURCES.md`) plus a hand-transcribed set of the verified llama.cpp rows from `spikes/gate4a/gate4a_v2.py`.

- [ ] **Step 1: Write the failing tests:** `sigma_between([60.0, 60.0], ["a","b"]) == 0.0`; `sigma_between([1.0, math.e], ["a","b"])` equals the MAD of `[0, 1]` (0.5); a single value returns `None`; the same submitter twice counts once (`sigma_between([50,60], ["a","a"]) is None`); `ValueError` on a zero or negative value; `interval(60, 0.0) == (60, 60)`; `interval_error(pred=88.8, lo=55, hi=66)` returns `(88.8-66)/66`; `interval_error(60, 55, 66) == 0`. Dataset tests: `test_csv_validates_against_schema` (every row has `value_tps > 0` and a non-empty `source_url`); `test_secondary_rows_flagged` (rows from review sites have `source_kind == "secondary"`); `test_spec_decoding_column_present` (every row has `spec_decoding` set to `true` or `false`, so later baseline scripts can filter on it).
- [ ] **Step 2: Run** `python -m pytest tests/py/test_noise_floor.py tests/py/test_dataset.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the tool and importer; generate `data/spark_decode.csv` with the 15 readings from spec section 15, each with its provenance.
- [ ] **Step 4: Run** `python -m pytest tests/py -v`. Expected: PASS.
- [ ] **Step 5: Tick** the Gate 3a/3d line in `PLAN.md`; commit.

---

### Task 8: Gate 2 spike, `ptoxide` with one `mma` shape

**Files:**
- Create: `docs/decisions/0001-ptx-executor.md`, `spikes/ptoxide-mma/README.md`, `spikes/ptoxide-mma/kernels/mma_m16n8k16_f16.ptx` (hand-written), `spikes/ptoxide-mma/mma.patch` (the fork's diff, **not** a vendored copy), `tools/check_decision.py`
- Test: `tests/py/test_check_decision.py`

**Interfaces:**
- Consumes: `oracles/data/mma.json` case ids for `m16n8k16` f16 (Task 5); `crates/spark-num::mma::mma_ref_f32` (Task 6); `inventory/ptx_sm12x.json` clauses.
- Produces: the decision record with front-matter `decision: fork|fresh|numerics-only`, `hours_spent: <number>`, `ptoxide_commit: <sha>`, `licence_checked: true`. `tools/check_decision.py::check(path: Path) -> list[str]`.
- **Rule from the spec (Gate 2 spike):** if adding this one shape takes more than 2 working days, `decision: numerics-only` and Gate 2 is rescoped to numerics plus single-instruction PTX tests.

- [ ] **Step 1: Write the failing test** `test_decision_record_valid`: the front-matter exists, `decision` is one of the three values, `hours_spent` is a number, `licence_checked` is `true`, and `numerics-only` is chosen whenever `hours_spent > 16`.
- [ ] **Step 2: Run** `python -m pytest tests/py/test_check_decision.py -v`. Expected: FAIL (no record).
- [ ] **Step 3: Spike, timeboxed, record hours as you go.** Clone `https://github.com/gvilums/ptoxide` into `spikes/ptoxide-mma/vendor/` (gitignored); read its `LICENSE` files and confirm Apache-2.0-or-MIT; record the commit sha and licence in `docs/PROVENANCE.md`. Run its tests; grep it for `mma`, `shfl`, `atom` and record what exists. Write the hand-written PTX kernel: one warp loads the A/B/C fragments per the lane layout in the inventory, executes `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, stores D. Add the instruction to the fork until the kernel's D matches the `mma.json` `exact` and `banded` cases for that shape (reuse `spark-num` as a path dependency). Save the diff as `mma.patch`.
- [ ] **Step 4: Write `docs/decisions/0001-ptx-executor.md`:** parser fit for our syntax (`.param`, `ld/st.global`, `ld/st.shared`, `bar.sync`, predicated `bra`, `mad/fma`, `cvt`, `mma`), hours per sub-step, and the decision. Run `python -m pytest tests/py/test_check_decision.py -v`. Expected: PASS.
- [ ] **Step 5: Tick** the Gate 2 spike box in `PLAN.md`; commit.

---

### Task 9: Close-out and next-plan scope

**Files:**
- Modify: `PLAN.md`, `docs/LIMITS.md`
- Create: `docs/superpowers/plans/NEXT.md`

- [ ] **Step 1:** Run the full suite (`python -m pytest tests/py`, `cargo test --workspace`, `cargo deny check`, `python tools/check_governance.py`). Expected: all pass.
- [ ] **Step 2:** Write `NEXT.md` listing Plan B (Gate 2 executor or numerics-only per the decision record, memcheck checks, hand-written PTX kernel corpus, quantised-model sanity, lane layouts for the `k32` types) and Plan C (Gate 3b outside-tester harness with the `ptxas`-accepts check, Gate 4b calibrated model with leave-one-model-out, block-scaled FP4/FP6 and TF32 oracles, Spark Arena data if `spark-arena.com` is allowed). Note the unresolved Gate 0b `cvt` question if still open.
- [ ] **Step 3:** Confirm every ticked `PLAN.md` box links a merged commit; update `docs/LIMITS.md` with what each label currently covers. Commit.

---

## Self-review

**Spec coverage.** Gate 0b: Task 2. Gate 1: Tasks 3-6 (element formats, FP16/BF16/FP8/INT8 `mma`). Gate 3a/3d: Task 7. Gate 2 spike and its decision rule: Task 8. Governance, MIT, licence boundaries and the no-NVIDIA-artifacts rule: Task 1. **Gaps, deliberate and listed under "Scope":** Gate 2 executor, Gate 3b/3c, Gate 4b, block-scaled FP4/FP6 and TF32 oracles, `k32` lane layouts.

**Type consistency.** `Fmt`, `decode`, `encode_rne`, `pack_x2` (Task 4) are the names Task 6 consumes (`elem::decode`). `mma_ref_f32` and `MmaShape` (Task 6) match Task 8. JSON shapes of `elements.json` and `mma.json` are defined once (Tasks 3, 5) and read by Tasks 4, 6.

**Proportion.** About 330 lines against a spec of about 470; code is limited to signatures, test names and assertions with the spec's values.

**Open items the executor must not paper over.** (1) Task 2's `cvt` assertions encode a flattened-text read; a disagreement is recorded, not hidden. (2) The `mma` band formula in Task 5 is my derivation; the owner may overrule it in review. (3) Whether `spark-arena.com` is allowed is still the owner's call (Plan C).
