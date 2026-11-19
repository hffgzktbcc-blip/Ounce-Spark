# What each label means, and what each result does not prove

| Label | Meaning |
|---|---|
| research-only | explored, not dependable |
| bounded-verified | matches a reference on the listed inputs, with retained evidence |
| independently reproduced | someone outside the project reran it and got the same result |
| publicly supported | documented, versioned, and maintained for users |
| **spec-verified** | matches an oracle derived independently from NVIDIA's public specification, committed before the emulator ran. **Not** hardware agreement. PTX-level semantics only. |
| **inferred** | rests on a reverse-engineered SASS encoding; never upgraded to spec-verified |

Without Blackwell hardware, GPU claims cap at **spec-verified** until a Spark owner reproduces them.

## Current coverage

Nothing is verified yet beyond the oracle data below. Entries are added as each gate lands.

### Element-format oracles (Task 3, `oracles/data/elements.json`)
- Decode tables are exhaustive for e4m3, e5m2, e2m3, e3m2, e2m1 and ue8m0; gfloat and ml_dtypes agree on every code.
- Encode cases: ties-to-even midpoints between every adjacent pair of values (including zero), min subnormal, max,
  max plus half a gap, +-1e30, +-0.0, with `sat` true; non-saturating cases exist only for e4m3 and e5m2 and are marked
  `ptx_valid: false` (PTX makes `.satfinite` mandatory for these destination types).
- Only **in-range** encodes are cross-checked by both libraries; overflow, Inf and NaN expectations come from gfloat alone.
- **Omitted:** NaN/Inf inputs under `.satfinite` (the inventory does not define them); NaN/Inf inputs for formats that
  have neither (e2m1, e2m3, e3m2); `ue8m0` encode (PTX offers only `.rz`/`.rp` for it and the libraries disagree on RN).
  Label: **spec-verified** for what is listed (element semantics follow the OCP formats that PTX's `cvt` uses).

### Rust element codecs (Task 4, `crates/spark-num/src/elem.rs`)
- `decode` matches the oracle bit-exactly for every code of all six formats; `encode_rne` matches all encode cases.
- `encode_rne(Ue8m0, ..)` returns `Err(NoRoundToNearest)`: PTX has only `.rz`/`.rp` for ue8m0 (deferred).
- NaN results are checked as "any NaN code": the spec fixes no canonical NaN code.
- `pack_x2` returns `u16`; the first operand goes in the upper lane (documented for e2m1x2, e4m3x2, e5m2x2, e2m3x2, e3m2x2,
  ue8m0x2 in the PTX text). The `x4` forms and reverse-direction unpacking are not modelled yet.

### `mma` oracles (Task 5, `oracles/data/mma.json`)
- Combinations (all non-block-scaled, listed by the PTX inventory): `m16n8k8` and `m16n8k16` f16, `m16n8k16` bf16,
  `m16n8k32` e4m3 and e5m2 with f32 accumulate, `m16n8k32` s8 with s32 accumulate. 20 cases each, 120 in total.
- `exact` cases are bit-exact in **any** accumulation order (inputs chosen so every partial sum is exactly representable;
  s8 is integer arithmetic). `banded` cases carry a tolerance band **derived by me**: gamma_{K+2} * (sum |a*b| + |c|),
  gamma_n = n*2^-24/(1 - n*2^-24). It bounds any sequential f32 summation of K products plus c and the final rounding.
  The Plan's `K * 2^-24 * (...)` was not an upper bound (it ignored the addition of c and the final rounding).
- PTX leaves accumulation order and internal precision unspecified (the band is therefore not a hardware claim).
- **Not covered:** e4m3/e5m2 at `m16n8k16` (the plain f8 form also lists it), mixed e4m3 x e5m2, f16 accumulate, TF32,
  `m16n8k8` bf16, double precision, u8/u4/s4/b1, sparse `mma.sp`, block-scaled and `.kind::` forms, NaN/Inf inputs.
- Label: **spec-verified** for the listed cases (the Rust reference of Task 6 passes them); not hardware agreement.

### Rust `mma` reference (Task 6, `crates/spark-num/src/mma.rs`)
- `mma_ref_f32` accumulates in f32 in increasing k from c; all 60 `exact` oracle cases are bit-exact (sign of a zero sum is not
  compared) and all 60 `banded` cases are inside their band. PTX only requires "at least single precision" and leaves order,
  rounding and subnormal handling unspecified, so hardware may differ within the band.
- `mma_ref_s32` wraps without `.satfinite` (stated by PTX); the saturating form is not implemented.
- Lane fragment layouts exist only for the two f16 shapes (m16n8k8, m16n8k16), checked against the PTX formulas and for
  exactly-once coverage. Layouts for the k32 types are Plan B.
- NaN and Inf propagate as IEEE f32 arithmetic does; this is tested on 2x2x2 matrices, not taken from the oracle.
- Label: **spec-verified** for the listed combinations, not hardware agreement.

### Calibration dataset and noise floor (Task 7, `data/`, `tools/noise_floor.py`)
- 15 readings: 8 `verified_raw` (field notes JSONL) and 7 `secondary` (including the 2 llama.cpp rows, which were read through a
  page-summary tool, so they are **not** raw-verified; spec section 15 called them verified and that was too strong).
- One source is internally inconsistent: the 30.3 tok/s Gemma-4 NVFP4 row is labelled "SEM spec-decoding" in results.jsonl but
  "no spec-decoding" in the README and RESULTS.md. It is recorded with `spec_decoding = unknown`; Gate 4a v2 had counted it as
  non-speculative (it moves the Gemma cell median from 0.30 to 0.29, no conclusion changes).
- **The noise floor cannot be estimated yet.** Exactly one group has two distinct submitters (gpt-oss-120B MXFP4 on llama.cpp:
  eugr and verdictbits) and the tool reports `sigma_between = 0.047`. That is half the log difference of two medians, the
  verdictbits figure probably comes from the same llama.cpp results file, and the 25% within-operator spread across builds
  (45.87 vs 60.57) is hidden because values from one submitter are reduced to their median. Do not quote 0.047 as a noise floor.
- `sigma_between` is the unscaled MAD of ln(x) across submitters, as the spec states; build and kernel fixed effects are not modelled.

### Gate 2 spike (Task 8, `spikes/ptoxide-mma/`)
- One hand-written PTX kernel executes `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` for one warp through a fork of ptoxide and matches
  all 8 `exact` oracle cases bit for bit and all 12 `banded` cases within band. Label: **spec-verified** for that one instruction and shape.
- Not covered: every other shape and type, divergence (a lane exiting early is reported, not simulated), shared memory and barriers in a kernel,
  shuffles, atomics, anything beyond one warp.
- Assumption (not verified): the lower-index element of a packed f16x2 fragment register is in bits 15:0.
- The "hours" in the decision record are the executing agent's wall-clock (about 8 minutes), a weak proxy for the spec's 16-hour rule.
