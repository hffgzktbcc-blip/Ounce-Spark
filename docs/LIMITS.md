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
