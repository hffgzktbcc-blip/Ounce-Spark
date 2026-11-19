# Gate 0b open questions (owner: confirm each against the live page)

Source: PTX ISA 9.5 (`inventory/SOURCE.json`: URL, fetch time, sha256). Facts below come from the **structured**
parse (`inventory/extract_ptx.py`), not flattened text. Anchors are on the page above.

## 1. Does `sm_121a` support `cvt` to/from `e2m1x2`?  (clause 9.7.10.24, anchor `data-movement-and-conversion-instructions-cvt`)
The statements "cvt.rn.satfinite{.relu}.{e2m1x2/e2m3x2/e3m2x2/ue8m0x2}.f32 is supported on following architectures"
and "cvt.rn{.relu}.f16x2.{e2m1x2/e2m3x2/e3m2x2} ..." list, as architecture-specific targets, `sm_100a`, `sm_101a`
(renamed `sm_110a` from PTX 9.0) and `sm_120a`. **`sm_121a` is not named.** The family-specific list (PTX 8.8+)
names `sm_100f`, `sm_110f` and `sm_120f` "or higher in the same family". A forum report says SM121 lacks
`cvt.e2m1x2`. The page does not say whether an `a` target of a later family member (`sm_121a`) inherits what the
family-specific `sm_120f` entry grants. **Unresolved. It decides how FP4 conversion is modelled.**
(Only `cvt ... .s2f6x2` names `sm_121a` explicitly.)

## 2. Block-scaled `mma` on `sm_121a`  (plain `mma`: clause 9.7.16.5.14; sparse `mma.sp`: clause 9.7.16.6.3)
- Plain `mma`: "Support for .kind, .block_scale, .scale_vec_size qualifier requires sm_120a and are supported on
  sm_120f or higher in the same family from PTX ISA version 8.8." **`sm_121a` is not named.**
- `mma.sp` (sparse): the same support is stated for the family "except for `.kind::mxf4nvf4`/`.kind::mxf4`", which are
  listed explicitly for `sm_120a` and `sm_121a` only.
- Same inheritance question as item 1.

## 3. Correction to spec section 16
Spec section 16 said the `.kind::mxf4nvf4`/`.kind::mxf4` qualifiers are "listed as supported on `sm_120a` and
`sm_121a`" for `mma`, and that `.acc::f16` and the `e4m3/e5m2` `mma` forms are listed for `sm_120a`/`sm_121a`. The
structured parse shows those lists belong to **`mma.sp`** (first claim) and to **`multimem.ld_reduce`** (the `.acc::f16`
and `e4m3/e5m2` lists, which are reduction operations, not `mma`). Plain `mma` documents `e4m3`/`e5m2` as
"requires sm_89 or higher". Section 16 has been corrected.

## 4. x2 packing order is documented
For `cvt` to `e2m1x2`/`e2m3x2`/`e3m2x2`/`e4m3x2`/`e5m2x2`/`ue8m0x2` from two `f32` inputs the page states the value
from input `a` goes in the **upper** half of `d`. Entries with `first_operand_upper: null` are packing statements
that do not use that wording (for example the `x4` forms and the reverse direction) and are not asserted.
