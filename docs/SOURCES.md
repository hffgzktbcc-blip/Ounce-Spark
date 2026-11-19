# Allowed and forbidden sources

## Allowed
- `ounce-core` (owner's MIT code), Mesa NAK and envytools (MIT, notices in `THIRD_PARTY.md`), and other
  MIT/Apache-2.0 sources, each recorded in `docs/PROVENANCE.md`.
- NVIDIA's public documentation, read for facts and citations (HTML not committed).
- Public benchmark data with source URL, build, kernel, stack and date kept.

## Never used
- `ounce-runtime` and `oboromi` (GPL-3.0) code or data.
- Anything from private `ounce-nx2`.
- Retail content, firmware, keys.
- Any NVIDIA toolchain output: no `nvcc`, `ptxas` or `cuobjdump`, and no NVIDIA-built output executed,
  disassembled or committed (decision D1).

## Read-for-orientation only (licence not recorded)
CuAssembler, turingas, DocumentSASS, SASSquatch.

Not legal advice.
