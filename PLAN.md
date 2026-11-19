# Gate checklist

Rule: a gate is **done** only when its work is merged, CI is green, and its box is ticked here.
Spec: `docs/superpowers/specs/2026-11-18-ounce-spark-design.md` (section 9.3).

- [ ] Gate 0b: PTX ISA inventory (Plan A, Task 2)
- [ ] Gate 1: spec-first numeric oracles (Plan A, Tasks 3-6) — Plan A scope done; TF32 and block-scaled FP4/FP6 remain (Plan C)
  - [x] element formats (Tasks 3-4): oracles + Rust codecs bit-exact; label spec-verified, scope in docs/LIMITS.md
  - [x] `mma` for FP16, BF16, FP8, INT8 (Tasks 5-6): oracles + Rust reference; label spec-verified, scope in docs/LIMITS.md
- [x] Gate 2 spike: ptoxide with one `mma` shape (Plan A, Task 8): decision **fork**, see docs/decisions/0001-ptx-executor.md
- [ ] Gate 2: functional compute path (Plan B)
- [x] Gate 3a/3d: calibration dataset and noise-floor tool (Plan A, Task 7): 15 readings; the noise floor itself is not yet estimable, see docs/LIMITS.md
- [ ] Gate 3b/3c: outside-tester harness, free hardware access (Plan C)
- [ ] Gate 4b: calibrated performance model (Plan C)

Gate 4a (zero-parameter roofline) is done: see spec section 15. Gate 0a is cut (decision D1).
