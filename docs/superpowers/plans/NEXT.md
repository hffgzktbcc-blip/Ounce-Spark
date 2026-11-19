# After Plan A: what is next, and what Plan A found wrong with itself

Status at the end of Plan A (branch `plan-a`, 9 tasks, local repository only: no remote exists, so "merged" means committed here).

## Plan B: Gate 2, the functional compute path (basis: decision `fork` in `docs/decisions/0001-ptx-executor.md`)
- Generalise the VM's `mma` instruction from one typed variant to shape and type fields; add the other oracle combinations
  (`m16n8k8` and `m16n8k16` f16, bf16, e4m3/e5m2 at `m16n8k16` and `m16n8k32`, s8) and lane layouts for the `k32` types.
- Instruction coverage the kernel corpus needs and ptoxide lacks: shift-right, `mul.hi`, most `cvt` pairs, f16 arithmetic, `shfl.sync`,
  atomics, `ex2`/`rcp`/`sqrt`/`rsqrt` with the `.approx` error bounds from the inventory (`error_bound` entries); exercise `ld/st.shared`
  and `bar.sync` in kernels.
- The hand-written PTX kernel corpus (GEMM per Gate 1 format, reduction, softmax, layernorm, shared-memory transpose, atomics, async copy)
  with a sizes table, an input manifest (zeros, subnormals, max, NaN/Inf, fixed seed) and numpy references committed **before** the
  emulator runs.
- Memcheck-style checks (out of bounds, shared-memory races, uninitialised reads); one tiny MXFP4/NVFP4 quantised GEMM validated against
  torchao/ml_dtypes.
- Verify the packed `.f16x2` element order against the PTX figure (currently an assumption), then resolve it in `docs/LIMITS.md`.

## Plan C: Gates 3b/3c and 4b, plus the oracle gaps
- Gate 3b outside-tester harness: a JSON schema, a run on the CI emulator, and a "does `ptxas` accept this PTX" check for owners (decision D1).
- Gate 3c: apply for free hardware access (NVIDIA LaunchPad, partner programs); approval is never assumed.
- More data: the other field-notes rows, Spark Arena (needs `spark-arena.com` allowed in the environment), the JetsonHacks table (its URL
  returned 404), and a raw-page check of the llama.cpp rows (github.com returned HTTP 403). The noise floor is **not yet estimable**
  (docs/LIMITS.md).
- Gate 4b: per-stack, per-format calibrated roofline, leave-one-model-out, sample-size condition (spec section 9.3, amended by D3).
- Oracles still missing: TF32, block-scaled FP4/FP6 `mma` (read CUTLASS for scale layouts), `ue8m0` encode (`.rz`/`.rp`), the `x4` pack
  forms, `mma.sp` forms, `.satfinite` integer accumulate, f16 accumulate.

## Open owner items
- **Gate 0b is not ticked.** `inventory/OPEN_QUESTIONS.md` asks the owner to confirm visually: does `sm_121a` support `cvt` to/from `e2m1x2`,
  and is plain block-scaled `mma` documented for `sm_121a`? (The inventory shows the page names `sm_120a` and a family `sm_120f` but not `sm_121a`.)
- Creating a GitHub remote, and the NVIDIA licence reading behind decision D1, are the owner's calls.

## Defects Plan A's execution found in the plan itself
1. Task 8 Step 3 said to run ptoxide's own tests, which execute `nvcc`-built kernels: a breach of decision D1 (disclosed in the decision record).
2. Task 2 recorded only "requires sm_X" statements, not the instruction forms, so Task 5 could not honestly refuse unlisted combinations;
   forms were added to the extractor during Task 5.
3. Task 2's real-page assertion (block-scale `mma` listing `sm_121a`) was a misattributed flattened-text reading; it belongs to `mma.sp`.
4. Task 5's band `K * 2^-24 * (...)` was not an upper bound; replaced by gamma_{K+2}.
5. Task 4's signatures needed `Result` (ue8m0, NaN) and `u16` (x2 packs of 8-bit formats).
6. Task 3 pinned one NaN code; the spec fixes none, so the oracle now records "result is NaN".
7. Task 1's HTML guard flagged its own source; Task 7's `spec_decoding` needed a third value; llama.cpp rows are `secondary`, not raw-verified.
