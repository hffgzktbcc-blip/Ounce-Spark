---
decision: fork
hours_spent: 0.13
ptoxide_commit: 522cde29551cdb66a8a8456a36da21bae57d72fe
licence_checked: true
---

# 0001: build the PTX executor by forking ptoxide

Date: 2026-11-19. Spike: Plan A, Task 8 (spec section 9.3, "Gate 2 spike"). **Decision: fork ptoxide**
(<https://github.com/gvilums/ptoxide>, MIT OR Apache-2.0). The fallback `numerics-only` was not needed.

## What was measured

One hand-written PTX kernel (`spikes/ptoxide-mma/kernels/mma_m16n8k16_f16.ptx`) runs one warp through the fork and executes
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. It matches the 8 `exact` oracle cases bit for bit and the 12 `banded` cases
within their band (`oracles/data/mma.json`). Three failure paths are tested and each is caught by a mutation: a block smaller than a
warp, an unsupported type combination, and a lane that exits before the collective. Reproduced on a pristine clone at the pinned
commit with `mma.patch` (`spikes/ptoxide-mma/README.md`).

The change is 306 inserted lines in 6 files (of which 131 are tests): lexer 12, parser 33, compiler 32, VM 94, `Cargo.toml` 6.

## Hours

Wall-clock from the first clone (08:34:33 UTC) to the passing, mutation-checked, reproduced result (about 08:43) is about 8 minutes,
recorded as `hours_spent: 0.13`. That is the executing agent's wall-clock, not a human estimate, so the spec's 16-hour threshold
is a weak proxy here: it would have been a poor test of a human builder's cost. The comparison that matters is the shape of the work:
no part of ptoxide had to be redesigned. A warp collective reuses ptoxide's existing "park a thread until a rendezvous completes"
barrier mechanism, and lane layouts and the reference arithmetic come from `spark-num`.

## Fit of ptoxide's parser to hand-written PTX (what worked unmodified)
`.version/.target/.address_size`, `.entry` with `.param`, `.reg` vectors (`%r<16>`), `ld.param.u64`, `cvta.to.global.u64`,
`mov.u32 %r, %tid.x`, `and.b32`, `sub.u32`, `shl.b32`, `add.u32/.u64`, `cvt.u64.u32`, `ld.global.u32/.f32` with `[reg+imm]`,
`st.global.f32`, `setp` plus a predicated `bra` (used by the divergence test), `ret`. Needed new: three qualifier tokens, a
brace-enclosed register vector operand, and the instruction itself.

## Known gaps and risks (these are Plan B's work, not reasons to avoid the fork)
- ptoxide is a learning project: 32 `todo!()`s, a 4k-line codebase. Missing for our Gate 2 kernel corpus: shift-right, `mul.hi`,
  most `cvt` pairs, f16 arithmetic, `shfl.sync`, atomics, `ex2`/`rcp`/`sqrt` approximations; shared memory and `bar.sync` exist but were not
  exercised here.
- The new VM instruction is typed per shape (`MmaSyncM16N8K16F16F32`); Plan B should generalise to shape and type fields.
- The VM runs threads one at a time with a rendezvous per warp: no divergence/reconvergence model (a lane that exits early is reported, not
  simulated), matching the design's "unmodelled divergence stops the run".
- **Assumption, not verified:** in a packed `.f16x2` fragment register the lower-index element is in bits 15:0. The PTX text states only that a
  register "contains two .f16 elements"; the figure was not read. The kernel's own little-endian loads and the oracle agree with this, but
  hardware agreement is unknown.
- Hand-written PTX has no `ptxas` check; the kernel is "valid" only by the PTX documentation (decision D1).

## Licence
`Cargo.toml` says `MIT OR Apache-2.0`; both `LICENSE-MIT` and `LICENSE-APACHE` are present in the clone. Recorded in `docs/PROVENANCE.md`.
Only `mma.patch` (our additions plus a few lines of context) is committed; the clone is git-ignored.

## Incident disclosure (decision D1)
Plan Task 8 Step 3 said to run ptoxide's own tests. Those tests execute kernels that were produced by NVIDIA's `nvcc`, which D1 forbids
(no NVIDIA-built output executed or disassembled). They were run once (18 passed) and one such kernel was printed. Nothing was committed
or used for any result; the clone's `kernels/`, `tests/` and `examples/` were then deleted and every kernel and test used afterwards is
hand-written. The plan step was the defect and should be corrected.
