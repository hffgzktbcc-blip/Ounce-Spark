# Gate 2 spike: ptoxide with one `mma` shape

Question (spec section 9.3): can ptoxide, an MIT OR Apache-2.0 PTX interpreter, be extended with `mma.sync` cheaply enough to fork it?
Answer: yes, see `docs/decisions/0001-ptx-executor.md`.

## Contents
- `kernels/mma_m16n8k16_f16.ptx`: hand-written PTX (no NVIDIA toolchain). One warp, `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
- `mma.patch`: our changes against ptoxide commit `522cde29551cdb66a8a8456a36da21bae57d72fe` (lexer, parser, compiler, VM, a path dependency on
  `crates/spark-num` for lane layouts and the reference arithmetic, and `tests/mma_spike.rs`). The patch contains a few lines of ptoxide context.
  ptoxide is dual-licensed MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE` in its repository).
- `vendor/` (git-ignored, excluded from the Cargo workspace): the working clone.

## Reproduce
```sh
cd spikes/ptoxide-mma
git clone https://github.com/gvilums/ptoxide vendor
git -C vendor checkout 522cde29551cdb66a8a8456a36da21bae57d72fe
git -C vendor apply ../mma.patch
cd vendor && cargo test --test mma_spike     # run ONLY this test (see the warning below)
```
Expected: 5 passed (8 exact cases bit-exact, 12 banded cases in band, 3 failure-path tests).

**Warning (decision D1):** ptoxide's own `kernels/`, `tests/test.rs` and `examples/` are built from NVIDIA `nvcc` output. Do not run
`cargo test` without `--test mma_spike`, and delete those directories from your clone, as this project did.

## Assumption
A packed `.f16x2` fragment register holds the lower-index element in bits 15:0. Not verified against the PTX figure (see the decision record).
