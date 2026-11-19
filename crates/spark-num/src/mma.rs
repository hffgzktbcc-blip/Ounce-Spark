//! Reference semantics of `mma.sync` for the combinations covered by `oracles/data/mma.json`.
//!
//! Accumulation order and internal precision are unspecified by PTX (it requires "at least single precision"), so
//! this reference accumulates in `f32` in increasing `k` starting from `c`; the oracle bands bound any such order.
//!
//! The per-lane fragment layout functions are ported from `ounce-core` (`src/gpu/tensor.rs`, MIT, Copyright (c) 2026
//! hffgzktbcc-blip) and checked against the formulas in NVIDIA's public PTX ISA text (see the tests).

/// A matrix-multiply-accumulate shape: `A` is `m x k`, `B` is `k x n`, `C` and `D` are `m x n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmaShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

fn check_lengths(shape: MmaShape, a: usize, b: usize, c: usize) {
    assert_eq!(a, shape.m * shape.k, "A length must be m*k");
    assert_eq!(b, shape.k * shape.n, "B length must be k*n");
    assert_eq!(c, shape.m * shape.n, "C length must be m*n");
}

/// `D = A * B + C` with f32 accumulation. `a` is row-major `m x k`, `b` is row-major `k x n`, `c` is row-major
/// `m x n`; inputs are already decoded to exact values. Products are rounded to f32 (exact for the oracle formats).
/// Panics if a slice has the wrong length (a programmer error, not a data error).
pub fn mma_ref_f32(shape: MmaShape, a: &[f64], b: &[f64], c: &[f32]) -> Vec<f32> {
    check_lengths(shape, a.len(), b.len(), c.len());
    let mut d = Vec::with_capacity(shape.m * shape.n);
    for i in 0..shape.m {
        for j in 0..shape.n {
            let mut acc = c[i * shape.n + j];
            for t in 0..shape.k {
                acc += (a[i * shape.k + t] * b[t * shape.n + j]) as f32;
            }
            d.push(acc);
        }
    }
    d
}

/// Integer `D = A * B + C` with s32 accumulation. Without `.satfinite` the accumulator wraps (PTX: "If .satfinite is
/// not specified, the accumulated value is wrapped instead"); the saturating form is not implemented yet.
pub fn mma_ref_s32(shape: MmaShape, a: &[i8], b: &[i8], c: &[i32]) -> Vec<i32> {
    check_lengths(shape, a.len(), b.len(), c.len());
    let mut d = Vec::with_capacity(shape.m * shape.n);
    for i in 0..shape.m {
        for j in 0..shape.n {
            let mut acc = c[i * shape.n + j];
            for t in 0..shape.k {
                acc = acc.wrapping_add(a[i * shape.k + t] as i32 * b[t * shape.n + j] as i32);
            }
            d.push(acc);
        }
    }
    d
}

/// `Some(true)` for `m16n8k16`, `Some(false)` for `m16n8k8`, `None` for every other shape (layouts not implemented).
fn f16_shape_is_k16(shape: MmaShape) -> Option<bool> {
    match (shape.m, shape.n, shape.k) {
        (16, 8, 8) => Some(false),
        (16, 8, 16) => Some(true),
        _ => None,
    }
}

fn group_and_tid(lane: usize) -> (usize, usize) {
    let lane = lane % 32;
    (lane >> 2, lane % 4)
}

/// `(row, col)` of each `A` fragment register held by `lane`, in register order (`A` row-major `m x k`).
/// Only `m16n8k8` and `m16n8k16` f16 layouts exist; other shapes return `None`.
pub fn lane_a_coords(shape: MmaShape, lane: usize) -> Option<Vec<(usize, usize)>> {
    let k16 = f16_shape_is_k16(shape)?;
    let (group, tid) = group_and_tid(lane);
    let col = tid * 2;
    let mut coords = vec![
        (group, col),
        (group, col + 1),
        (group + 8, col),
        (group + 8, col + 1),
    ];
    if k16 {
        coords.extend([
            (group, col + 8),
            (group, col + 9),
            (group + 8, col + 8),
            (group + 8, col + 9),
        ]);
    }
    Some(coords)
}

/// `(row, col)` of each `B` fragment register held by `lane`, in register order (`B` is `k x n`).
pub fn lane_b_coords(shape: MmaShape, lane: usize) -> Option<Vec<(usize, usize)>> {
    let k16 = f16_shape_is_k16(shape)?;
    let (group, tid) = group_and_tid(lane);
    let row = tid * 2;
    let mut coords = vec![(row, group), (row + 1, group)];
    if k16 {
        coords.extend([(row + 8, group), (row + 9, group)]);
    }
    Some(coords)
}

/// `(row, col)` of each `C`/`D` fragment register held by `lane` (identical for every `m16n8` shape).
pub fn lane_cd_coords(lane: usize) -> [(usize, usize); 4] {
    let (group, tid) = group_and_tid(lane);
    let col = tid * 2;
    [
        (group, col),
        (group, col + 1),
        (group + 8, col),
        (group + 8, col + 1),
    ]
}
