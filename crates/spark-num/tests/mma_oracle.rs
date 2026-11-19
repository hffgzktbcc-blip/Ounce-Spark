//! Task 6: the Rust `mma` reference must match the committed spec-first oracles (oracles/data/mma.json).
mod common;

use common::{parse, read};
use serde_json::Value;
use spark_num::elem::{decode, Fmt};
use spark_num::mma::{
    lane_a_coords, lane_b_coords, lane_cd_coords, mma_ref_f32, mma_ref_s32, MmaShape,
};

fn shape_of(case: &Value) -> MmaShape {
    let s = case["shape"].as_array().unwrap();
    MmaShape {
        m: s[0].as_u64().unwrap() as usize,
        n: s[1].as_u64().unwrap() as usize,
        k: s[2].as_u64().unwrap() as usize,
    }
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| parse(s.as_str().unwrap()))
        .collect()
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn cases() -> Vec<Value> {
    read("oracles/data/mma.json")["cases"]
        .as_array()
        .unwrap()
        .clone()
}

/// Same value; zeros compare equal regardless of sign (the sign of a zero sum is not specified).
fn same(got: f32, want: f32) -> bool {
    (got == 0.0 && want == 0.0) || got.to_bits() == want.to_bits()
}

#[test]
fn exact_cases_bit_exact() {
    let mut checked = 0;
    for case in cases().iter().filter(|c| c["class"] == "exact") {
        let shape = shape_of(case);
        if case["acc"] == "s32" {
            let (a, b, c, want) = (
                ints(&case["a"]),
                ints(&case["b"]),
                ints(&case["c"]),
                ints(&case["expected"]),
            );
            let a: Vec<i8> = a.iter().map(|&x| x as i8).collect();
            let b: Vec<i8> = b.iter().map(|&x| x as i8).collect();
            let c: Vec<i32> = c.iter().map(|&x| x as i32).collect();
            let got = mma_ref_s32(shape, &a, &b, &c);
            assert_eq!(
                got.iter().map(|&x| x as i64).collect::<Vec<_>>(),
                want,
                "{}",
                case["id"]
            );
        } else {
            let (a, b) = (floats(&case["a"]), floats(&case["b"]));
            let c: Vec<f32> = floats(&case["c"]).iter().map(|&x| x as f32).collect();
            let want: Vec<f32> = floats(&case["expected"])
                .iter()
                .map(|&x| x as f32)
                .collect();
            let got = mma_ref_f32(shape, &a, &b, &c);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    same(*g, *w),
                    "{} element {i}: got {g}, want {w}",
                    case["id"]
                );
            }
        }
        checked += 1;
    }
    assert!(checked >= 60, "only {checked} exact cases");
}

#[test]
fn banded_cases_within_band() {
    let mut checked = 0;
    for case in cases().iter().filter(|c| c["class"] == "banded") {
        let (a, b) = (floats(&case["a"]), floats(&case["b"]));
        let c: Vec<f32> = floats(&case["c"]).iter().map(|&x| x as f32).collect();
        let (want, band) = (floats(&case["expected"]), floats(&case["band"]));
        let got = mma_ref_f32(shape_of(case), &a, &b, &c);
        for i in 0..got.len() {
            let err = (got[i] as f64 - want[i]).abs();
            assert!(
                err <= band[i],
                "{} element {i}: error {err:e} exceeds band {:e}",
                case["id"],
                band[i]
            );
        }
        checked += 1;
    }
    assert!(checked >= 60, "only {checked} banded cases");
}

fn f16_to_f64(bits: u16) -> f64 {
    let (sign, exp, man) = (
        (bits >> 15) & 1,
        ((bits >> 10) & 0x1F) as i32,
        (bits & 0x3FF) as f64,
    );
    let v = if exp == 0 {
        man * 2f64.powi(-24)
    } else {
        (1.0 + man / 1024.0) * 2f64.powi(exp - 15)
    };
    if sign == 1 {
        -v
    } else {
        v
    }
}

fn bf16_to_f64(bits: u16) -> f64 {
    let (sign, exp, man) = (
        (bits >> 15) & 1,
        ((bits >> 7) & 0xFF) as i32,
        (bits & 0x7F) as f64,
    );
    let v = if exp == 0 {
        man * 2f64.powi(-133)
    } else {
        (1.0 + man / 128.0) * 2f64.powi(exp - 127)
    };
    if sign == 1 {
        -v
    } else {
        v
    }
}

#[test]
fn input_bits_decode_to_the_stated_values() {
    for case in cases() {
        let fmt = case["a_fmt"].as_str().unwrap();
        for (vals, bits) in [("a", "a_bits"), ("b", "b_bits")] {
            let bits: Vec<u64> = case[bits]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect();
            if fmt == "s8" {
                let want = ints(&case[vals]);
                assert!(
                    bits.iter()
                        .zip(&want)
                        .all(|(&b, &w)| (b as u8) as i8 as i64 == w),
                    "{}",
                    case["id"]
                );
                continue;
            }
            let want = floats(&case[vals]);
            for (b, w) in bits.iter().zip(&want) {
                let got = match fmt {
                    "f16" => f16_to_f64(*b as u16),
                    "bf16" => bf16_to_f64(*b as u16),
                    "e4m3" => decode(Fmt::E4M3, *b as u8),
                    "e5m2" => decode(Fmt::E5M2, *b as u8),
                    other => panic!("unknown format {other}"),
                };
                assert_eq!(
                    got.to_bits(),
                    w.to_bits(),
                    "{} {fmt} bits {b:#x}",
                    case["id"]
                );
            }
        }
    }
}

const SHAPE_1: MmaShape = MmaShape { m: 2, n: 2, k: 2 };

#[test]
fn nan_input_propagates_only_to_the_affected_row() {
    let got = mma_ref_f32(SHAPE_1, &[1.0, f64::NAN, 1.0, 1.0], &[1.0; 4], &[0.0; 4]);
    assert!(got[0].is_nan() && got[1].is_nan());
    assert_eq!((got[2], got[3]), (2.0, 2.0));
}

#[test]
fn inf_minus_inf_is_nan() {
    let got = mma_ref_f32(
        SHAPE_1,
        &[f64::INFINITY, f64::NEG_INFINITY, 1.0, 1.0],
        &[1.0; 4],
        &[0.0; 4],
    );
    assert!(got[0].is_nan() && got[1].is_nan());
}

#[test]
fn all_zero_inputs_return_c() {
    let c = [1.5f32, -2.5, 3.0, 0.125];
    assert_eq!(mma_ref_f32(SHAPE_1, &[0.0; 4], &[0.0; 4], &c), c.to_vec());
}

#[test]
fn empty_k_returns_c() {
    let c = [1.5f32, -2.5, 3.0, 0.125];
    assert_eq!(
        mma_ref_f32(MmaShape { m: 2, n: 2, k: 0 }, &[], &[], &c),
        c.to_vec()
    );
}

#[test]
fn s32_accumulate_wraps_without_satfinite() {
    // PTX: "If .satfinite is not specified, the accumulated value is wrapped instead."
    let got = mma_ref_s32(MmaShape { m: 1, n: 1, k: 1 }, &[1], &[1], &[i32::MAX]);
    assert_eq!(got, vec![i32::MIN]);
}

#[test]
#[should_panic(expected = "length")]
fn wrong_input_length_is_a_loud_error() {
    mma_ref_f32(SHAPE_1, &[1.0; 3], &[1.0; 4], &[0.0; 4]);
}

const K8: MmaShape = MmaShape { m: 16, n: 8, k: 8 };
const K16: MmaShape = MmaShape { m: 16, n: 8, k: 16 };

#[test]
fn lane_coords_cover_each_matrix_element_exactly_once() {
    for shape in [K8, K16] {
        let (mut a, mut b, mut c) = (
            vec![0u32; shape.m * shape.k],
            vec![0u32; shape.k * shape.n],
            vec![0u32; shape.m * shape.n],
        );
        for lane in 0..32 {
            for (r, col) in lane_a_coords(shape, lane).unwrap() {
                a[r * shape.k + col] += 1;
            }
            for (r, col) in lane_b_coords(shape, lane).unwrap() {
                b[r * shape.n + col] += 1;
            }
            for (r, col) in lane_cd_coords(lane) {
                c[r * shape.n + col] += 1;
            }
        }
        assert!(a.iter().chain(&b).chain(&c).all(|&n| n == 1), "{shape:?}");
    }
}

#[test]
fn lane_coords_match_the_ptx_formulas() {
    // Formulas read from the PTX ISA "Matrix Fragments for mma.m16n8k8 / m16n8k16" sections (inventory/OPEN_QUESTIONS.md
    // cites the page). g = laneid >> 2, t = laneid % 4.
    for lane in 0..32usize {
        let (g, t) = (lane >> 2, lane % 4);
        let a8: Vec<_> = (0..4)
            .map(|i| (if i < 2 { g } else { g + 8 }, t * 2 + (i & 1)))
            .collect();
        assert_eq!(lane_a_coords(K8, lane).unwrap(), a8);
        let a16: Vec<_> = (0..8)
            .map(|i| {
                (
                    if i < 2 || (4..6).contains(&i) {
                        g
                    } else {
                        g + 8
                    },
                    t * 2 + (i & 1) + if i >= 4 { 8 } else { 0 },
                )
            })
            .collect();
        assert_eq!(lane_a_coords(K16, lane).unwrap(), a16);
        let b8: Vec<_> = (0..2).map(|i| (t * 2 + i, g)).collect();
        assert_eq!(lane_b_coords(K8, lane).unwrap(), b8);
        let b16: Vec<_> = (0..4)
            .map(|i| (t * 2 + (i & 1) + if i >= 2 { 8 } else { 0 }, g))
            .collect();
        assert_eq!(lane_b_coords(K16, lane).unwrap(), b16);
        let cd: Vec<_> = (0..4)
            .map(|i| (if i < 2 { g } else { g + 8 }, t * 2 + (i & 1)))
            .collect();
        assert_eq!(lane_cd_coords(lane).to_vec(), cd);
    }
}

#[test]
fn lane_layouts_exist_only_for_the_two_f16_shapes() {
    assert!(lane_a_coords(MmaShape { m: 16, n: 8, k: 32 }, 0).is_none());
    assert!(lane_b_coords(MmaShape { m: 16, n: 8, k: 32 }, 0).is_none());
}
