//! Task 4: the Rust codecs must match the committed spec-first oracles (oracles/data/elements.json) bit for bit.
mod common;

use common::{parse, read};
use serde_json::Value;
use spark_num::elem::{decode, encode_rne, pack_x2, ElemError, Fmt};

fn fmt_of(name: &str) -> Fmt {
    match name {
        "e4m3" => Fmt::E4M3,
        "e5m2" => Fmt::E5M2,
        "e2m3" => Fmt::E2M3,
        "e3m2" => Fmt::E3M2,
        "e2m1" => Fmt::E2M1,
        "ue8m0" => Fmt::Ue8m0,
        other => panic!("unknown format {other}"),
    }
}

#[test]
fn decode_matches_oracle_exhaustively() {
    let oracle = read("oracles/data/elements.json");
    for (name, f) in oracle["formats"].as_object().unwrap() {
        for (code, s) in f["decode"].as_array().unwrap().iter().enumerate() {
            let want = parse(s.as_str().unwrap());
            let got = decode(fmt_of(name), code as u8);
            if want.is_nan() {
                assert!(got.is_nan(), "{name} code {code}: want NaN, got {got}");
            } else {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{name} code {code}: want {want}, got {got}"
                );
            }
        }
    }
}

#[test]
fn encode_matches_oracle_cases() {
    let oracle = read("oracles/data/elements.json");
    let cases = oracle["encode_cases"].as_array().unwrap();
    assert!(cases.len() > 1000);
    for c in cases {
        let (name, x, sat) = (
            c["fmt"].as_str().unwrap(),
            parse(c["x"].as_str().unwrap()),
            c["sat"].as_bool().unwrap(),
        );
        let fmt = fmt_of(name);
        let got = encode_rne(fmt, x, sat)
            .unwrap_or_else(|e| panic!("{name} x={} sat={sat}: {e:?}", c["x"]));
        if c["nan"].as_bool().unwrap() {
            // The spec fixes no canonical NaN code: any NaN code is correct.
            assert!(
                decode(fmt, got).is_nan(),
                "{name} x={} sat={sat}: code {got:#x} is not NaN",
                c["x"]
            );
        } else {
            assert_eq!(
                got,
                c["code"].as_u64().unwrap() as u8,
                "{name} x={} sat={sat}",
                c["x"]
            );
        }
    }
}

#[test]
fn e2m1_has_no_inf_or_nan_codes() {
    assert!((0..16u8).all(|c| decode(Fmt::E2M1, c).is_finite()));
}

#[test]
fn negative_zero_survives() {
    let z = decode(Fmt::E2M1, 0b1000);
    assert!(z == 0.0 && z.is_sign_negative());
    assert_eq!(encode_rne(Fmt::E2M1, -0.0, true), Ok(0b1000));
}

#[test]
fn satfinite_clamps_overflow() {
    assert_eq!(encode_rne(Fmt::E4M3, 1e30, true), Ok(0x7E));
    assert_eq!(decode(Fmt::E4M3, 0x7E), 448.0);
    assert_eq!(encode_rne(Fmt::E2M1, -1e30, true), Ok(0b1111));
}

#[test]
fn ue8m0_has_no_round_to_nearest() {
    assert_eq!(
        encode_rne(Fmt::Ue8m0, 1.0, true),
        Err(ElemError::NoRoundToNearest)
    );
}

#[test]
fn nan_under_satfinite_becomes_positive_max_norm_where_the_format_has_no_nan() {
    // PTX cvt: "For .e2m1x2, .e2m3x2, .e3m2x2 ... destination formats NaN results are converted to positive MAX_NORM."
    for (fmt, max_code) in [(Fmt::E2M1, 0b0111u8), (Fmt::E2M3, 0x1F), (Fmt::E3M2, 0x1F)] {
        assert_eq!(encode_rne(fmt, f64::NAN, true), Ok(max_code), "{fmt:?}");
        assert_eq!(
            encode_rne(fmt, -f64::NAN, true),
            Ok(max_code),
            "{fmt:?}: the sign of a NaN must not leak into the result"
        );
    }
}

#[test]
fn without_satfinite_nan_and_overflow_have_no_answer_where_the_format_has_no_nan() {
    assert_eq!(
        encode_rne(Fmt::E2M1, f64::NAN, false),
        Err(ElemError::Unrepresentable)
    );
    assert_eq!(
        encode_rne(Fmt::E2M1, 1e30, false),
        Err(ElemError::Unrepresentable)
    );
}

#[test]
fn nan_stays_nan_where_the_format_has_nan() {
    for fmt in [Fmt::E4M3, Fmt::E5M2] {
        for sat in [true, false] {
            let code = encode_rne(fmt, f64::NAN, sat).unwrap();
            assert!(decode(fmt, code).is_nan(), "{fmt:?} sat={sat}");
        }
    }
}

#[test]
fn pack_x2_order_matches_inventory() {
    let inv = read("inventory/ptx_sm12x.json");
    let packs: Vec<&Value> = inv["instructions"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["kind"] == "packing" && e["name"] == "cvt")
        .collect();
    for (fmt, token, first, second, want) in [
        (Fmt::E2M1, "e2m1x2", 0x3, 0x1, 0x31u16),
        (Fmt::E4M3, "e4m3x2", 0x3, 0x1, 0x0301),
        (Fmt::E5M2, "e5m2x2", 0x3, 0x1, 0x0301),
        (Fmt::E2M3, "e2m3x2", 0x3, 0x1, 0x0301),
        (Fmt::E3M2, "e3m2x2", 0x3, 0x1, 0x0301),
        (Fmt::Ue8m0, "ue8m0x2", 0x3, 0x1, 0x0301),
    ] {
        let documented = packs.iter().any(|e| {
            e["statement"].as_str().unwrap().contains(token) && e["first_operand_upper"] == true
        });
        assert!(
            documented,
            "inventory does not document first-operand-upper for {token}"
        );
        assert_eq!(pack_x2(fmt, first, second), want, "{token}");
    }
}
