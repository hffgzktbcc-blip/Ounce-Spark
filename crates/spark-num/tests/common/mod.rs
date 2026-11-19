//! Helpers shared by the oracle integration tests.
#![allow(dead_code)]
use serde_json::Value;

pub fn read(rel: &str) -> Value {
    let path = format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel);
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}")))
        .unwrap()
}

/// Parse Python's `float.hex()` output ("0x1.8000000000000p+0", "-0x0.0p+0") plus "nan", "inf", "-inf".
pub fn parse(s: &str) -> f64 {
    match s {
        "nan" => return f64::NAN,
        "inf" => return f64::INFINITY,
        "-inf" => return f64::NEG_INFINITY,
        _ => {}
    }
    let (neg, rest) = s.strip_prefix('-').map_or((false, s), |r| (true, r));
    let rest = rest.strip_prefix("0x").expect("hex float");
    let (mant, exp) = rest.split_once('p').expect("exponent");
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let mut value = u64::from_str_radix(int, 16).unwrap() as f64;
    let mut scale = 1.0 / 16.0;
    for d in frac.chars() {
        value += d.to_digit(16).unwrap() as f64 * scale;
        scale /= 16.0;
    }
    let value = value * 2f64.powi(exp.parse::<i32>().unwrap());
    if neg {
        -value
    } else {
        value
    }
}
