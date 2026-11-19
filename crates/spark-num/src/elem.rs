//! Narrow float formats used by the Spark's PTX `cvt`/`mma`: E4M3, E5M2, E2M3, E3M2, E2M1 and UE8M0.
//!
//! Decode is exhaustive and exact. Encode is round-to-nearest-even on the bit pattern; PTX makes `.satfinite`
//! mandatory for these destination types, so `sat == false` is library-style behaviour kept for cross-checking.

/// A narrow element format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fmt {
    E4M3,
    E5M2,
    E2M3,
    E3M2,
    E2M1,
    Ue8m0,
}

/// Why an encode request has no answer.
#[derive(Debug, PartialEq, Eq)]
pub enum ElemError {
    /// UE8M0 has no round-to-nearest conversion in PTX (only `.rz`/`.rp`).
    NoRoundToNearest,
    /// The format has no code for the value (NaN, or overflow without saturation, on a format lacking NaN/Inf).
    Unrepresentable,
}

struct Layout {
    bits: u32,
    exp: u32,
    man: u32,
    bias: i32,
    max_code: u8,
}

impl Fmt {
    fn layout(self) -> Option<Layout> {
        Some(match self {
            Fmt::E4M3 => Layout {
                bits: 8,
                exp: 4,
                man: 3,
                bias: 7,
                max_code: 0x7E,
            },
            Fmt::E5M2 => Layout {
                bits: 8,
                exp: 5,
                man: 2,
                bias: 15,
                max_code: 0x7B,
            },
            Fmt::E2M3 => Layout {
                bits: 6,
                exp: 2,
                man: 3,
                bias: 1,
                max_code: 0x1F,
            },
            Fmt::E3M2 => Layout {
                bits: 6,
                exp: 3,
                man: 2,
                bias: 3,
                max_code: 0x1F,
            },
            Fmt::E2M1 => Layout {
                bits: 4,
                exp: 2,
                man: 1,
                bias: 1,
                max_code: 0x07,
            },
            Fmt::Ue8m0 => return None,
        })
    }

    fn nan_code(self) -> Option<u8> {
        match self {
            Fmt::E4M3 | Fmt::E5M2 => Some(0x7F),
            _ => None,
        }
    }
}

/// Width in bits of one lane when two values are packed into a register (`.x2` forms).
pub fn lane_bits(fmt: Fmt) -> u32 {
    if fmt == Fmt::E2M1 {
        4
    } else {
        8
    }
}

/// Decode one code. Bits above the format width are ignored. NaN codes return `f64::NAN`.
pub fn decode(fmt: Fmt, code: u8) -> f64 {
    let Some(l) = fmt.layout() else {
        return if code == 255 {
            f64::NAN
        } else {
            2f64.powi(code as i32 - 127)
        };
    };
    let code = code & (((1u16 << l.bits) - 1) as u8);
    let sign = (code >> (l.bits - 1)) & 1 == 1;
    let e = ((code >> l.man) & ((1 << l.exp) - 1)) as i32;
    let m = (code & ((1 << l.man) - 1)) as f64;
    let top = (1i32 << l.exp) - 1;
    let magnitude = match fmt {
        Fmt::E4M3 if e == top && m as u8 == (1 << l.man) - 1 => return f64::NAN,
        Fmt::E5M2 if e == top => {
            return if m == 0.0 {
                if sign {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                }
            } else {
                f64::NAN
            }
        }
        _ if e == 0 => m * 2f64.powi(1 - l.bias - l.man as i32),
        _ => (1.0 + m / (1u32 << l.man) as f64) * 2f64.powi(e - l.bias),
    };
    if sign {
        -magnitude
    } else {
        magnitude
    }
}

/// Encode with round-to-nearest-even. `sat` clamps overflow to the largest finite value.
pub fn encode_rne(fmt: Fmt, x: f64, sat: bool) -> Result<u8, ElemError> {
    let l = fmt.layout().ok_or(ElemError::NoRoundToNearest)?;
    let sign = if x.is_sign_negative() {
        1u8 << (l.bits - 1)
    } else {
        0
    };
    if x.is_nan() {
        // PTX cvt: where the destination format has NaN the result is NaN; where it has none (e2m1, e2m3, e3m2) the
        // result is positive MAX_NORM under `.satfinite`, whatever the sign of the NaN. Without `.satfinite` there is no answer.
        return match (fmt.nan_code(), sat) {
            (Some(code), _) => Ok(code),
            (None, true) => Ok(l.max_code),
            (None, false) => Err(ElemError::Unrepresentable),
        };
    }
    let a = x.abs();
    let max = decode(fmt, l.max_code);
    let half_gap = (max - decode(fmt, l.max_code - 1)) / 2.0;
    // Past the midpoint above max we overflow; exactly on it, ties-to-even decides (the code after max is even only
    // when max's code is odd: E5M2 overflows to Inf at the tie, E4M3 keeps max because the next code is NaN).
    let beyond = a > max + half_gap || (a == max + half_gap && l.max_code % 2 == 1);
    if a.is_infinite() || beyond {
        return if sat {
            Ok(sign | l.max_code)
        } else if fmt == Fmt::E5M2 {
            Ok(sign | 0x7C)
        } else {
            fmt.nan_code()
                .map(|c| sign | c)
                .ok_or(ElemError::Unrepresentable)
        };
    }
    // Largest magnitude code whose value is <= a (magnitudes increase with the code).
    let (mut lo, mut hi) = (0u8, l.max_code);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if decode(fmt, mid) <= a {
            lo = mid
        } else {
            hi = mid - 1
        }
    }
    let code = if lo == l.max_code || decode(fmt, lo) == a {
        lo
    } else {
        let midpoint = (decode(fmt, lo) + decode(fmt, lo + 1)) / 2.0;
        if a < midpoint {
            lo
        } else if a > midpoint {
            lo + 1
        } else if lo % 2 == 0 {
            lo
        } else {
            lo + 1
        }
    };
    Ok(sign | code)
}

/// Pack two element codes into one register value; `first` goes in the **upper** lane, as the PTX `cvt` text states
/// ("the value converted from input a is stored in the upper ... bits of d").
pub fn pack_x2(fmt: Fmt, first: u8, second: u8) -> u16 {
    let w = lane_bits(fmt);
    let mask = (1u16 << w) - 1;
    ((first as u16 & mask) << w) | (second as u16 & mask)
}
