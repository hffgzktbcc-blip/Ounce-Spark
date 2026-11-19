#!/usr/bin/env python3
"""Spec-first element-format oracles (Task 3, Gate 1a).

Exhaustive decode tables and ties-to-even encode cases for the 4/6/8-bit formats the Spark's PTX lists, computed
with TWO independent libraries (gfloat, ml_dtypes) that must agree. Output is committed before any Rust consumer exists.

Scope notes (see docs/LIMITS.md): ue8m0 has decode only (PTX offers just .rz/.rp for it, and the libraries disagree on RN);
PTX makes .satfinite mandatory for every narrow destination type, so non-saturating cases are library semantics
(`ptx_valid: false`). NaN/Inf inputs under .satfinite follow the rules the PTX cvt text states (`basis: ptx_cvt_text`).
"""
import importlib.metadata
import json
import math
import pathlib
import re
import sys

import gfloat
import gfloat.formats as F
import ml_dtypes as md
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import hexf, write_manifest  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "inventory" / "ptx_sm12x.json"
KNOWN = ROOT / "oracles" / "KNOWN_DISAGREEMENTS.md"

# name -> (bits, gfloat FormatInfo, ml_dtypes type)
FORMATS = {
    "e4m3": (8, F.format_info_ocp_e4m3, md.float8_e4m3fn),
    "e5m2": (8, F.format_info_ocp_e5m2, md.float8_e5m2),
    "e2m3": (6, F.format_info_ocp_e2m3, md.float6_e2m3fn),
    "e3m2": (6, F.format_info_ocp_e3m2, md.float6_e3m2fn),
    "e2m1": (4, F.format_info_ocp_e2m1, md.float4_e2m1fn),
    "ue8m0": (8, F.format_info_ocp_e8m0, md.float8_e8m0fnu),
}
ENCODE_FORMATS = ["e4m3", "e5m2", "e2m3", "e3m2", "e2m1"]


def known_keys() -> set:
    if not KNOWN.exists():
        return set()
    return set(re.findall(r"^- `([^`]+)`", KNOWN.read_text(), flags=re.M))


def check_inventory(path: pathlib.Path) -> None:
    inv = json.loads(pathlib.Path(path).read_text())
    text = " ".join(e["statement"] for e in inv["instructions"] if e["name"] in ("cvt", "mma"))
    for fmt in FORMATS:
        if fmt not in text:
            raise ValueError(f"format {fmt} is not mentioned by any cvt/mma statement in the PTX inventory")


def ml_decode(fmt: str, code: int) -> float:
    return float(np.array([code], dtype=np.uint8).view(FORMATS[fmt][2])[0])


def ml_encode(fmt: str, x: float) -> int:
    return int(np.array([x], dtype=np.float64).astype(FORMATS[fmt][2]).view(np.uint8)[0])


def same(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b and math.copysign(1, a) == math.copysign(1, b)


def build(inventory_path: pathlib.Path = DEFAULT_INVENTORY) -> dict:
    check_inventory(inventory_path)
    known = known_keys()
    unknown, listed = [], []
    formats, decoded = {}, {}
    for name, (bits, fi, _) in FORMATS.items():
        values = []
        for code in range(1 << bits):
            g = gfloat.decode_float(fi, code).fval
            m = ml_decode(name, code)
            values.append(g)
            if not same(g, m):
                key = f"decode|{name}|{code}"
                (listed if key in known else unknown).append({"key": key, "fmt": name, "code": code, "gfloat": hexf(g), "ml_dtypes": hexf(m)})
        finite = [v for v in values if math.isfinite(v)]
        decoded[name] = values
        formats[name] = {
            "bits": bits,
            "decode": [hexf(v) for v in values],
            "max": max(finite),
            "has_inf": any(math.isinf(v) for v in values),
            "has_nan": any(math.isnan(v) for v in values),
        }
    cases = []
    for name in ENCODE_FORMATS:
        fi, vals = FORMATS[name][1], decoded[name]
        has_special = formats[name]["has_nan"]
        pos = sorted({v for v in vals if math.isfinite(v) and v > 0})
        bits = formats[name]["bits"]
        max_code = decoded[name].index(pos[-1])
        xs = [0.0, -0.0, pos[0], pos[-1], pos[-1] + (pos[-1] - pos[-2]) / 2, 1e30, -1e30, math.inf, -math.inf, math.nan]
        for a, b in zip([0.0] + pos, pos):
            xs += [(a + b) / 2, -(a + b) / 2]
            for frac in (0.25, 0.75):          # off-grid, not ties: must go to the NEARER neighbour
                xs += [a + (b - a) * frac, -(a + (b - a) * frac)]
        for x in xs:
            special = math.isnan(x) or math.isinf(x)
            for sat in (True, False):
                if not has_special and not sat:
                    continue  # PTX makes .satfinite mandatory; libraries cannot represent the overflow
                if special and not sat and not has_special:
                    continue
                if special and sat:
                    # Rules stated by the PTX cvt text (not computed by either library): NaN stays NaN where the format has
                    # NaN and becomes positive MAX_NORM where it does not; any |x| > MAX_NORM, infinity included, becomes
                    # sign-preserved MAX_NORM.
                    if math.isnan(x):
                        code, is_nan = (None, True) if has_special else (max_code, False)
                    else:
                        code, is_nan = max_code | ((1 << (bits - 1)) if x < 0 else 0), False
                    cases.append({"fmt": name, "x": hexf(x), "sat": sat, "code": code, "nan": is_nan, "ptx_valid": True, "basis": "ptx_cvt_text"})
                    continue
                code = gfloat.encode_float(fi, gfloat.round_float(fi, x, sat=sat))
                is_nan = math.isnan(gfloat.decode_float(fi, code).fval)
                in_range = math.isfinite(x) and abs(x) <= pos[-1]
                if in_range and not is_nan:
                    other = ml_encode(name, x)
                    if other != code:
                        key = f"encode|{name}|{hexf(x)}|{sat}"
                        (listed if key in known else unknown).append({"key": key, "fmt": name, "x": hexf(x), "sat": sat, "gfloat": code, "ml_dtypes": other})
                # The spec fixes no canonical NaN code (gfloat and ml_dtypes differ), so NaN results carry no code.
                cases.append({"fmt": name, "x": hexf(x), "sat": sat, "code": None if is_nan else code, "nan": is_nan, "ptx_valid": sat, "basis": "libraries"})
    if unknown:
        raise AssertionError(f"unlisted disagreements between gfloat and ml_dtypes: {unknown[:5]} (add to oracles/KNOWN_DISAGREEMENTS.md with a reason)")
    cases.sort(key=lambda c: (c["fmt"], c["sat"], float.fromhex(c["x"]) if c["x"].startswith(("0x", "-0x")) else (math.inf if c["x"] == "inf" else -math.inf if c["x"] == "-inf" else math.inf * 2), c["x"]))
    return {
        "formats": formats,
        "encode_cases": cases,
        "disagreements": {
            "decode": [d for d in listed if d["key"].startswith("decode")],
            "encode": [d for d in listed if d["key"].startswith("encode")],
        },
        "libraries": {"gfloat": importlib.metadata.version("gfloat"), "ml_dtypes": md.__version__},
    }


def main(out_dir: pathlib.Path) -> int:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "elements.json").write_text(json.dumps(build(), indent=1, sort_keys=True) + "\n")
    write_manifest(out_dir)
    print(f"wrote {out_dir}/elements.json and MANIFEST.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "oracles" / "data")))
