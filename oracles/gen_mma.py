#!/usr/bin/env python3
"""Spec-first `mma` oracles (Task 5, Gate 1b): FP16, BF16, FP8 (E4M3/E5M2) with f32 accumulate and INT8 with s32.

Each case carries the exact rational result rounded once to f32. `exact` cases use inputs for which every partial sum is
exactly representable, so ANY accumulation order gives the same bits (compare bit-exact). `banded` cases carry a tolerance
band because PTX leaves accumulation order and internal precision unspecified. Band (my derivation, standard
floating-point summation error analysis): gamma_{K+2} * (sum_k |a_k b_k| + |c|) with gamma_n = n*2^-24 / (1 - n*2^-24)
(K products plus c, plus the final rounding). Combinations are refused unless the PTX inventory lists them.
"""
import json
import math
import pathlib
import random
import sys
from fractions import Fraction

import ml_dtypes as md
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import hexf, write_manifest  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "inventory" / "ptx_sm12x.json"
ELEMENTS = ROOT / "oracles" / "data" / "elements.json"
M, N = 16, 8
COMBOS = [
    ("m16n8k8", "f16", "f32"),
    ("m16n8k16", "f16", "f32"),
    ("m16n8k16", "bf16", "f32"),
    ("m16n8k32", "e4m3", "f32"),
    ("m16n8k32", "e5m2", "f32"),
    ("m16n8k32", "s8", "s32"),
]
EXACT_POOL = [0.0, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]
MAX_SM = 121


def band_for(k: int, abs_sum: Fraction, abs_c: Fraction) -> Fraction:
    n = Fraction(k + 2, 2**24)
    return n / (1 - n) * (abs_sum + abs_c)


def fraction_to_f32(x: Fraction) -> float:
    """Round an exact rational to the nearest f32 (ties to even), returned as a Python float."""
    if x == 0:
        return 0.0
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length() - 24
    while a / Fraction(2) ** e >= 2**24:
        e += 1
    while a / Fraction(2) ** e < 2**23:
        e -= 1
    e = max(e, -149)
    q = a / Fraction(2) ** e
    n, rem = divmod(q.numerator, q.denominator)
    twice = 2 * rem
    if twice > q.denominator or (twice == q.denominator and n % 2 == 1):
        n += 1
    value = math.ldexp(n, e)
    return -value if x < 0 else value


def check_inventory(path: pathlib.Path) -> None:
    inv = json.loads(pathlib.Path(path).read_text())
    forms = [e for e in inv["instructions"] if e["kind"] == "form" and e["name"] == "mma" and not e["kinds"]]
    notes = [e for e in inv["instructions"] if e["kind"] == "target_note" and e["name"] == "mma"]
    for shape, fmt, acc in COMBOS:
        listed = any(
            shape in f["shapes"] and fmt in f["a_types"] and fmt in f["b_types"] and acc in f["c_types"] and acc in f["d_types"]
            for f in forms
        )
        if not listed:
            raise ValueError(f"mma {shape} {fmt} with {acc} accumulate is not listed by any plain mma form in the PTX inventory")
        needs = [int(e["requires"].split("_")[1].rstrip("af")) for e in notes if f".{fmt}" in e["statement"] and e["requires"]]
        if not needs or min(needs) > MAX_SM:
            raise ValueError(f"mma type {fmt}: no 'requires sm_N or higher' statement with N <= {MAX_SM} in the PTX inventory")


class Elements:
    """Representable values per input format, with their bit patterns."""

    def __init__(self):
        data = json.loads(ELEMENTS.read_text())
        self.fp8 = {}
        for name in ("e4m3", "e5m2"):
            table = [(code, float.fromhex(s)) for code, s in enumerate(data["formats"][name]["decode"]) if s not in ("nan", "inf", "-inf")]
            self.fp8[name] = table

    def encode(self, fmt: str, value: float):
        """(bits, decoded value) for a value that must be exactly representable in `fmt`."""
        if fmt == "f16":
            h = np.float16(value)
            return int(h.view(np.uint16)), float(h)
        if fmt == "bf16":
            h = np.float32(value).astype(md.bfloat16)
            return int(h.view(np.uint16)), float(h)
        if fmt == "s8":
            return int(value) & 0xFF, int(value)
        for code, v in self.fp8[fmt]:
            if v == value and math.copysign(1, v) == math.copysign(1, value):
                return code, v
        raise ValueError(f"{value} is not representable in {fmt}")

    def random_value(self, fmt: str, rnd: random.Random):
        if fmt == "f16":
            while True:
                bits = rnd.getrandbits(16)
                if (bits >> 10) & 0x1F != 0x1F:
                    return bits, float(np.uint16(bits).view(np.float16))
        if fmt == "bf16":
            while True:
                bits = rnd.getrandbits(16)
                if 100 <= (bits >> 7) & 0xFF <= 154:  # keep products and sums far from f32 overflow
                    return bits, float(np.array([bits], dtype=np.uint16).view(md.bfloat16)[0])
        if fmt == "s8":
            v = rnd.randint(-128, 127)
            return v & 0xFF, v
        return rnd.choice(self.fp8[fmt])


def exact_sum(a, b, c, m, n, k, as_int):
    out = []
    for i in range(m):
        for j in range(n):
            terms = [Fraction(a[i * k + t]) * Fraction(b[t * n + j]) for t in range(k)]
            out.append((Fraction(c[i * n + j]), terms))
    return out


def make_case(case_id, shape, fmt, acc, cls, a_pairs, b_pairs, c_vals):
    m, n, k = shape
    a, b = [v for _, v in a_pairs], [v for _, v in b_pairs]
    as_int = acc == "s32"
    expected, band = [], []
    for c, terms in exact_sum(a, b, c_vals, m, n, k, as_int):
        total = c + sum(terms, Fraction(0))
        if as_int:
            expected.append(int(total))
            band.append(0)
        else:
            expected.append(hexf(fraction_to_f32(total)))
            band.append(hexf(float(band_for(k, sum((abs(t) for t in terms), Fraction(0)), abs(c)))) if cls == "banded" else hexf(0.0))
    conv = (lambda v: v) if as_int else hexf
    return {
        "id": case_id, "shape": [m, n, k], "a_fmt": fmt, "b_fmt": fmt, "acc": acc, "class": cls,
        "a": [conv(v) for v in a], "a_bits": [bt for bt, _ in a_pairs],
        "b": [conv(v) for v in b], "b_bits": [bt for bt, _ in b_pairs],
        "c": [conv(v) for v in c_vals], "expected": expected, "band": band,
    }


def build(seed: int = 20261118, inventory_path: pathlib.Path = DEFAULT_INVENTORY) -> dict:
    check_inventory(inventory_path)
    elems = Elements()
    cases = []
    for shape_name, fmt, acc in COMBOS:
        k = int(shape_name.split("k")[1])
        shape = (M, N, k)
        rnd = random.Random(f"{seed}-{shape_name}-{fmt}")
        as_int = acc == "s32"
        n_exact, n_banded = (20, 0) if as_int else (8, 12)
        for idx in range(n_exact + n_banded):
            cls = "exact" if idx < n_exact else "banded"
            if cls == "exact" and as_int:
                pick = lambda: elems.random_value(fmt, rnd)  # noqa: E731
                if idx == 0:
                    pick = lambda: elems.encode(fmt, -128)  # noqa: E731  (largest magnitude products)
                a_pairs = [pick() for _ in range(M * k)]
                b_pairs = [pick() for _ in range(k * N)]
                c_vals = [rnd.randint(-1000, 1000) for _ in range(M * N)]
            elif cls == "exact":
                pool = (lambda: elems.encode(fmt, rnd.choice(EXACT_POOL)))
                if idx == 0:   # zeros times anything, c carries the result
                    a_pairs, b_pairs = [elems.encode(fmt, 0.0)] * (M * k), [elems.encode(fmt, 1.0)] * (k * N)
                elif idx == 1:  # all ones: result is k
                    a_pairs, b_pairs = [elems.encode(fmt, 1.0)] * (M * k), [elems.encode(fmt, 1.0)] * (k * N)
                else:
                    a_pairs, b_pairs = [pool() for _ in range(M * k)], [pool() for _ in range(k * N)]
                c_vals = [rnd.choice(EXACT_POOL) for _ in range(M * N)]
            else:
                a_pairs = [elems.random_value(fmt, rnd) for _ in range(M * k)]
                b_pairs = [elems.random_value(fmt, rnd) for _ in range(k * N)]
                c_vals = [float(np.float32(rnd.gauss(0, 100))) for _ in range(M * N)]
            cases.append(make_case(f"{fmt}-{shape_name}-{cls}-{idx}", shape, fmt, acc, cls, a_pairs, b_pairs, c_vals))
    return {"cases": cases, "seed": seed, "band": "gamma_(K+2) * (sum|a*b| + |c|); gamma_n = n*2^-24/(1-n*2^-24)"}


def main(out_dir: pathlib.Path) -> int:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mma.json").write_text(json.dumps(build(), sort_keys=True, separators=(",", ":")) + "\n")
    write_manifest(out_dir)
    print(f"wrote {out_dir}/mma.json and MANIFEST.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "oracles" / "data")))
