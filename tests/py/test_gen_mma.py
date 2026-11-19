"""Tests for oracles/gen_mma.py (Task 5, Gate 1b)."""
import importlib.util
import json
import math
import pathlib
import random
from fractions import Fraction

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
INVENTORY = REPO / "inventory" / "ptx_sm12x.json"
COMBOS = [
    ("m16n8k8", "f16", "f32"),
    ("m16n8k16", "f16", "f32"),
    ("m16n8k16", "bf16", "f32"),
    ("m16n8k32", "e4m3", "f32"),
    ("m16n8k32", "e5m2", "f32"),
    ("m16n8k32", "s8", "s32"),
]


def load(name="gen_mma"):
    spec = importlib.util.spec_from_file_location(name, REPO / "oracles" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def built():
    return load().build()


def num(s, acc):
    if acc == "s32":
        return int(s)
    return float.fromhex(s) if s.startswith(("0x", "-0x")) else float(s)


def test_deterministic_for_seed():
    g = load()
    assert json.dumps(g.build(seed=1), sort_keys=True) == json.dumps(g.build(seed=1), sort_keys=True)
    assert json.dumps(g.build(seed=1), sort_keys=True) != json.dumps(g.build(seed=2), sort_keys=True)


def test_exact_cases_are_order_independent(built):
    checked = 0
    for c in built["cases"]:
        if c["class"] != "exact":
            continue
        m, n, k = c["shape"]
        a, b, cc, exp = (list(map(lambda s: num(s, c["acc"]), c[key])) for key in ("a", "b", "c", "expected"))
        rnd = random.Random(7)
        for i in range(m):
            for j in range(n):
                terms = [cc[i * n + j]] + [a[i * k + t] * b[t * n + j] for t in range(k)]
                shuffled = terms[:]
                rnd.shuffle(shuffled)
                sums = {sum(terms), sum(reversed(terms)), sum(shuffled), math.fsum(terms)}
                assert sums == {exp[i * n + j]}, (c["id"], i, j, sums, exp[i * n + j])
        checked += 1
    assert checked >= 30


def test_case_counts_per_combination(built):
    for shape, fmt, acc in COMBOS:
        cases = [c for c in built["cases"] if c["shape"] == [int(x) for x in (16, 8, int(shape.split("k")[1]))] and c["a_fmt"] == fmt]
        assert len(cases) >= 20, (shape, fmt, len(cases))
        assert sum(c["class"] == "exact" for c in cases) >= 5, (shape, fmt)
        assert all(c["acc"] == acc and c["b_fmt"] == fmt for c in cases)


def test_s8_cases_are_always_exact_integers(built):
    cases = [c for c in built["cases"] if c["a_fmt"] == "s8"]
    assert cases and all(c["class"] == "exact" for c in cases)
    assert all(isinstance(x, int) for c in cases for x in c["expected"])


def test_refuses_a_combination_missing_from_the_inventory(tmp_path):
    inv = json.loads(INVENTORY.read_text())
    for e in inv["instructions"]:
        if e["kind"] == "form":
            e["a_types"] = [t for t in e["a_types"] if t != "bf16"]
    bad = tmp_path / "inv.json"
    bad.write_text(json.dumps(inv))
    with pytest.raises(ValueError, match="bf16"):
        load().build(inventory_path=bad)


def test_fraction_to_f32_rounds_half_to_even():
    f = load().fraction_to_f32
    u = Fraction(1, 2**24)
    assert f(1 + u) == 1.0                              # tie between 1 and 1+2^-23: even mantissa is 1.0
    assert f(1 + 3 * u) == 1 + 2.0 ** -22               # tie between 1+2^-23 (odd) and 1+2^-22 (even)
    assert f(-(1 + 3 * u)) == -(1 + 2.0 ** -22)
    assert f(Fraction(1, 2**150)) == 0.0                # tie between 0 and 2^-149: even is 0
    assert f(Fraction(3, 2**150)) == 2.0 ** -148        # tie between 2^-149 (odd) and 2^-148 (even)
    assert f(Fraction(0)) == 0.0 and f(Fraction(5, 4)) == 1.25


def test_band_formula_is_gamma_of_k_plus_2_with_unit_roundoff_2_pow_minus_23():
    # u = 2^-23 covers faithful rounding of the accumulator (round-to-nearest, round-toward-zero, any directed mode)
    # and truncating alignment of the addends, not only round-to-nearest (u = 2^-24).
    g = load()
    k, abs_sum, abs_c = 16, Fraction(2), Fraction(1)
    n = Fraction(k + 2, 2**23)
    assert g.band_for(k, abs_sum, abs_c) == n / (1 - n) * (abs_sum + abs_c)


def test_the_band_covers_a_truncating_accumulator_that_round_to_nearest_would_not():
    # Worst case for round-toward-zero: c = 1.0 and 16 products of (1023/1024) * 2^-23, each just under one ulp of 1.0.
    # A truncating f32 accumulator loses every product (result 1.0); round-to-nearest rounds each up and stays close.
    import numpy as np
    g = load()
    k = 16
    term = Fraction(1023, 1024) * Fraction(1, 2**23)           # = 2^-12 * (2^-11 * (1 - 2^-10)), exact in f16 x f16
    exact = Fraction(1) + k * term
    acc = np.float32(1.0)
    for _ in range(k):                                         # round toward zero after every addition
        s = np.float32(acc + np.float32(float(term)))
        if Fraction(float(s)) > Fraction(float(acc)) + term:   # nearest rounded up: step back one ulp
            s = np.nextafter(s, np.float32(0.0))
        acc = s
    error = abs(Fraction(float(acc)) - exact)
    assert acc == np.float32(1.0)                              # every product was truncated away
    nearest_only_band = Fraction(k + 2, 2**24) / (1 - Fraction(k + 2, 2**24)) * (k * term + 1)
    assert error > nearest_only_band                           # a round-to-nearest band is NOT enough
    assert error <= g.band_for(k, k * term, Fraction(1))       # the band in the oracle IS


def test_stored_band_matches_the_formula_for_one_banded_case(built):
    g = load()
    case = next(c for c in built["cases"] if c["class"] == "banded" and c["a_fmt"] == "f16")
    m, n, k = case["shape"]
    a, b, c = (list(map(lambda s: num(s, "f32"), case[key])) for key in ("a", "b", "c"))
    i, j = 3, 5
    abs_sum = sum(Fraction(abs(a[i * k + t])) * Fraction(abs(b[t * n + j])) for t in range(k))
    want = float(g.band_for(k, abs_sum, Fraction(abs(c[i * n + j]))))
    assert num(case["band"][i * n + j], "f32") == pytest.approx(want, rel=1e-12)


def test_banded_expected_is_the_correctly_rounded_exact_sum(built):
    for case in [c for c in built["cases"] if c["class"] == "banded"][:25]:
        m, n, k = case["shape"]
        a, b, c, exp = (list(map(lambda s: num(s, "f32"), case[key])) for key in ("a", "b", "c", "expected"))
        for idx in (0, 17, n * m - 1):
            i, j = divmod(idx, n)
            exact = math.fsum([c[idx]] + [a[i * k + t] * b[t * n + j] for t in range(k)])
            assert abs(exp[idx] - exact) <= abs(exact) * 2.0 ** -24 + 2.0 ** -149, (case["id"], idx)


def test_inputs_carry_bits_and_decoded_values(built):
    case = next(c for c in built["cases"] if c["a_fmt"] == "f16")
    assert len(case["a"]) == len(case["a_bits"]) == 16 * case["shape"][2]
    assert len(case["b"]) == len(case["b_bits"]) == 8 * case["shape"][2]
    assert len(case["c"]) == len(case["expected"]) == len(case["band"]) == 16 * 8


def test_manifest_lists_every_oracle_file(tmp_path):
    elements, mma = load("gen_elements"), load()
    out = tmp_path / "data"
    elements.main(out)
    mma.main(out)
    manifest = (tmp_path / "MANIFEST.sha256").read_text()
    assert "data/elements.json" in manifest and "data/mma.json" in manifest


def test_committed_mma_oracle_matches_a_fresh_build(tmp_path):
    g = load()
    g.main(tmp_path / "data")
    committed = REPO / "oracles" / "data" / "mma.json"
    assert committed.exists(), "run python oracles/gen_mma.py oracles/data"
    assert committed.read_bytes() == (tmp_path / "data" / "mma.json").read_bytes()
