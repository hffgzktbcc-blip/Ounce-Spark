"""Tests for oracles/gen_elements.py (Task 3, Gate 1a). Facts below are asserted independently of the libraries."""
import copy
import hashlib
import importlib.util
import json
import math
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
INVENTORY = REPO / "inventory" / "ptx_sm12x.json"


def load():
    spec = importlib.util.spec_from_file_location("gen_elements", REPO / "oracles" / "gen_elements.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def built():
    return load().build()


def val(s):
    return float(s)  # accepts "nan", "inf", "-inf" and float.hex() strings via float.fromhex fallback


def decode(built, fmt):
    return [float.fromhex(s) if s.startswith(("0x", "-0x")) else float(s) for s in built["formats"][fmt]["decode"]]


def xval(s):
    return float.fromhex(s) if s.startswith(("0x", "-0x")) else float(s)


def case(built, fmt, x, sat):
    hits = [c for c in built["encode_cases"] if c["fmt"] == fmt and c["sat"] == sat and not math.isnan(xval(c["x"]))
            and xval(c["x"]) == x and math.copysign(1, xval(c["x"])) == math.copysign(1, x)]
    assert hits, f"no encode case for {fmt} {x} sat={sat}"
    return hits[0]


def special(built, fmt, token, sat):
    hits = [c for c in built["encode_cases"] if c["fmt"] == fmt and c["x"] == token and c["sat"] == sat]
    assert hits, f"no encode case for {fmt} {token} sat={sat}"
    return hits[0]


def test_e2m1_decode_values_are_the_ocp_set(built):
    d = decode(built, "e2m1")
    assert d[:8] == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    assert d[8:] == [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    assert math.copysign(1, d[8]) == -1.0


@pytest.mark.parametrize("fmt,mx", [("e4m3", 448.0), ("e5m2", 57344.0), ("e2m3", 7.5), ("e3m2", 28.0), ("e2m1", 6.0)])
def test_format_maxima(built, fmt, mx):
    assert built["formats"][fmt]["max"] == mx
    assert max(v for v in decode(built, fmt) if math.isfinite(v)) == mx


def test_ue8m0_decode(built):
    d = decode(built, "ue8m0")
    assert d[127] == 1.0 and d[0] == 2.0 ** -127 and d[254] == 2.0 ** 127
    assert math.isnan(d[255])
    assert not any(math.isnan(v) for v in d[:255])


def test_inf_and_nan_flags(built):
    f = built["formats"]
    for fmt in ("e2m1", "e2m3", "e3m2"):
        assert f[fmt]["has_inf"] is False and f[fmt]["has_nan"] is False
        assert all(math.isfinite(v) for v in decode(built, fmt))
    assert (f["e4m3"]["has_inf"], f["e4m3"]["has_nan"]) == (False, True)
    assert (f["e5m2"]["has_inf"], f["e5m2"]["has_nan"]) == (True, True)
    assert (f["ue8m0"]["has_inf"], f["ue8m0"]["has_nan"]) == (False, True)


def test_e4m3_nan_is_only_code_7f_and_ff(built):
    assert {i for i, v in enumerate(decode(built, "e4m3")) if math.isnan(v)} == {0x7F, 0xFF}


def test_ties_round_to_even_e2m1(built):
    assert case(built, "e2m1", 2.5, True)["code"] == 0b0100    # tie between 2 (even) and 3 -> 2
    assert case(built, "e2m1", 0.25, True)["code"] == 0b0000   # tie between 0 (even) and 0.5 -> 0


def test_ties_round_to_even_e4m3(built):
    assert case(built, "e4m3", 1.0625, False)["code"] == 0x38  # tie between 1.0 (0x38) and 1.125 (0x39) -> even 0x38
    assert case(built, "e4m3", 1.1875, False)["code"] == 0x3A  # tie between 1.125 (0x39) and 1.25 (0x3A) -> even 0x3A


def test_satfinite_clamps_and_negative_zero_survives(built):
    assert case(built, "e2m1", 7.0, True)["code"] == 0b0111
    assert case(built, "e2m1", 1e30, True)["code"] == 0b0111
    assert case(built, "e4m3", 1e30, True)["code"] == 0x7E     # 448 is code 0x7E
    assert case(built, "e2m1", -0.0, True)["code"] == 0b1000


def test_nan_under_satfinite_is_positive_max_norm_for_formats_without_nan(built):
    # PTX cvt: "For .e2m1x2, .e2m3x2, .e3m2x2 ... destination formats NaN results are converted to positive MAX_NORM."
    for fmt, max_code in [("e2m1", 0b0111), ("e2m3", 0x1F), ("e3m2", 0x1F)]:
        c = special(built, fmt, "nan", True)
        assert c["code"] == max_code and c["nan"] is False, fmt


def test_infinity_under_satfinite_is_sign_preserved_max_norm(built):
    # PTX cvt: "If the absolute value of input is greater than MAX_NORM ... the result is sign-preserved MAX_NORM."
    for fmt, max_code, sign in [("e4m3", 0x7E, 0x80), ("e5m2", 0x7B, 0x80), ("e2m3", 0x1F, 0x20), ("e3m2", 0x1F, 0x20), ("e2m1", 0x07, 0x08)]:
        assert special(built, fmt, "inf", True)["code"] == max_code, fmt
        assert special(built, fmt, "-inf", True)["code"] == sign | max_code, fmt


def test_nan_under_satfinite_stays_nan_for_formats_that_have_nan(built):
    # PTX cvt: for .e4m3x2/.e5m2x2 "if the input value is NaN, then the result is NaN in the specified destination format."
    for fmt in ("e4m3", "e5m2"):
        c = special(built, fmt, "nan", True)
        assert c["nan"] is True and c["code"] is None


def test_non_saturating_conversion_exists_only_for_formats_with_nan(built):
    for c in built["encode_cases"]:
        if c["fmt"] in ("e2m1", "e2m3", "e3m2"):
            assert c["sat"] is True, "PTX makes .satfinite mandatory and the libraries cannot represent the overflow"


def test_off_grid_values_round_to_the_nearer_code(built):
    # Independent of the libraries: for EVERY gap of EVERY format, the 1/4 and 3/4 points must go to the nearer neighbour.
    for fmt in ("e4m3", "e5m2", "e2m3", "e3m2", "e2m1"):
        values = decode(built, fmt)
        pos = sorted((v, code) for code, v in enumerate(values) if math.isfinite(v) and v > 0 and math.copysign(1, v) > 0)
        pairs = list(zip([(0.0, 0)] + pos, pos))
        for (a, ca), (b, cb) in pairs:
            for frac, want in ((0.25, ca), (0.75, cb)):
                x = a + (b - a) * frac
                assert case(built, fmt, x, True)["code"] == want, (fmt, x)
                assert case(built, fmt, -x, True)["code"] == (want | (1 << (built["formats"][fmt]["bits"] - 1))), (fmt, -x)


def test_known_off_grid_points(built):
    assert case(built, "e2m1", 0.125, True)["code"] == 0b0000
    assert case(built, "e2m1", 0.375, True)["code"] == 0b0001
    assert case(built, "e2m1", 2.25, True)["code"] == 0b0100
    assert case(built, "e2m1", 2.75, True)["code"] == 0b0101
    assert case(built, "e4m3", 1.03125, True)["code"] == 0x38
    assert case(built, "e4m3", 1.09375, True)["code"] == 0x39


def test_ue8m0_has_no_encode_cases(built):
    assert not [c for c in built["encode_cases"] if c["fmt"] == "ue8m0"]


def test_two_libraries_agree_on_every_decode(built):
    assert built["disagreements"]["decode"] == []


def test_build_is_deterministic():
    g = load()
    assert json.dumps(g.build(), sort_keys=True) == json.dumps(g.build(), sort_keys=True)


def test_refuses_a_format_missing_from_the_inventory(tmp_path):
    inv = json.loads(INVENTORY.read_text())
    for e in inv["instructions"]:
        e["statement"] = e["statement"].replace("e2m3", "xxxx")
    bad = tmp_path / "inv.json"
    bad.write_text(json.dumps(inv))
    with pytest.raises(ValueError, match="e2m3"):
        load().build(inventory_path=bad)


def test_manifest_matches_files(tmp_path):
    g = load()
    out = tmp_path / "data"
    g.main(out)
    manifest = (tmp_path / "MANIFEST.sha256").read_text().splitlines()
    assert manifest, "empty manifest"
    for line in manifest:
        digest, rel = line.split("  ", 1)
        assert hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest() == digest
    assert any(line.endswith("data/elements.json") for line in manifest)


def test_committed_oracle_files_match_a_fresh_build(tmp_path):
    g = load()
    g.main(tmp_path / "data")
    committed = REPO / "oracles" / "data" / "elements.json"
    assert committed.exists(), "run python oracles/gen_elements.py oracles/data"
    assert committed.read_bytes() == (tmp_path / "data" / "elements.json").read_bytes()
    line = (tmp_path / "MANIFEST.sha256").read_text().strip()   # the committed manifest also lists mma.json
    assert line in (REPO / "oracles" / "MANIFEST.sha256").read_text().splitlines()


def test_unlisted_library_disagreement_fails_generation(monkeypatch):
    g = load()
    monkeypatch.setattr(g, "ml_encode", lambda fmt, x: g.gfloat.encode_float(g.FORMATS[fmt][1], 1.0))
    with pytest.raises(AssertionError, match="unlisted disagreements"):
        g.build()


def test_nan_results_are_recorded_as_nan_without_a_canonical_code(built):
    assert all("nan" in c for c in built["encode_cases"])
    overflow = [c for c in built["encode_cases"] if c["fmt"] == "e4m3" and c["sat"] is False and float.fromhex(c["x"]) == 1e30]
    assert overflow and overflow[0]["nan"] is True and overflow[0]["code"] is None
    ordinary = case(built, "e2m1", 2.5, True)
    assert ordinary["nan"] is False and ordinary["code"] == 4
