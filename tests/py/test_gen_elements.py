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


def case(built, fmt, x, sat):
    hits = [c for c in built["encode_cases"] if c["fmt"] == fmt and float.fromhex(c["x"]) == x and c["sat"] == sat
            and math.copysign(1, float.fromhex(c["x"])) == math.copysign(1, x)]
    assert hits, f"no encode case for {fmt} {x} sat={sat}"
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


def test_no_nan_or_inf_cases_for_formats_without_them(built):
    for c in built["encode_cases"]:
        if c["fmt"] in ("e2m1", "e2m3", "e3m2"):
            assert math.isfinite(float.fromhex(c["x"]) if c["x"].startswith(("0x", "-0x")) else float(c["x"]))
            assert c["sat"] is True, "PTX makes .satfinite mandatory for these destination types"


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
    assert (REPO / "oracles" / "MANIFEST.sha256").read_bytes() == (tmp_path / "MANIFEST.sha256").read_bytes()


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
