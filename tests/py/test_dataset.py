"""Tests for data/spark_decode.csv, data/schema.json and tools/import_field_notes.py (Task 7, Gate 3a)."""
import contextlib
import csv
import importlib.util
import io
import json
import pathlib

import jsonschema
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
CSV = REPO / "data" / "spark_decode.csv"
NUMERIC = {"active_params_b": float, "params_total_b": float, "bits_per_weight": float, "depth": int, "batch": int, "value_tps": float, "value_pm": float}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows():
    out = []
    for r in csv.DictReader(CSV.open()):
        out.append({k: (NUMERIC[k](v) if k in NUMERIC and v != "" else v) for k, v in r.items() if not (v == "" and k in NUMERIC)})
    return out


def test_csv_validates_against_schema():
    schema = json.loads((REPO / "data" / "schema.json").read_text())
    data = rows()
    assert len(data) == 15
    for r in data:
        jsonschema.validate(r, schema)
        assert r["value_tps"] > 0 and r["source_url"]


def test_ids_are_unique():
    ids = [r["id"] for r in rows()]
    assert len(ids) == len(set(ids))


def test_secondary_sources_are_flagged_and_raw_files_are_marked_verified():
    for r in rows():
        if any(h in r["source_url"] for h in ("verdictbits.com", "developer.nvidia.com/blog", "lmsys.org", "ggml-org/llama.cpp")):
            assert r["source_kind"] == "secondary", r["id"]
        if "dgx-spark-field-notes" in r["source_url"]:
            assert r["source_kind"] == "verified_raw", r["id"]


def test_spec_decoding_column_is_true_false_or_unknown():
    assert {r["spec_decoding"] for r in rows()} <= {"true", "false", "unknown"}


def test_the_ambiguous_gemma_reading_is_recorded_as_unknown():
    # field notes label the 30.3 row "SEM spec-decoding" in results.jsonl but call it "no spec" in README/RESULTS.md
    r = next(r for r in rows() if r["value_tps"] == 30.3)
    assert r["spec_decoding"] == "unknown" and "label" in r["notes"]


def test_values_match_the_gate_4a_analysis():
    spike = load(REPO / "spikes" / "gate4a" / "gate4a_v2.py", "gate4a_v2")  # noqa: F841 (prints on import)
    with contextlib.redirect_stdout(io.StringIO()):
        spike = load(REPO / "spikes" / "gate4a" / "gate4a_v2.py", "gate4a_v2b")
    assert sorted(r["value_tps"] for r in rows()) == sorted(x[3] for x in spike.R)


def test_committed_csv_matches_the_importer(tmp_path):
    imp = load(REPO / "tools" / "import_field_notes.py", "import_field_notes")
    out = tmp_path / "spark_decode.csv"
    imp.main(out)
    assert out.read_bytes() == CSV.read_bytes()


def test_importer_reads_the_raw_field_notes_file():
    imp = load(REPO / "tools" / "import_field_notes.py", "import_field_notes2")
    raw = [r for r in imp.rows() if r["source_kind"] == "verified_raw"]
    assert len(raw) == 8 and all("dgx-spark-field-notes" in r["source_url"] for r in raw)
    assert {r["stack"] for r in raw} == {"vLLM", "Ollama"}
