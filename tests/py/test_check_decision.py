"""Tests for tools/check_decision.py (Task 8): the Gate 2 spike's decision record."""
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SHA = "522cde29551cdb66a8a8456a36da21bae57d72fe"


def load():
    spec = importlib.util.spec_from_file_location("check_decision", REPO / "tools" / "check_decision.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def record(tmp_path, decision="fork", hours="0.13", licence="true", sha=SHA, front=True):
    text = f"---\ndecision: {decision}\nhours_spent: {hours}\nptoxide_commit: {sha}\nlicence_checked: {licence}\n---\n\n# Body\n" if front else "# no front matter\n"
    p = tmp_path / "0001.md"
    p.write_text(text)
    return p


def test_the_real_decision_record_is_valid():
    assert load().check(REPO / "docs" / "decisions" / "0001-ptx-executor.md") == []


def test_a_valid_record_passes(tmp_path):
    assert load().check(record(tmp_path)) == []


def test_missing_front_matter_is_reported(tmp_path):
    assert "front matter missing" in load().check(record(tmp_path, front=False))[0]


@pytest.mark.parametrize("decision", ["maybe", "", "Fork"])
def test_decision_must_be_one_of_the_three_values(tmp_path, decision):
    assert any("decision must be one of fork, fresh, numerics-only" in e for e in load().check(record(tmp_path, decision=decision)))


@pytest.mark.parametrize("hours", ["many", "", "-1", "nan"])
def test_hours_spent_must_be_a_non_negative_number(tmp_path, hours):
    assert any("hours_spent" in e for e in load().check(record(tmp_path, hours=hours)))


def test_licence_checked_must_be_true(tmp_path):
    assert any("licence_checked must be true" in e for e in load().check(record(tmp_path, licence="false")))


@pytest.mark.parametrize("sha", ["abc123", "", "Z" * 40])
def test_ptoxide_commit_must_be_a_full_sha(tmp_path, sha):
    assert any("ptoxide_commit" in e for e in load().check(record(tmp_path, sha=sha)))


def test_more_than_16_hours_requires_numerics_only(tmp_path):
    assert any("numerics-only is required when hours_spent > 16" in e for e in load().check(record(tmp_path, decision="fork", hours="16.5")))
    assert load().check(record(tmp_path, decision="numerics-only", hours="40")) == []
    assert load().check(record(tmp_path, decision="fork", hours="16")) == []
