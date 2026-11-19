"""PLAN.md rule: a gate is done only when merged with CI green. Ticks made on local evidence must say so."""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]


def ticked_lines():
    return [line for line in (REPO / "PLAN.md").read_text().splitlines() if re.match(r"\s*- \[x\]", line)]


def test_there_are_ticked_boxes_to_check():
    assert len(ticked_lines()) >= 4


def test_every_tick_names_its_evidence_and_says_ci_never_ran():
    for line in ticked_lines():
        assert "local evidence only" in line and "CI never run" in line, line
