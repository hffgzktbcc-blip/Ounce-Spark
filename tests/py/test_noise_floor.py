"""Tests for tools/noise_floor.py (Task 7, Gate 3d)."""
import importlib.util
import math
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


def load():
    spec = importlib.util.spec_from_file_location("noise_floor", REPO / "tools" / "noise_floor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_identical_values_from_two_submitters_have_zero_sigma():
    assert load().sigma_between([60.0, 60.0], ["a", "b"]) == 0.0


def test_sigma_is_the_unscaled_mad_of_ln_x():
    # ln values are [0, 1]; median 0.5; absolute deviations [0.5, 0.5]; MAD 0.5 (no 1.4826 factor, as the spec states)
    assert load().sigma_between([1.0, math.e], ["a", "b"]) == pytest.approx(0.5)
    assert load().sigma_between([1.0, 2.0, 4.0], ["a", "b", "c"]) == pytest.approx(math.log(2))


def test_a_single_value_has_no_sigma():
    assert load().sigma_between([60.0], ["a"]) is None


def test_the_same_submitter_twice_counts_once():
    assert load().sigma_between([50.0, 60.0], ["a", "a"]) is None


def test_values_from_one_submitter_are_reduced_by_median_before_the_mad():
    # submitter a -> median(50, 60) = 55; submitter b -> 100. ln values [ln55, ln100]; MAD = |ln100 - ln55| / 2
    got = load().sigma_between([50.0, 60.0, 100.0], ["a", "a", "b"])
    assert got == pytest.approx(abs(math.log(100) - math.log(55)) / 2)


@pytest.mark.parametrize("values", [[0.0, 1.0], [-1.0, 1.0], [float("nan"), 1.0], [float("inf"), 1.0]])
def test_non_positive_or_non_finite_values_are_rejected(values):
    with pytest.raises(ValueError, match="positive"):
        load().sigma_between(values, ["a", "b"])


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        load().sigma_between([1.0, 2.0], ["a"])


def test_interval_with_zero_sigma_is_a_point():
    assert load().interval(60.0, 0.0) == (60.0, 60.0)


def test_interval_widens_by_exp_of_1_96_sigma():
    lo, hi = load().interval(60.0, 0.1)
    assert lo == pytest.approx(60 * math.exp(-0.196)) and hi == pytest.approx(60 * math.exp(0.196))


def test_interval_error_is_zero_inside_and_relative_to_the_nearest_edge_outside():
    f = load().interval_error
    assert f(60.0, 55.0, 66.0) == 0
    assert f(55.0, 55.0, 66.0) == 0 and f(66.0, 55.0, 66.0) == 0
    assert f(88.8, 55.0, 66.0) == pytest.approx((88.8 - 66.0) / 66.0)
    assert f(44.0, 55.0, 66.0) == pytest.approx((55.0 - 44.0) / 55.0)


def row(model="m", fmt="MXFP4", stack="vLLM", depth=0, metric="tg", operator="a", value=60.0):
    return {"model": model, "fmt": fmt, "stack": stack, "depth": depth, "metric": metric, "operator": operator, "value_tps": value}


def test_group_rows_keys_on_model_fmt_stack_depth_metric():
    groups = load().group_rows([row(operator="a"), row(operator="b"), row(stack="Ollama"), row(depth=4096)])
    assert set(groups) == {("m", "MXFP4", "vLLM", 0, "tg"), ("m", "MXFP4", "Ollama", 0, "tg"), ("m", "MXFP4", "vLLM", 4096, "tg")}
    assert len(groups[("m", "MXFP4", "vLLM", 0, "tg")]) == 2


def test_noise_floor_reports_only_groups_with_two_distinct_submitters():
    rows = [row(operator="a", value=60.0), row(operator="b", value=45.0), row(stack="Ollama", operator="a", value=42.0), row(stack="Ollama", operator="a", value=43.0)]
    out = load().noise_floor(rows)
    assert list(out) == [("m", "MXFP4", "vLLM", 0, "tg")]
    entry = out[("m", "MXFP4", "vLLM", 0, "tg")]
    assert entry["n_submitters"] == 2
    assert entry["sigma_between"] == pytest.approx(abs(math.log(60) - math.log(45)) / 2)
    lo, hi = entry["interval"]
    assert lo < entry["median"] < hi
