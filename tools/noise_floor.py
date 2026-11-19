#!/usr/bin/env python3
"""Noise floor and 'error against an interval' for public Spark benchmark data (Task 7, Gate 3d).

sigma_between is the UNSCALED median absolute deviation (MAD) of ln(x) across distinct submitters, as the spec states.
Several values from one submitter are reduced to their median first, so duplicates cannot masquerade as several
submitters. Intervals are [x e^(-1.96 sigma), x e^(+1.96 sigma)].
"""
import csv
import math
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
NUMERIC = {"active_params_b": float, "params_total_b": float, "bits_per_weight": float, "depth": int, "batch": int, "value_tps": float, "value_pm": float}


def sigma_between(values, submitters):
    if len(values) != len(submitters):
        raise ValueError("values and submitters must have the same length")
    for v in values:
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"values must be positive and finite, got {v}")
    per_submitter = {}
    for v, s in zip(values, submitters):
        per_submitter.setdefault(s, []).append(v)
    if len(per_submitter) < 2:
        return None
    logs = [math.log(statistics.median(vs)) for vs in per_submitter.values()]
    centre = statistics.median(logs)
    return statistics.median(abs(x - centre) for x in logs)


def interval(x_med, sigma):
    return (x_med * math.exp(-1.96 * sigma), x_med * math.exp(1.96 * sigma))


def interval_error(pred, lo, hi):
    """0 inside [lo, hi]; otherwise the relative distance to the nearest edge."""
    if pred < lo:
        return (lo - pred) / lo
    if pred > hi:
        return (pred - hi) / hi
    return 0.0


def group_rows(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["fmt"], r["stack"], r["depth"], r["metric"]), []).append(r)
    return groups


def noise_floor(rows):
    out = {}
    for key, group in group_rows(rows).items():
        values, submitters = [r["value_tps"] for r in group], [r["operator"] for r in group]
        sigma = sigma_between(values, submitters)
        if sigma is None:
            continue
        per = {}
        for v, s in zip(values, submitters):
            per.setdefault(s, []).append(v)
        median = statistics.median(statistics.median(vs) for vs in per.values())
        out[key] = {"n_submitters": len(per), "sigma_between": sigma, "median": median, "interval": interval(median, sigma)}
    return out


def load_rows(path):
    rows = []
    for r in csv.DictReader(pathlib.Path(path).open()):
        rows.append({k: (NUMERIC[k](v) if k in NUMERIC and v != "" else v) for k, v in r.items() if not (v == "" and k in NUMERIC)})
    return rows


def main(path=ROOT / "data" / "spark_decode.csv"):
    result = noise_floor(load_rows(path))
    if not result:
        print("no group has two distinct submitters yet: the noise floor cannot be estimated from this dataset")
    for key, e in result.items():
        print(key, f"submitters={e['n_submitters']} sigma_between={e['sigma_between']:.3f} median={e['median']:.2f} interval=({e['interval'][0]:.2f}, {e['interval'][1]:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
