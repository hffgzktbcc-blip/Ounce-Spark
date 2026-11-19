#!/usr/bin/env python3
"""Validate the Gate 2 spike's decision record (Task 8): front matter, allowed values, and the spec's 16-hour rule."""
import math
import pathlib
import re
import sys

DECISIONS = ("fork", "fresh", "numerics-only")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def front_matter(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, flags=re.S)
    if not m:
        return None
    fields = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def check(path):
    fields = front_matter(pathlib.Path(path).read_text())
    if fields is None:
        return ["front matter missing"]
    errors = []
    decision = fields.get("decision", "")
    if decision not in DECISIONS:
        errors.append("decision must be one of fork, fresh, numerics-only")
    hours = None
    try:
        hours = float(fields.get("hours_spent", ""))
        if not math.isfinite(hours) or hours < 0:
            raise ValueError
    except ValueError:
        hours = None
        errors.append("hours_spent must be a non-negative number")
    if fields.get("licence_checked") != "true":
        errors.append("licence_checked must be true")
    if not re.fullmatch(r"[0-9a-f]{40}", fields.get("ptoxide_commit", "")):
        errors.append("ptoxide_commit must be a full 40-character sha")
    if hours is not None and hours > 16 and decision != "numerics-only":
        errors.append("numerics-only is required when hours_spent > 16 (spec, Gate 2 spike rule)")
    return errors


def main(path=ROOT / "docs" / "decisions" / "0001-ptx-executor.md"):
    errors = check(path)
    for e in errors:
        print(e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
