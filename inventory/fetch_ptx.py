#!/usr/bin/env python3
"""Fetch NVIDIA's public PTX ISA page into the untracked cache (Task 2).

The HTML is NEVER written under a tracked path: the default destination is .cache/ (gitignored) and
tools/check_governance.py rejects NVIDIA documentation HTML anywhere else.
"""
import datetime
import hashlib
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
URL = "https://docs.nvidia.com/cuda/parallel-thread-execution/"
DEFAULT_DEST = ROOT / ".cache" / "ptx.html"
DEFAULT_SOURCE = ROOT / "inventory" / "SOURCE.json"


def fetch(dest: pathlib.Path = DEFAULT_DEST, url: str = URL, source_json: pathlib.Path = DEFAULT_SOURCE) -> pathlib.Path:
    dest, source_json = pathlib.Path(dest), pathlib.Path(source_json)
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    record = {
        "url": url,
        "fetched_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    source_json.parent.mkdir(parents=True, exist_ok=True)
    source_json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return dest


if __name__ == "__main__":
    print(fetch())
