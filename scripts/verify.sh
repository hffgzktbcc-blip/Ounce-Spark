#!/usr/bin/env bash
# Strict whole-repo verification: any failing step fails the script.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest tests/py -q
cargo test --workspace -q
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo deny check
python tools/check_governance.py
echo "verify: all steps passed"
