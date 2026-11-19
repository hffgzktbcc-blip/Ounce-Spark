#!/usr/bin/env python3
"""Governance guard for ounce-spark (Task 1).

Rejects what the spec forbids committing (NVIDIA binaries, NVIDIA documentation HTML),
requires the governance files, and requires #![forbid(unsafe_code)] in every crate root.
"""
import pathlib
import sys

REQUIRED = [
    "LICENSE", "README.md", "PLAN.md", "THIRD_PARTY.md", "deny.toml", "rust-toolchain.toml",
    "docs/LIMITS.md", "docs/PROVENANCE.md", "docs/SOURCES.md",
]
SKIP_DIRS = {".git", "target", ".cache", ".superpowers", "__pycache__", ".pytest_cache"}
BINARY_EXTENSIONS = {".cubin", ".fatbin"}
HTML_EXTENSIONS = {".html", ".htm"}
ELF_MAGIC = b"\x7fELF"
FORBID_UNSAFE = "#![forbid(unsafe_code)]"


def _looks_like_html(path: pathlib.Path, head: bytes) -> bool:
    """An HTML document by extension, or by starting with a doctype/html tag (after BOM/whitespace)."""
    if path.suffix.lower() in HTML_EXTENSIONS:
        return True
    start = head.lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    return start.startswith((b"<!doctype html", b"<html"))


def _files(root: pathlib.Path):
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            yield path


def check(root: pathlib.Path) -> list[str]:
    root = pathlib.Path(root)
    violations = [f"missing: {rel}" for rel in REQUIRED if not (root / rel).is_file()]
    for path in _files(root):
        rel = path.relative_to(root).as_posix()
        head = path.read_bytes()[:2048]
        if head.startswith(ELF_MAGIC) or path.suffix in BINARY_EXTENSIONS:
            violations.append(f"binary rejected: {rel}")
        elif _looks_like_html(path, head) and b"PTX ISA" in head:
            violations.append(f"NVIDIA documentation HTML rejected: {rel}")
        parts = path.relative_to(root).parts
        if len(parts) == 4 and parts[0] == "crates" and parts[2:] == ("src", "lib.rs"):
            if FORBID_UNSAFE not in path.read_text(errors="replace"):
                violations.append(f"forbid(unsafe_code) missing: {rel}")
    return violations


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    violations = check(root)
    for v in violations:
        print(v)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
