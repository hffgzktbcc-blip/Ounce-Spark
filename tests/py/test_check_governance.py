"""Tests for tools/check_governance.py (Task 1)."""
import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
REQUIRED = [
    "LICENSE", "README.md", "PLAN.md", "THIRD_PARTY.md", "deny.toml",
    "rust-toolchain.toml", "docs/LIMITS.md", "docs/PROVENANCE.md", "docs/SOURCES.md",
]


def load():
    spec = importlib.util.spec_from_file_location("check_governance", REPO / "tools" / "check_governance.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_tree(root: pathlib.Path) -> pathlib.Path:
    for rel in REQUIRED:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
    return root


def test_clean_tree_passes(tmp_path):
    assert load().check(make_tree(tmp_path)) == []


def test_missing_required_file_reported(tmp_path):
    root = make_tree(tmp_path)
    (root / "LICENSE").unlink()
    assert "missing: LICENSE" in load().check(root)


def test_forbid_unsafe_required(tmp_path):
    root = make_tree(tmp_path)
    lib = root / "crates" / "x" / "src" / "lib.rs"
    lib.parent.mkdir(parents=True)
    lib.write_text("pub fn f() {}\n")
    assert "forbid(unsafe_code) missing: crates/x/src/lib.rs" in load().check(root)


def test_forbid_unsafe_present_passes(tmp_path):
    root = make_tree(tmp_path)
    lib = root / "crates" / "x" / "src" / "lib.rs"
    lib.parent.mkdir(parents=True)
    lib.write_text("#![forbid(unsafe_code)]\npub fn f() {}\n")
    assert load().check(root) == []


def test_elf_binary_rejected(tmp_path):
    root = make_tree(tmp_path)
    (root / "blob.bin").write_bytes(b"\x7fELF\x02\x01\x01")
    assert "binary rejected: blob.bin" in load().check(root)


def test_cubin_extension_rejected(tmp_path):
    root = make_tree(tmp_path)
    (root / "a.cubin").write_text("not really")
    (root / "sub").mkdir()
    (root / "sub" / "b.fatbin").write_text("not really")
    out = load().check(root)
    assert "binary rejected: a.cubin" in out
    assert "binary rejected: sub/b.fatbin" in out


def test_nvidia_doc_html_rejected(tmp_path):
    root = make_tree(tmp_path)
    (root / "ptx.html").write_text("<!DOCTYPE html><html><title>PTX ISA 9.5</title></html>")
    assert "NVIDIA documentation HTML rejected: ptx.html" in load().check(root)


def test_plain_html_without_ptx_isa_allowed(tmp_path):
    root = make_tree(tmp_path)
    (root / "page.html").write_text("<html><title>hello</title></html>")
    assert load().check(root) == []


def test_cache_dir_is_skipped(tmp_path):
    root = make_tree(tmp_path)
    (root / ".cache").mkdir()
    (root / ".cache" / "ptx.html").write_text("<html>PTX ISA</html>")
    assert load().check(root) == []


def test_real_repo_passes():
    assert load().check(REPO) == []


def test_source_file_mentioning_both_markers_allowed(tmp_path):
    root = make_tree(tmp_path)
    (root / "guard.py").write_text('MARKERS = (b"<html", b"PTX ISA")\n')
    assert load().check(root) == []


def test_html_page_without_extension_still_rejected(tmp_path):
    root = make_tree(tmp_path)
    (root / "saved_page").write_text("﻿  \n<!doctype html><title>PTX ISA 9.5</title>")
    assert "NVIDIA documentation HTML rejected: saved_page" in load().check(root)
