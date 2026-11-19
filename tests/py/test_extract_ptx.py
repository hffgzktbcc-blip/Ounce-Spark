"""Tests for inventory/extract_ptx.py and inventory/fetch_ptx.py (Task 2, Gate 0b).

Unit tests use SYNTHETIC html written in our own words that mimics the page's structure. The integration
test runs against the real page only when .cache/ptx.html exists.
"""
import hashlib
import importlib.util
import json
import pathlib

import jsonschema
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
REAL_PAGE = REPO / ".cache" / "ptx.html"


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "inventory" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# name -> (category, title as printed in the heading)
INSTRUCTIONS = {
    "mma": ("Multiply-and-Accumulate Instruction", "mma"),
    "mma.sp": ("Multiply-and-Accumulate Instruction", "mma.sp / mma.sp::ordered_metadata"),
    "cvt": ("Data Movement and Conversion Instructions", "cvt"),
    "cvt.pack": ("Data Movement and Conversion Instructions", "cvt.pack"),
    "cp.async": ("Data Movement and Conversion Instructions", "cp.async"),
    "ldmatrix": ("Warp-level matrix load instruction", "ldmatrix"),
    "shfl.sync": ("Data Movement and Conversion Instructions", "shfl.sync"),
    "bar": ("Parallel Synchronization and Communication Instructions", "bar , barrier"),
    "atom": ("Parallel Synchronization and Communication Instructions", "atom"),
    "red": ("Parallel Synchronization and Communication Instructions", "red"),
    "rcp": ("Floating Point Instructions", "rcp"),
    "sqrt": ("Floating Point Instructions", "sqrt"),
    "rsqrt": ("Floating Point Instructions", "rsqrt"),
    "sin": ("Floating Point Instructions", "sin"),
    "cos": ("Floating Point Instructions", "cos"),
    "lg2": ("Floating Point Instructions", "lg2"),
    "ex2": ("Floating Point Instructions", "ex2"),
    "tanh": ("Floating Point Instructions", "tanh"),
}

SMEM = (
    '<section id="shared-state-space"><h3>5.1.7. Shared State Space <a></a></h3>'
    "<table><tr><th>Target architecture</th><th>Maximum statically allocated shared memory size</th></tr>"
    "<tr><td>sm_90a</td><td>228 KB</td></tr><tr><td>sm_120a , sm_121a</td><td>100 KB</td></tr></table></section>"
)
TARGET = (
    '<section id="target-dir"><h3>11.1.2. PTX Module Directives: .target <a></a></h3>'
    '<div class="highlight-text"><pre>.target stringlist\nstring = { sm_120a, sm_120f, sm_120, sm_121a, sm_121f, sm_121, sm_90 };</pre></div></section>'
)


def instruction_section(i, name, notes="", description=""):
    category, title = INSTRUCTIONS[name]
    return (
        f'<section id="sec-{name.replace(".", "-")}"><h4>9.7.{i}. {category}: {title} <a></a></h4>'
        '<p class="rubric">Syntax</p><div class="highlight-text"><pre>demo.form  d, a;</pre></div>'
        f'<p class="rubric">Description</p>{description}'
        f'<p class="rubric">Target ISA Notes</p>{notes}<p class="rubric">Examples</p></section>'
    )


def page(omit=(), include_smem=True, notes=None, descriptions=None, title="1. Introduction — PTX ISA 9.5 documentation"):
    notes, descriptions = notes or {}, descriptions or {}
    parts = [f"<html><head><title>{title}</title></head><body>", TARGET]
    if include_smem:
        parts.append(SMEM)
    for i, name in enumerate(INSTRUCTIONS, start=1):
        if name not in omit:
            parts.append(instruction_section(i, name, notes.get(name, "<p>Requires sm_70 or higher.</p>"), descriptions.get(name, "")))
    parts.append("</body></html>")
    return "".join(parts)


CVT_NOTES = (
    "<p>The f4 forms are supported on following architectures:</p>"
    '<ul class="simple"><li>sm_100a</li><li>sm_120a</li>'
    "<li>And on family-specific architectures from PTX ISA version 8.8:<ul><li>sm_100f or higher in the same family</li>"
    "<li>sm_120f or higher in the same family</li></ul></li></ul>"
)


def entries(inv, name, kind=None):
    return [e for e in inv["instructions"] if e["name"] == name and (kind is None or e["kind"] == kind)]


def test_extract_rejects_non_ptx_page():
    ex = load("extract_ptx")
    with pytest.raises(ex.ExtractionError, match="not a PTX ISA page"):
        ex.extract("<html><head><title>Sign in</title></head><body>please log in</body></html>")


def test_extract_fails_on_missing_shared_memory_section():
    ex = load("extract_ptx")
    with pytest.raises(ex.ExtractionError, match="static shared memory"):
        ex.extract(page(include_smem=False))


def test_extract_fails_on_missing_instruction_section():
    ex = load("extract_ptx")
    with pytest.raises(ex.ExtractionError, match="ldmatrix"):
        ex.extract(page(omit=("ldmatrix",)))


def test_schema_validates_output_and_every_entry_cites_a_clause():
    ex = load("extract_ptx")
    inv = ex.extract(page(notes={"cvt": CVT_NOTES}))
    schema = json.loads((REPO / "inventory" / "schema.json").read_text())
    jsonschema.validate(inv, schema)
    assert inv["instructions"], "expected entries"
    assert all(e["clause"] and e["anchor"] for e in inv["instructions"])
    assert inv["ptx_version"] == "9.5"
    assert inv["source_sha256"] == hashlib.sha256(page(notes={"cvt": CVT_NOTES}).encode()).hexdigest()


def test_targets_are_the_sm12x_names_from_the_target_directive():
    ex = load("extract_ptx")
    assert ex.extract(page())["targets"] == ["sm_120", "sm_120a", "sm_120f", "sm_121", "sm_121a", "sm_121f"]


def test_static_smem_table_splits_multi_target_cells():
    ex = load("extract_ptx")
    smem = ex.extract(page())["static_smem_kb"]
    assert smem == {"sm_90a": 228, "sm_120a": 100, "sm_121a": 100}


def test_arch_list_with_nested_family_list_is_parsed():
    ex = load("extract_ptx")
    (e,) = entries(ex.extract(page(notes={"cvt": CVT_NOTES})), "cvt", "target_note")
    assert e["archs"] == ["sm_100a", "sm_120a"]
    assert e["family_archs"] == ["sm_100f", "sm_120f"]
    assert e["sm_121a_in_archs"] is False
    assert e["family_lists_sm120f"] is True


def test_requires_or_higher_is_parsed_and_sm89_is_normalised():
    ex = load("extract_ptx")
    notes = "<p>cvt with e4m3x2 requires sm89 or higher.</p><p>.tf32 forms require sm_100 or higher.</p>"
    got = {e["requires"] for e in entries(ex.extract(page(notes={"cvt": notes})), "cvt", "target_note")}
    assert got == {"sm_89", "sm_100"}


def test_arch_specific_requirement_with_family_support_sentence():
    ex = load("extract_ptx")
    notes = "<p>Block scale requires sm_120a and is supported on sm_120f or higher in the same family from PTX ISA version 8.8.</p>"
    (e,) = entries(ex.extract(page(notes={"mma": notes})), "mma", "target_note")
    assert e["requires"] == "sm_120a"
    assert e["archs"] == ["sm_120a"]
    assert e["family_archs"] == ["sm_120f"]
    assert e["sm_121a_in_archs"] is False and e["family_lists_sm120f"] is True


def test_qualifiers_are_collected_from_the_statement():
    ex = load("extract_ptx")
    notes = "<p>Qualifiers .kind::mxf4nvf4 and .kind::mxf4 are supported on following architectures:</p><ul><li>sm_120a</li><li>sm_121a</li></ul>"
    (e,) = entries(ex.extract(page(notes={"mma.sp": notes})), "mma.sp", "target_note")
    assert {".kind::mxf4nvf4", ".kind::mxf4"} <= set(e["qualifiers"])
    assert e["archs"] == ["sm_120a", "sm_121a"] and e["sm_121a_in_archs"] is True


def test_long_statements_are_truncated_to_200_characters():
    ex = load("extract_ptx")
    notes = "<p>" + "word " * 100 + "requires sm_80 or higher.</p>"
    (e,) = entries(ex.extract(page(notes={"cvt": notes})), "cvt", "target_note")
    assert len(e["statement"]) <= 200


def test_x2_packing_order_is_extracted_from_the_description():
    ex = load("extract_ptx")
    desc = (
        "<p>When converting two f32 inputs to e2m1x2, the converted values are packed in d such that the value "
        "converted from input a is stored in the upper 4 bits of d and the value from input b in the lower 4 bits.</p>"
        "<p>An unrelated paragraph about rounding.</p>"
    )
    (e,) = entries(ex.extract(page(descriptions={"cvt": desc})), "cvt", "packing")
    assert e["first_operand_upper"] is True
    assert "e2m1x2" in e["statement"]


def test_approx_error_bounds_are_extracted_for_approx_instructions():
    ex = load("extract_ptx")
    desc = "<p>The approx variant has a maximum ulp error of 1 across the full range of inputs.</p>"
    (e,) = entries(ex.extract(page(descriptions={"rcp": desc})), "rcp", "error_bound")
    assert "ulp error of 1" in e["statement"]


def test_fetch_writes_the_page_and_records_source(tmp_path):
    fetch = load("fetch_ptx")
    src = tmp_path / "src.html"
    src.write_text(page())
    dest, meta = tmp_path / "cache" / "ptx.html", tmp_path / "SOURCE.json"
    assert fetch.fetch(dest, url=src.as_uri(), source_json=meta) == dest
    assert dest.read_bytes() == src.read_bytes()
    rec = json.loads(meta.read_text())
    assert rec["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert rec["url"] == src.as_uri() and rec["fetched_utc"]


@pytest.mark.skipif(not REAL_PAGE.exists(), reason="run inventory/fetch_ptx.py first")
def test_real_page_facts():
    ex = load("extract_ptx")
    inv = ex.extract(REAL_PAGE.read_text(errors="replace"))
    assert inv["ptx_version"] == "9.5"
    assert inv["static_smem_kb"]["sm_121a"] == 100
    assert inv["static_smem_kb"]["sm_120a"] == 100
    assert inv["static_smem_kb"]["sm_90a"] == 228
    # cvt to e2m1x2 from f32: arch-specific list names sm_120a but not sm_121a; family list names sm_120f.
    cvt = [e for e in entries(inv, "cvt", "target_note") if "e2m1x2" in e["statement"] and "f32" in e["statement"]]
    assert cvt, "no cvt e2m1x2 .f32 statement found"
    assert "sm_120a" in cvt[0]["archs"]
    assert cvt[0]["sm_121a_in_archs"] is False
    assert cvt[0]["family_lists_sm120f"] is True
    # CORRECTION (see inventory/OPEN_QUESTIONS.md): block-scale on plain mma is stated as 'requires sm_120a' plus
    # family sm_120f; sm_121a is NOT named. The .kind::mxf4nvf4 / .kind::mxf4 sentence naming sm_120a AND sm_121a
    # belongs to mma.sp (sparse).
    block = [e for e in entries(inv, "mma", "target_note") if ".block_scale" in e["statement"]]
    assert block and block[0]["requires"] == "sm_120a"
    assert block[0]["sm_121a_in_archs"] is False and block[0]["family_lists_sm120f"] is True
    sp = [e for e in entries(inv, "mma.sp", "target_note") if e["statement"].startswith("Qualifiers .kind::mxf4nvf4")]
    assert sp and {"sm_120a", "sm_121a"} <= set(sp[0]["archs"])
    # x2 packing order is documented: first operand goes to the upper half.
    pack = [e for e in entries(inv, "cvt", "packing") if "e2m1x2" in e["statement"]]
    assert pack and pack[0]["first_operand_upper"] is True
    assert entries(inv, "rcp", "error_bound") and entries(inv, "ex2", "error_bound")


def test_paragraph_before_any_rubric_is_ignored_not_a_crash():
    ex = load("extract_ptx")
    html = page().replace("rcp <a></a></h4>", "rcp <a></a></h4><p>stray ulp error 3 note.</p>")
    assert "stray ulp error 3 note." in html
    inv = ex.extract(html)
    assert not [e for e in inv["instructions"] if "stray" in e["statement"]]
