#!/usr/bin/env python3
"""Extract the sm_12x facts the project needs from NVIDIA's public PTX ISA page (Task 2, Gate 0b).

Works from document structure (sections, rubric blocks, tables, nested lists), never from flattened text.
Fails loudly (ExtractionError) when an expected section is missing: no partial inventory is ever emitted.
Excerpts of NVIDIA text are capped at 200 characters per entry; the HTML itself is never stored.
"""
import hashlib
import json
import pathlib
import re
import sys

from bs4 import BeautifulSoup

ROOT = pathlib.Path(__file__).resolve().parents[1]
HEADING = re.compile(r"^h[1-6]$")
TITLE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(.*?):\s+(.*)$")
MAX_EXCERPT = 200
APPROX = {"rcp", "sqrt", "rsqrt", "sin", "cos", "lg2", "ex2", "tanh"}
WANTED = {
    "mma": lambda t: t == "mma",
    "mma.sp": lambda t: t.startswith("mma.sp"),
    "cvt": lambda t: t == "cvt",
    "cvt.pack": lambda t: t == "cvt.pack",
    "cp.async": lambda t: t == "cp.async",
    "ldmatrix": lambda t: t == "ldmatrix",
    "shfl.sync": lambda t: t == "shfl.sync",
    "bar": lambda t: t.split(",")[0].strip() == "bar",
    "atom": lambda t: t == "atom",
    "red": lambda t: t == "red",
    **{n: (lambda t, n=n: t == n) for n in sorted(APPROX)},
}
QUALIFIER = re.compile(r"\.[A-Za-z][A-Za-z0-9_]*(?:::[A-Za-z0-9_]+)*")
REQUIRES = re.compile(r"\brequires?\s+(sm_?\d+[a-z]?)\b")
FAMILY_SENTENCE = re.compile(r"supported on (sm_?\d+f)\b")
LEADING_TARGET = re.compile(r"^(sm_\d+[a-z]?)\b(\s+or higher in the same family)?")
ERROR_WORDS = re.compile(r"ulp|relative error|absolute error|maximum error", re.I)


class ExtractionError(Exception):
    pass


def _text(node) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).replace("", "").strip()


def _excerpt(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= MAX_EXCERPT else text[: MAX_EXCERPT - 1] + "…"


def _norm_target(token: str) -> str:
    return re.sub(r"^sm(\d)", r"sm_\1", token)


def _sections(soup):
    for sec in soup.find_all("section"):
        heading = sec.find(HEADING, recursive=False)
        if heading is not None:
            yield sec, _text(heading)


def _rubric(sec, name):
    """Direct children of `sec` that follow the rubric paragraph called `name`, up to the next rubric."""
    out, grab = [], False
    for child in sec.children:
        if not getattr(child, "name", None):
            continue
        if child.name == "p" and "rubric" in (child.get("class") or []):
            grab = _text(child) == name
            continue
        if grab:
            out.append(child)
    return out


def _arch_lists(ul):
    """(archs, family_archs) from a possibly nested list, classified by each item's own leading text."""
    archs, family = [], []
    for li in ul.find_all("li"):
        own = " ".join(
            c.get_text(" ") if getattr(c, "name", None) else str(c)
            for c in li.children
            if getattr(c, "name", None) not in ("ul", "ol")
        )
        m = LEADING_TARGET.match(re.sub(r"\s+", " ", own).strip())
        if not m:
            continue
        target = m.group(1)
        bucket = family if m.group(2) else archs
        if target not in bucket:
            bucket.append(target)
    return archs, family


def _target_note(statement: str, ul):
    archs, family = _arch_lists(ul) if ul is not None else ([], [])
    m = REQUIRES.search(statement)
    requires = _norm_target(m.group(1)) if m else None
    if requires and requires.endswith("a") and requires not in archs:
        archs.append(requires)
    for fam in FAMILY_SENTENCE.findall(statement):
        fam = _norm_target(fam)
        if fam not in family:
            family.append(fam)
    return {
        "qualifiers": sorted(set(QUALIFIER.findall(statement))),
        "requires": requires,
        "archs": archs,
        "family_archs": family,
        "sm_121a_in_archs": "sm_121a" in archs,
        "family_lists_sm120f": "sm_120f" in family,
    }


FORM_LINE = re.compile(r"^mma\.sync\.aligned\.(\S+)\s")
DEF_LINE = re.compile(r"^\.(\w+)\s*=\s*\{([^}]*)\}")
SHAPE_TOKEN = re.compile(r"^m\d+n\d+k\d+$")
LAYOUT_TOKENS = {"row", "col", "alayout", "blayout"}
MODIFIER_TOKENS = {"kind", "block_scale", "scale_vec_size", "satfinite"}


def _members(text: str):
    return [m.strip().lstrip(".") for m in text.split(",") if m.strip()]


def _resolve_form(line: str, spec: str, defs: dict, base: dict) -> dict:
    """One `mma.sync.aligned.<spec>` line plus the named type sets defined after it."""
    tokens = re.sub(r"\{[^}]*\}", "", spec).split(".")
    shape, rest = tokens[0], [t for t in tokens[1:] if t not in LAYOUT_TOKENS]
    has_kind = "kind" in rest
    rest = [t for t in rest if t not in MODIFIER_TOKENS]
    d, a, b, c = (defs.get(t, [t]) for t in rest[:4])
    return dict(
        base,
        kind="form",
        statement=_excerpt(line),
        shapes=defs.get("shape", []) if shape == "shape" else ([shape] if SHAPE_TOKEN.match(shape) else []),
        d_types=d, a_types=a, b_types=b, c_types=c,
        kinds=defs.get("kind", []) if has_kind else [],
        scale_types=defs.get("stype", []) if "stype" in rest[4:] else [],
    )


def _parse_forms(text: str, base: dict):
    """Definitions (`.name = {..}`) apply to every instruction line since the previous definitions group."""
    out, pending, defs, after_defs = [], [], {}, False
    for raw in text.splitlines():
        line = raw.strip()
        if m := FORM_LINE.match(line + " "):
            if after_defs:
                out += [_resolve_form(ln, sp, defs, base) for ln, sp in pending]
                pending, defs, after_defs = [], {}, False
            pending.append((line, m.group(1)))
        elif d := DEF_LINE.match(line):
            defs[d.group(1)] = _members(d.group(2))
            after_defs = True
    return out + [_resolve_form(ln, sp, defs, base) for ln, sp in pending]


def _prose(sec):
    """Paragraphs and tables outside the syntax, ISA-notes and examples regions (rubrics can nest, e.g. under Notes)."""
    skip, current = {"Syntax", "PTX ISA Notes", "Target ISA Notes", "Examples"}, None
    for child in sec.children:
        if not getattr(child, "name", None):
            continue
        if child.name == "p" and "rubric" in (child.get("class") or []):
            current = _text(child)
        elif child.name in ("p", "table") and current is not None and current not in skip and not current.startswith("Examples"):
            yield child


def _entries_for(sec, name, number):
    base = {"name": name, "clause": number, "anchor": sec.get("id") or ""}
    syntax = ""
    blocks = _rubric(sec, "Syntax")
    if blocks:
        syntax = _excerpt(blocks[0].get_text(" "))
    out = []
    if name == "mma":
        for block in _rubric(sec, "Syntax"):
            if block.name in ("div", "pre"):
                out += _parse_forms(block.get_text(), base)
    notes = _rubric(sec, "Target ISA Notes")
    i = 0
    while i < len(notes):
        node = notes[i]
        if node.name == "p":
            ul = notes[i + 1] if i + 1 < len(notes) and notes[i + 1].name in ("ul", "ol") else None
            statement = _text(node)
            entry = dict(base, kind="target_note", statement=_excerpt(statement), syntax=syntax)
            entry.update(_target_note(statement, ul))
            out.append(entry)
            i += 2 if ul is not None else 1
        else:
            i += 1
    descr = _rubric(sec, "Description") + _rubric(sec, "Notes")
    if name in ("cvt", "cvt.pack"):
        for node in descr:
            text = _text(node)
            if node.name == "p" and "packed in" in text:
                upper = None
                if re.search(r"input a is stored in the upper", text):
                    upper = True
                elif re.search(r"input a is stored in the lower", text):
                    upper = False
                out.append(dict(base, kind="packing", statement=_excerpt(text), first_operand_upper=upper))
    if name in APPROX:
        error_intro = sub_intro = ""
        for node in _prose(sec):
            text = _text(node)
            if node.name == "table":
                # e.g. sin/cos: the bound is a table of ranges and errors under an introducing sentence ("The maximum
                # absolute error ...") and a sub-heading ("For radians as input:"); keep both so the kind of error survives
                if re.search(r"\bError\b", text) and re.search(r"\d", text):
                    parts = [error_intro, sub_intro if sub_intro != error_intro else "", text]
                    out.append(dict(base, kind="error_bound", statement=_excerpt(" ".join(x for x in parts if x))))
                continue
            sub_intro = text
            if re.search(r"\berror\b", text, re.I):
                error_intro = text
            hits = [s for s in re.split(r"(?<=\.)\s+", text) if ERROR_WORDS.search(s) and re.search(r"\d", s)]
            if hits:
                out.append(dict(base, kind="error_bound", statement=_excerpt(" ".join(hits))))
    return out


def _require_content(name: str, kinds: set) -> None:
    """A section that parsed but yielded nothing means the page structure changed: say so instead of emitting less."""
    hint = "the page structure may have changed"
    if "target_note" not in kinds:
        raise ExtractionError(f"no Target ISA Notes statements found for {name}: {hint}")
    if name == "mma" and "form" not in kinds:
        raise ExtractionError(f"no instruction forms found for {name}: {hint}")
    if name == "cvt" and "packing" not in kinds:
        raise ExtractionError(f"no x2 packing statement found for {name}: {hint}")
    if name in APPROX and "error_bound" not in kinds:
        raise ExtractionError(f"no error bound found for approx instruction {name}: {hint}")


def extract(html: str, source_sha256: str | None = None) -> dict:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    version = re.search(r"PTX ISA (\d+\.\d+)", title)
    if not version:
        raise ExtractionError("not a PTX ISA page")
    sections = list(_sections(soup))

    target_secs = [s for s, t in sections if t.endswith(": .target")]
    if not target_secs or target_secs[0].find("pre") is None:
        raise ExtractionError("target directive section (.target) not found")
    targets = sorted(set(re.findall(r"sm_12\d[af]?", target_secs[0].find("pre").get_text())))

    smem = {}
    table = next((t for t in soup.find_all("table") if "Maximum statically allocated shared memory" in _text(t)), None)
    if table is None:
        raise ExtractionError("static shared memory table not found")
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) == 2 and (size := re.search(r"(\d+)\s*KB", _text(cells[1]))):
            for target in _text(cells[0]).split(","):
                smem[target.strip()] = int(size.group(1))
    if not smem:
        raise ExtractionError("static shared memory table has no rows")

    instructions = []
    for name, matches in WANTED.items():
        found, mine = False, []
        for sec, heading in sections:
            m = TITLE.match(heading)
            if m and matches(m.group(3).strip()):
                found = True
                mine.extend(_entries_for(sec, name, m.group(1)))
        if not found:
            raise ExtractionError(f"instruction section not found: {name}")
        _require_content(name, {e["kind"] for e in mine})
        instructions.extend(mine)
    instructions.sort(key=lambda e: [int(p) for p in e["clause"].split(".")])
    return {
        "ptx_version": version.group(1),
        "source_sha256": source_sha256 or hashlib.sha256(html.encode("utf-8")).hexdigest(),
        "targets": targets,
        "static_smem_kb": smem,
        "instructions": instructions,
    }


def main(src: pathlib.Path = ROOT / ".cache" / "ptx.html", out: pathlib.Path = ROOT / "inventory" / "ptx_sm12x.json") -> int:
    raw = pathlib.Path(src).read_bytes()
    inv = extract(raw.decode("utf-8", errors="replace"), hashlib.sha256(raw).hexdigest())
    pathlib.Path(out).write_text(json.dumps(inv, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}: {len(inv['instructions'])} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
