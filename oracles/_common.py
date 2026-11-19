"""Helpers shared by the oracle generators."""
import hashlib
import math
import pathlib


def hexf(x: float) -> str:
    """float.hex() with nan/inf spelled out, so values round-trip through JSON."""
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x.hex()


def write_manifest(out_dir: pathlib.Path) -> None:
    """MANIFEST.sha256 next to `out_dir`: one `sha256  <dir>/<file>` line per file currently in it."""
    out_dir = pathlib.Path(out_dir)
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {out_dir.name}/{p.name}" for p in sorted(out_dir.iterdir()) if p.is_file()]
    (out_dir.parent / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
