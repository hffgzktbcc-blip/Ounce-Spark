#!/usr/bin/env python3
"""Build data/spark_decode.csv: the 15 readings behind Gate 4a (spec section 15) with their provenance (Task 7, Gate 3a).

Eight rows are read from the raw, MIT-licensed field-notes results file; seven are hand-transcribed from secondary sources.
Inputs marked in `notes` (active parameters, bits per weight) come from model memory or model names, not from the sources.
"""
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
JSONL = ROOT / "spikes" / "gate4a" / "field_notes_results.jsonl"
OUT = ROOT / "data" / "spark_decode.csv"
FIELDS = ["id", "source_url", "source_kind", "operator", "model", "active_params_b", "params_total_b", "fmt", "bits_per_weight",
          "stack", "stack_version", "build", "kernel", "depth", "metric", "batch", "spec_decoding", "value_tps", "value_pm", "date", "notes"]

MODEL = {  # model -> (active params B, total params B), from model memory/names
    "gpt-oss-120B": (5.1, 116.83), "gpt-oss-20B": (3.6, 20.9), "Qwen3-30B-A3B": (3.3, 30.5),
    "Gemma-4-26B-A4B": (4.0, 26.0), "Llama3.1-8B": (8.03, 8.03), "Llama3.1-70B": (70.6, 70.6),
}
BITS = {"MXFP4": 4.25, "NVFP4": 4.5, "Q4_K_M": 4.85, "FP8": 8.0}  # from model memory
MEMORY_NOTE = "active_params_b and bits_per_weight come from model memory/names, not from the source"
FIELD_NOTES_URL = "https://github.com/sergioamsilva/dgx-spark-field-notes"
LLAMACPP_URL = "https://github.com/ggml-org/llama.cpp/discussions/16578"
LLAMACPP_NOTE = "read via a page-summary tool (the raw page returned HTTP 403); verbatim row text; tg32 at depth 0; " + MEMORY_NOTE


def base(**kw):
    m = MODEL[kw["model"]]
    row = {k: "" for k in FIELDS}
    row.update(active_params_b=m[0], params_total_b=m[1], bits_per_weight=BITS[kw["fmt"]], depth=0, metric="tg", batch=1, **kw)
    return row


def manual_rows():
    return [
        base(id="llamacpp-16578-b6922", source_url=LLAMACPP_URL, source_kind="secondary", operator="eugr", model="gpt-oss-120B", fmt="MXFP4",
             stack="llama.cpp", build="7db35a7 (6922)", kernel="6.17.1 (Fedora 43 Server)", spec_decoding="false", value_tps=60.57, value_pm=0.25, notes=LLAMACPP_NOTE),
        base(id="llamacpp-16578-b6816", source_url=LLAMACPP_URL, source_kind="secondary", operator="eugr", model="gpt-oss-120B", fmt="MXFP4",
             stack="llama.cpp", build="03792ad (6816)", spec_decoding="false", value_tps=45.87, value_pm=0.74, notes="kernel not stated; " + LLAMACPP_NOTE),
        base(id="verdictbits-gptoss120b", source_url="https://verdictbits.com/reviews/dgx-spark-benchmarks/", source_kind="secondary", operator="verdictbits",
             model="gpt-oss-120B", fmt="MXFP4", stack="llama.cpp", spec_decoding="false", value_tps=58.48,
             notes="review site quoting llama.cpp's results file; seen only as a search snippet; " + MEMORY_NOTE),
        base(id="nvidia-blog-gptoss120b", source_url="https://developer.nvidia.com/blog/?p=107650", source_kind="secondary", operator="nvidia-blog",
             model="gpt-oss-120B", fmt="MXFP4", stack="unknown", spec_decoding="unknown", value_tps=55.37, notes="vendor figure, stack not stated; search snippet; " + MEMORY_NOTE),
        base(id="nvidia-blog-gptoss20b", source_url="https://developer.nvidia.com/blog/?p=107650", source_kind="secondary", operator="nvidia-blog",
             model="gpt-oss-20B", fmt="MXFP4", stack="unknown", spec_decoding="unknown", value_tps=82.74, notes="vendor figure, stack not stated; search snippet; " + MEMORY_NOTE),
        base(id="lmsys-llama31-8b-fp8", source_url="https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark", source_kind="secondary", operator="lmsys",
             model="Llama3.1-8B", fmt="FP8", stack="SGLang", spec_decoding="false", value_tps=20.5, notes="batch 1; search snippet; " + MEMORY_NOTE),
        base(id="lmsys-llama31-70b-fp8", source_url="https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark", source_kind="secondary", operator="lmsys",
             model="Llama3.1-70B", fmt="FP8", stack="SGLang", spec_decoding="false", value_tps=2.7, notes="batch 1; search snippet; " + MEMORY_NOTE),
    ]


# results.jsonl label -> row metadata. The label is the key; the value is read from the raw file.
FIELD_NOTES = [
    ("GPT-OSS-120B MXFP4/CUTLASS (vLLM, solo)", dict(model="gpt-oss-120B", fmt="MXFP4", stack="vLLM", spec_decoding="false", notes="")),
    ("GPT-OSS-120B MXFP4/CUTLASS (vLLM eugr, solo)", dict(model="gpt-oss-120B", fmt="MXFP4", stack="vLLM", spec_decoding="false", notes="eugr/spark-vllm image")),
    ("Qwen3-30B-A3B-NVFP4 (vLLM eugr, solo)", dict(model="Qwen3-30B-A3B", fmt="NVFP4", stack="vLLM", spec_decoding="false", notes="eugr/spark-vllm image")),
    ("Gemma-4-26B-A4B-NVFP4 (vLLM, solo, no spec-decoding)", dict(model="Gemma-4-26B-A4B", fmt="NVFP4", stack="vLLM", spec_decoding="false", notes="")),
    ("Gemma-4-26B-A4B-NVFP4 (vLLM, solo, SEM spec-decoding)", dict(model="Gemma-4-26B-A4B", fmt="NVFP4", stack="vLLM", spec_decoding="unknown",
         notes="results.jsonl label says 'SEM spec-decoding' but the README and RESULTS.md call this 30.3 tok/s figure 'no spec-decoding'; the source is inconsistent")),
    ("gpt-oss:120b (Ollama, same harness)", dict(model="gpt-oss-120B", fmt="MXFP4", stack="Ollama", spec_decoding="false", notes="label says 'Ollama Q4'; the model ships as MXFP4 (assumed)")),
    ("qwen3:30b-a3b (Ollama, same harness)", dict(model="Qwen3-30B-A3B", fmt="Q4_K_M", stack="Ollama", spec_decoding="false", notes="")),
    ("gemma4-26b:latest (Ollama, same harness)", dict(model="Gemma-4-26B-A4B", fmt="Q4_K_M", stack="Ollama", spec_decoding="false", notes="")),
]


def field_notes_rows(jsonl=JSONL):
    by_label = {}
    for line in pathlib.Path(jsonl).read_text().splitlines():
        d = json.loads(line)
        by_label[d["label"]] = d
    rows = []
    for n, (label, meta) in enumerate(FIELD_NOTES, start=1):
        if label not in by_label:
            raise KeyError(f"label not found in results.jsonl: {label!r}")
        note = "; ".join(x for x in (meta["notes"], "single-stream decode, max_tokens=400, one operator and machine; software versions in the repo's benchmarks/RESULTS.md", MEMORY_NOTE) if x)
        row = base(id=f"fieldnotes-{n}", source_url=FIELD_NOTES_URL, source_kind="verified_raw", operator="sergioamsilva", date="2026-08-10",
                   value_tps=by_label[label]["single_tok_s"], **{k: v for k, v in meta.items() if k != "notes"}, notes=note)
        rows.append(row)
    return rows


def rows():
    return manual_rows() + field_notes_rows()


def main(out=OUT):
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows())
    print(f"wrote {out}: {len(rows())} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
