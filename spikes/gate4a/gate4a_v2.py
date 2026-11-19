# Gate 4a v2: zero-fitted-parameter decode roofline vs public Spark readings (single-stream, batch 1, no speculative decoding).
# pred_tps = (0.9*273 GB/s) / (active_params_B * bits_per_weight / 8).  Formula fixed in advance, not fitted.
# Inputs marked MEM (active params, bits/weight) are from model memory / model names, NOT fetched.
import statistics as st
BW=0.9*273.0
BPW={"MXFP4":4.25,"NVFP4":4.5,"Q4_K_M":4.85,"FP8":8.0}            # MEM
ACT={"gpt-oss-120B":5.1,"gpt-oss-20B":3.6,"Qwen3-30B-A3B":3.3,"Gemma-4-26B-A4B":4.0,"Llama3.1-8B":8.03,"Llama3.1-70B":70.6}  # MEM (Gemma from the 'A4B' name)
# (model, fmt, stack, measured tok/s, provenance)
R=[
 ("gpt-oss-120B","MXFP4","llama.cpp",60.57,"VERIFIED llama.cpp#16578 b6922 k6.17.1"),
 ("gpt-oss-120B","MXFP4","llama.cpp",45.87,"VERIFIED llama.cpp#16578 b6816"),
 ("gpt-oss-120B","MXFP4","llama.cpp",58.48,"SECONDARY verdictbits"),
 ("gpt-oss-120B","MXFP4","vendor?",55.37,"SECONDARY NVIDIA blog, stack unknown"),
 ("gpt-oss-20B","MXFP4","vendor?",82.74,"SECONDARY NVIDIA blog, stack unknown"),
 ("Llama3.1-8B","FP8","SGLang",20.5,"SECONDARY LMSYS"),
 ("Llama3.1-70B","FP8","SGLang",2.7,"SECONDARY LMSYS"),
 ("gpt-oss-120B","MXFP4","vLLM",60.7,"VERIFIED-RAW field-notes results.jsonl (MIT)"),
 ("gpt-oss-120B","MXFP4","vLLM",60.4,"VERIFIED-RAW field-notes results.jsonl (eugr)"),
 ("gpt-oss-120B","MXFP4","Ollama",42.1,"VERIFIED-RAW field-notes RESULTS.md"),
 ("Qwen3-30B-A3B","NVFP4","vLLM",77.1,"VERIFIED-RAW field-notes results.jsonl"),
 ("Qwen3-30B-A3B","Q4_K_M","Ollama",85.2,"VERIFIED-RAW field-notes RESULTS.md"),
 ("Gemma-4-26B-A4B","Q4_K_M","Ollama",49.6,"VERIFIED-RAW field-notes RESULTS.md"),
 ("Gemma-4-26B-A4B","NVFP4","vLLM",30.3,"VERIFIED-RAW field-notes (no spec-decoding)"),
 ("Gemma-4-26B-A4B","NVFP4","vLLM",32.6,"VERIFIED-RAW field-notes (no spec-decoding)"),
]
EXCLUDED="qwen3-coder-next, GLM-4.7-Flash (active params unknown); all MTP/speculative-decoding rows; llama.cpp unified-memory-env variant"
rows=[]
print(f"BW_eff = 0.9 x 273 = {BW:.1f} GB/s fixed in advance\n")
print(f"{'model':17}{'fmt':8}{'stack':10}{'GB/tok':>7}{'pred':>7}{'meas':>7}{'err%':>6}{'m/p':>6}  provenance")
for m,f,s,meas,p in R:
    gb=ACT[m]*BPW.get(f,0)/8 if f!="FP8" else ACT[m]*1.0
    pred=BW/gb; e=(pred/meas-1)*100; ratio=meas/pred
    rows.append((m,f,s,meas,pred,e,ratio)); print(f"{m:17}{f:8}{s:10}{gb:7.2f}{pred:7.1f}{meas:7.1f}{e:6.0f}{ratio:6.2f}  {p}")
ab=[abs(r[5]) for r in rows]; ra=[r[6] for r in rows]
print(f"\nn readings={len(rows)}  median|err|={st.median(ab):.0f}%  mean={st.mean(ab):.0f}%  measured/pred: min {min(ra):.2f} max {max(ra):.2f} median {st.median(ra):.2f}  all<1: {all(x<1 for x in ra)}")
cells={}
for m,f,s,meas,pred,e,r in rows: cells.setdefault((m,f,s),[]).append(r)
cm={k:st.median(v) for k,v in cells.items()}
print(f"independent (model,format,stack) cells = {len(cm)}; median of cell medians = {st.median(cm.values()):.2f}")
for k,v in sorted(cm.items(), key=lambda kv: kv[1]): print(f"   {k[0]:17}{k[1]:8}{k[2]:10} {v:.2f}")
print("\nper-stack (not pooled across stacks) median measured/pred:")
by={}
for (m,f,s),v in cm.items(): by.setdefault(s,[]).append(v)
for s,v in by.items(): print(f"   {s:10} cells={len(v)} median={st.median(v):.2f} range {min(v):.2f}-{max(v):.2f}")
print("\nper-format median measured/pred over cells:")
bf={}
for (m,f,s),v in cm.items(): bf.setdefault(f,[]).append(v)
for f,v in bf.items(): print(f"   {f:8} cells={len(v)} median={st.median(v):.2f} range {min(v):.2f}-{max(v):.2f}")
# two-format check on the SAME model and stack (Qwen3-30B-A3B, vLLM): FP8 ~53 (approx, author table) vs NVFP4 77.1
meas_ratio=77.1/53.0; bytes_ratio=(8.0)/(4.5)
print(f"\nTwo-format check (Qwen3-30B-A3B, vLLM): measured speedup FP8->NVFP4 = {meas_ratio:.2f}; predicted from bytes = {bytes_ratio:.2f}; "
      f"difference = {(meas_ratio/bytes_ratio-1)*100:.0f}%  (rule: >30% means per-format terms needed). FP8 figure is '~53' in the author's notes.")
print("EXCLUDED:",EXCLUDED)
