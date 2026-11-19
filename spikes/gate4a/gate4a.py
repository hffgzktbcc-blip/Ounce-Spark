# Gate 4a: zero-fitted-parameter decode roofline vs public Spark readings.
# pred_tps = BW_eff / bytes_per_token ; BW_eff = 0.9 * 273 GB/s (fixed in advance, not fitted)
import statistics as st
GiB=1.073741824
PEAK=273.0; BW=0.9*PEAK
# (cell, reading, model bytes GB, active fraction, measured tok/s, provenance)
rows=[
 ("gpt-oss-120B MXFP4 llama.cpp","b6922 k6.17.1", 59.02*GiB, 5.1/116.83, 60.57,"VERIFIED fetched llama.cpp#16578 (size,params); active 5.1B=MEMORY"),
 ("gpt-oss-120B MXFP4 llama.cpp","b6816",         59.02*GiB, 5.1/116.83, 45.87,"VERIFIED fetched llama.cpp#16578"),
 ("gpt-oss-120B MXFP4 llama.cpp","b6816 unified-mem env", 59.02*GiB, 5.1/116.83, 40.61,"VERIFIED fetched; config variant"),
 ("gpt-oss-120B MXFP4 (llama.cpp file)","verdictbits", 59.02*GiB, 5.1/116.83, 58.48,"SECONDARY search snippet"),
 ("gpt-oss-120B MXFP4 (NVIDIA blog)","stack unknown", 59.02*GiB, 5.1/116.83, 55.37,"SECONDARY vendor, snippet"),
 ("gpt-oss-20B MXFP4 (NVIDIA blog)","stack unknown", 12.1, 3.6/20.9, 82.74,"SECONDARY; size 12.1GB & active 3.6B = MEMORY"),
 ("Llama3.1-8B FP8 SGLang b=1","LMSYS", 8.03, 1.0, 20.5,"SECONDARY snippet; 8.03B params x1B = MEMORY"),
 ("Llama3.1-70B FP8 SGLang b=1","LMSYS", 70.6, 1.0, 2.7,"SECONDARY snippet; 70.6B params x1B = MEMORY"),
]
print(f"BW_eff = 0.9 x {PEAK} = {BW:.1f} GB/s (fixed before looking at data)\n")
print(f"{'cell':38}{'reading':24}{'GB/tok':>7}{'pred':>7}{'meas':>7}{'err%':>7}{'meas/pred':>10}")
errs=[];ratios=[];cells={}
for c,r,b,f,m,p in rows:
    gb=b*f; pred=BW/gb; e=(pred/m-1)*100; ratio=m/pred
    print(f"{c:38}{r:24}{gb:7.2f}{pred:7.1f}{m:7.2f}{e:7.0f}{ratio:10.2f}")
    errs.append(abs(e)); ratios.append(ratio); cells.setdefault(c.split(' (')[0].split(' llama')[0],[]).append(ratio)
print(f"\nreadings n={len(rows)}; median |err| = {st.median(errs):.0f}%; mean = {st.mean(errs):.0f}%")
print(f"measured/pred: min {min(ratios):.2f} max {max(ratios):.2f} median {st.median(ratios):.2f}  (all <1 => one-sided over-prediction: {all(r<1 for r in ratios)})")
# one reading per independent model cell (take median within a cell)
cellmed={k:st.median(v) for k,v in cells.items()}
print("per-cell median meas/pred:", {k:round(v,2) for k,v in cellmed.items()})
print(f"cells n={len(cellmed)}; median of cell medians = {st.median(cellmed.values()):.2f}")
# sensitivity to the bandwidth assumption using the reported achieved range 177..262
for bw in (177,200,262):
    r=[m/((bw/(b*f))) for _,_,b,f,m,_ in rows]
    print(f"BW={bw}: median meas/pred={st.median(r):.2f}")
# decision rule pre-registered in design doc: median |err| > 40% => stop; <=25% => continue
