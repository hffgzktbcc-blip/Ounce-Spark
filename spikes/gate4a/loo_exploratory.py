# EXPLORATORY, post-hoc (run AFTER seeing gate4a_v2 results). Not the pre-registered test.
# Leave-one-model-out: predict each held-out model's cells using ONE global efficiency factor
# (median measured/pred ratio over cells of the OTHER models). Error = |pred*factor/meas - 1|.
import statistics as st, importlib.util, sys, io, contextlib
spec=importlib.util.spec_from_file_location("g","gate4a_v2.py"); g=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(g)
rows=g.rows  # (model,fmt,stack,meas,pred,err,ratio)
models=sorted({r[0] for r in rows})
errs=[]; per={}
for hm in models:
    others=[r[6] for r in rows if r[0]!=hm]; f=st.median(others)
    for r in rows:
        if r[0]==hm:
            e=abs(r[4]*f/r[3]-1)*100; errs.append(e); per.setdefault(hm,[]).append(e)
print("leave-one-model-out, one global factor, per-reading |error| %:")
for m,v in per.items(): print(f"  held out {m:16} factor from others={st.median([r[6] for r in rows if r[0]!=m]):.2f}  errors={[round(x) for x in v]}")
errs.sort()
print(f"\nreadings={len(errs)}  median={st.median(errs):.0f}%  mean={st.mean(errs):.0f}%  p75={errs[int(.75*len(errs))]:.0f}%  max={max(errs):.0f}%")
print("share of readings within 25%:", f"{sum(e<=25 for e in errs)}/{len(errs)}")
