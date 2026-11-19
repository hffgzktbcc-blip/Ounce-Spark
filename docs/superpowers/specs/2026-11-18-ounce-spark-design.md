# ounce-spark: a free, evidence-first stand-in for the NVIDIA DGX Spark

Status: **DRAFT v3.2 for owner approval** (2026-11-18), revised after four independent reviews (section 13);
D1, D2 and D3 resolved by the owner (section 12). Design approval itself is still pending.
Nothing has been built. No repository exists yet.

Every claim carries a tag: **[verified]** (read in a file or source this session), **[secondary]** (forums,
blogs, news or search snippets; no primary NVIDIA document seen), **[memory]** (model knowledge, unchecked),
**[my call]** (a decision I made, for the owner to overrule), **[open]** (needs an owner decision, section 12).
Several external pages (arXiv, NVIDIA forums, docs.nvidia.com, LMSYS) were not reachable from the cloud
container; claims taken from them are **[secondary]**, from search snippets, not read.

## 1. Intent and why

The owner has no DGX Spark and no budget to rent GPU time. They want a stand-in: something that behaves like a
Spark closely enough that code written for it can be developed and tested, and that estimates how a workload
would perform on it. They already build evidence-first emulation tooling (the Ounce family) and want a new
sibling in that family.

Two questions are kept apart:

1. **Does it compute the right answer?** Emulation can answer this.
2. **How fast would it run?** Only a *model*, calibrated on real Spark measurements, can answer this. Emulator
   wall-clock time is never a performance claim.

An emulator does not supply GPU *power*; it runs on an ordinary CPU and is slower than the real machine. Its
value is development, testing and prediction.

## 2. Goals

- G1. Execute GPU kernels on an ordinary CPU at the **PTX level** (NVIDIA's public virtual ISA) with Spark-relevant
  semantics, and compare results with independent references.
- G2. Model the tensor-core `mma.sync` family for the formats the Spark supports (FP16, BF16, TF32, FP8, FP6, FP4,
  INT8) with spec-derived oracles.
- G3. Predict LLM inference throughput on a Spark with **published error bars**, scored on held-out workloads.
- G4. Provide a harness any Spark owner can run to return measurements, so claims can reach "independently
  reproduced".
- G5. Keep every claim at the maturity label it has earned.

## 3. Non-goals

- Cycle-accurate or bit-exact-with-silicon emulation. NVIDIA does not publish the Blackwell SASS encoding or
  microarchitecture. [verified for the decoder in ounce-core; secondary for Blackwell]
- Running arbitrary CUDA applications, PyTorch, vLLM, or closed NVIDIA libraries (cuBLAS, cuDNN).
- Executing SASS (see 9.3). SASS is at most studied, later.
- A full-system emulator of the Spark's Arm CPU half (cut from v1).
- Graphics on the Spark.

## 4. Anti-goals

- Presenting a model prediction as a measurement, or an emulator match as hardware agreement.
- Any "runs like a Spark" claim without Gate 4b's error bars and sample-size conditions.
- Reverse engineering NVIDIA's driver or closed libraries; redistributing NVIDIA binaries.
- Translating NVIDIA toolchain output to run on non-NVIDIA hardware before the owner has read and accepted the
  applicable licence terms (section 7, D1).
- Importing code or data from private or GPL sources.
- Quietly guessing: an unmodelled instruction stops the run with its line or offset. [verified: ounce-core rule,
  `src/gpu/sass/mod.rs`; the same idea appears in other tiny PTX interpreters]

## 5. What is known about the Spark

| Fact | Confidence |
|---|---|
| GB10: 20-core Arm CPU + Blackwell GPU, 128 GB unified LPDDR5x | [secondary] |
| **Peak** memory bandwidth 273 GB/s, 256-bit. **Achieved** bandwidth is reported at about 177 to 262 GB/s | [secondary: NVIDIA forum threads, blog] |
| GPU target is compute capability 12.1 (SM121) | [secondary]. CUTLASS's own table lists SM12.1 as DGX Spark, CUDA 13.0+ [reported]. |
| SM121 lacks `tcgen05` and TMEM, uses the `mma.sync` model plus FP4/FP6 | [secondary: forum thread quoting NVIDIA staff, Backend.AI write-up]. No NVIDIA document seen. |
| PTX 8.7 adds sm_120 with e3m2/e2m3/e2m1 and `.block_scale` for `mma`; LLVM lists m16n8k32 and m16n8k64 shapes | [secondary] |
| Block-scaled FP4/FP6 need the `a` target suffix (`sm_120a`/`sm_121a`) or the PTX 8.8 `f` family; plain `sm_120` cannot compile them, and shorthand can silently drop the `a` | [secondary: forum report] |
| The "1 PFLOP" figure is FP4 *with sparsity*; a forum post reports it reached only with 2:4-sparse NVFP4. It is **not** a dense-LLM peak and must not be used as one | [secondary] |
| Decode on Spark runs at about 90% of the memory roofline in one NVIDIA tutorial | [secondary: NVIDIA cuda-rust tutorial, reported] |
| Public Spark benchmark tables exist (see Gate 3 for the source list) | [verified for llama.cpp #16578; others secondary] |
| The same model on the same hardware varied about 25% (tg32 60.57 vs 45.87, gpt-oss-120B MXFP4) | [verified for the two numbers]. **Confounded**: the runs differ in llama.cpp build, OS and kernel; one reviewer could not confirm the "6.11" kernel figure. Cause not isolated. |
| Thermal throttling is reported on some units and not on others | [secondary, conflicting] |

[secondary] items become constants in code only after Gate 0 / Gate 3 re-check them. Bandwidth in particular
is a **fitted parameter**, never a hard-coded constant.

## 6. Existing assets and prior art

### 6.1 Owner's repos (public MIT `ounce-core` unless noted) [verified by two independent reads]

- `src/gpu/sass/` (about 4.5k non-test lines): a **graphics-shader** translator (SASS to SSA IR to WGSL).
  `ir::Function` is straight-line graphics IR (stage, attribute stores, shader-program-header input); no kernel
  entry, memory model, or cubin input. Decoder bit positions come from **SM86 and SM61 only**
  (CuAssembler nvdisasm words); **nothing is SM100/120/121**.
- Integer ops decode only in some forms: `IMAD.X`/negated, `IADD3.X`, `ISETP.EX`, `LOP3` predicate in/out,
  64-bit and funnel `SHF`, `FMNMX.NAN`, `MUFU.F16` and non-32-bit `LDC` are rejected. `BSSY`/`BSYNC`/`WARPSYNC`
  decode to no-op hints; `BRA.DIV`/`.CONV` are rejected. **No reconvergence semantics.** The `mod.rs` header
  ("no control flow or integer ops yet") is stale.
- `src/gpu/tensor.rs` (604 lines): two FP16 shapes (`m16n8k8`, `m16n8k16`), `.row.col`, f32 accumulate in
  increasing-k order. Its own text excludes f16-accumulate, sparsity, integer/bf16/tf32. Going beyond FP16 is a
  **rewrite**; the lane-fragment layout functions are reusable.
- Absent: `LDG/STG/LDS/STS`, barriers, shuffles, atomics, `HMMA` decode, a warp/lane model, shared memory.
- The **method** transfers: spec-first oracles with hashes committed before the emulator runs
  (`docs/corpus-oracles.md`), maturity labels (`docs/LIMITS.md`), provenance and licence registers. The **tools**
  mostly do not: `tools/hwcheck` is an AArch64-only CPU replay, `ounce_verify.py` validates an evidence-JSON
  schema, `tools/testfloat` covers Arm f32/f64 only, `corpus_oracle.py` makes console-frame hashes.
- `decode.rs` cites `nvdisasm_tests.rs`, which is not in the tree; it has no UNVERIFIED tags.
- `ounce-hb` (MIT): `Backend` trait, exit-code taxonomy, `src/report_v1.rs`. `ounce-core` has `run_report.rs` and
  `docs/OUNCE_RUN_V1_SCHEMA.md` (exit codes 0, 10-17). The `deckstream` environment is the template for Gate 0a's
  pinned, checksum-verified setup.

### 6.2 External prior art (licences and status as reported by reviewers; recheck before use)

**Functional PTX execution**
- **ptoxide** (<https://github.com/gvilums/ptoxide>): Rust, Apache-2.0 or MIT [reported verified], learning project
  with many `todo!`s, about 3.5k lines. Has global/shared/local memory and barriers; no mma, shfl or atomics
  mentioned. **The nearest thing to `spark-ptx`/`spark-warp`; fork-or-skim candidate.**
- GPGPU-Sim (<https://github.com/gpgpu-sim/gpgpu-sim_distribution>): licence unconfirmed; claims a tensor-core
  model, validated Volta/Turing-era; no Blackwell/FP4/FP6/block-scale found. Cross-check for plain FP16 only.
- GPU Ocelot (<https://github.com/gtcasl/gpuocelot>): BSD-3-Clause, unmaintained, pre-Volta PTX, no mma.
  Design reference only.
- PTX4CPU (MIT, prototype), PTXLab and a nfcxx PR (tiny subsets, stop on unknown instruction), pyptx
  (<https://github.com/patrick-toulme/pyptx>, Apache-2.0, a PTX DSL with an sm_120 target: possible *generator of test
  PTX*), Triton interpreter (`TRITON_INTERPRET=1`: numpy equivalents of Triton ops, never executes PTX, so a
  semantic cross-check only).
- **No project found that models block-scaled `mma.sync` on a CPU.** [my estimate from searches]

**Numeric oracles (independent of our emulator)**
- **ml_dtypes** (<https://github.com/jax-ml/ml_dtypes>, Apache-2.0): float8 e4m3fn/e5m2, float6 e2m3/e3m2, float4
  e2m1, e8m0; element-level only. Primary element oracle.
- **gfloat** (<https://github.com/graphcore-research/gfloat>, MIT): OCP MX E2M1/E2M3/E3M2/E8M0, FP8, P3109.
  A different implementation from ml_dtypes: two oracles can cross-check each other.
- **CUTLASS** (<https://github.com/NVIDIA/cutlass>, BSD-3): best independent source for scale-factor layout and
  semantics. **Read its code; do not run its output** (also keeps D1 clear).
- torchao `mx_formats` (NVFP4/MX reference; licence unverified), qwantize (pure-PyTorch NVFP4/MXFP4 reference),
  llama.cpp `ggml-quants` (MXFP4 block math; MIT per my memory).
- **None defines `mma` accumulation order.** Gate 1 tolerance bands remain ours to justify from the PTX ISA.

**Performance prediction (code and status as reported)**
- GenZ (<https://github.com/abhibambhaniya/GenZ-LLM-Analyzer>, MIT): operator roofline; custom hardware dicts, so a
  GB10 entry is a config edit. llm-analysis (<https://github.com/cli99/llm-analysis>, Apache-2.0): lower-bound
  latency/memory, 4-bit support, small GPU config. NeuSight (<https://github.com/sitar-lab/NeuSight>, MIT): device
  config file, trained on datacenter GPUs. LLMCompass (<https://github.com/PrincetonUniversity/LLMCompass>,
  BSD-3): hardware-design model. Vidur (MIT) needs a profiling phase on real GPUs, a poor fit. KernelSight-LM:
  paper, no code found.
- **No GB10/Spark profile found in any of them** except a hobby site whose author admitted it understates Spark
  decode.

### 6.3 Consequence

Reuse is real but narrower than first written. We reuse the **method**, bit-field helpers, `tensor.rs` lane
layouts, numeric-conversion code, report schema and CI shape. **The compute IR and executor are not in the repo.**
Whether to fork ptoxide or write them fresh is decided by the Gate 2 spike, not assumed.

## 7. Constraints and boundaries

- **No money.** Free tools and free access only.
- **MIT.** Port only from `ounce-core` (owner's MIT code), Mesa NAK and envytools (MIT, notices in
  `THIRD_PARTY.md`), and other MIT/Apache-2.0 sources with provenance recorded. CuAssembler, turingas, DocumentSASS
  and SASSquatch have **no recorded licence**: read-for-orientation until recorded in `PROVENANCE.md`. [my call]
- **Never used:** `ounce-runtime` and `oboromi` (GPL-3.0); anything from private `ounce-nx2` (sanitized read only,
  nothing needed or cited); retail content, firmware, keys.
- **D1, the NVIDIA licence (open).** Reports (news sites, not the EULA) say the CUDA SDK licence bars reverse
  engineering, decompiling or disassembling SDK **output** "for the purpose of translating such output
  artifacts to target a non-NVIDIA platform" (Tom's Hardware, Overclock3D, Wccftech). Reports disagree on the
  introducing version, and commenters disagree on whether it covers a CPU emulator. No ruling or NVIDIA statement
  was found. A CPU emulator that executes or disassembles `nvcc`-built output is plausibly in scope. A risk to
  resolve, not a legal conclusion. Until the owner reads the current EULA of the exact toolkit version: **no
  `nvcc`/`ptxas` output is executed in the emulator or disassembled in this project.**
- **NVIDIA toolchain access.** NVIDIA's download host and `docs.nvidia.com` were **not reachable** (D2). PyPI is
  reachable and hosts `nvidia-cuda-nvcc` and `nvidia-cuda-cuobjdump` 13.5.37 [verified]. Contents and SM121
  support are **unknown**.
- Public calibration data keeps source URL, build, kernel, stack and date. llama.cpp is MIT, but numbers users post
  in discussions carry no stated licence: keep attribution and URLs; check the licence of any dataset repository
  (e.g. Spark Arena) before ingesting. [my call]
- Not legal advice anywhere in this document.

## 8. Approaches considered

| | Idea | Trade-off | Verdict |
|---|---|---|---|
| A | Performance model only | Cheapest; answers "how fast", not "right answer" | Done first as a baseline (Gate 4a), then extended (4b) |
| **B** | **PTX-level functional emulator + spec-first oracles + calibrated performance model, new Ounce sibling** | Most work; rests on public specs; no NVIDIA binary needed to run | **Chosen** |
| B2 | Same, building the PTX executor by forking ptoxide | Saves the parser/VM; inherits a learning project's gaps | Decided by the Gate 2 spike |
| B' | Same, emulating SASS from `nvcc` output | Reverse-engineered decoder with no Blackwell encodings, plus D1 risk | Rejected for v1 |
| C | Extend GPGPU-Sim/Accel-Sim | Validated only to Turing; needs hardware-recorded traces | Rejected |
| D | Fork `oboromi` | GPL-3.0 | Rejected |
| E | Rent a Blackwell GPU | Best oracle | Rejected: no money |

## 9. Design

### 9.1 Evidence labels

`ounce-core`'s four labels (research-only, bounded-verified, independently reproduced, publicly supported) plus,
[my call]:

- **spec-verified**: matches an oracle derived independently from NVIDIA's *public specification* and committed
  before the emulator ran. Not hardware agreement. PTX-level semantics only.
- **inferred**: anything resting on a reverse-engineered SASS encoding. Never upgraded to spec-verified.

Without Blackwell hardware, GPU claims cap at spec-verified until a Spark owner reproduces them.

### 9.2 Components (working names; builder may restructure)

Rust workspace, `#![forbid(unsafe_code)]` where possible, same CI/`deny.toml` shape as `ounce-core`. [my call]

- `spark-ptx`: PTX parser and compute IR (kernel entry, parameters, state spaces). Fork-or-fresh per the Gate 2
  spike.
- `spark-num`: numeric formats and block-scale handling; oracle generators.
- `spark-warp`: executor with per-thread PTX semantics. Warp-level ops (`mma.sync.aligned`, `shfl`, `bar`) require
  full-warp convergence as PTX's `.aligned` qualifier does; otherwise the run stops (convergence-barrier model;
  unmodelled divergence stops, never guessed). [my call]
- `spark-mem`: global (one unified space), shared (functional; bank conflicts not modelled), local, constant,
  async copy (functional); **memcheck-style checks (out-of-bounds, shared-memory races, uninitialised reads)**,
  which are cheap in a VM and not found in the prior art.
- `spark-perf`: the performance model.
- `spark-cli`: run/diagnose/report, `ounce-hb`-style stable exit codes, versioned JSON.
- **Deferred / cut if time is short:** `spark-sass-lab` (SASS inventory, research only, blocked on D1) and the
  Arm CPU half.

### 9.3 Order and gates

A gate is done only when merged with CI green and its `PLAN.md` box ticked (created in Gate 0a by copying
`ounce-core`'s workflow). Every exit criterion is testable as stated.

**Order:** Gate 4a first (hours), then Gates 0a, 0b, 1 and the Gate 2 spike in parallel, then Gate 2, then Gate 3,
then Gate 4b. Gate 3's data ingestion starts alongside 4a. [my call, from two reviewers' independent advice]

**Gate 4a: zero-parameter roofline baseline (free, hours).** Take 15-20 public Spark rows from llama.cpp #16578.
Predict decode as `0.9 x 273 GB/s / model bytes` with no fitted parameters. Compare with the measurements,
including the conflicting 60.57 and 45.87 readings.
Decision rule: median error **above 40%**, or between-build noise larger than the model's margin means Gate 4b's
targets are meaningless; cut Gate 4b to "document the noise". Median error **25% or lower** means the performance
model is nearly free and effort goes to Gate 2. In between, proceed with caution. [my call, reviewer's rule]
Also run the cheap format check: predict vs reported for the same model in two formats (e.g. Llama 3.1 8B Q4_K_M vs
NVFP4). If the measured ratio differs from the predicted bytes ratio by more than 30%, per-format efficiency
terms are required.
**Outcome and amendment (2026-11-18, D3):** the rule fired on the second pass (section 15); the owner amended it as
recorded in section 12. The original decision rule above is kept unchanged for the record.

**Gate 0a: toolchain feasibility spike (free).** Find what the PyPI wheels contain (ptxas, libnvvm, headers,
host-compiler needs) and whether `sm_121a`/`sm_120a` is accepted, requiring the explicit `a` suffix. Record exact
invocations, outputs, exit codes. **Execution of any `nvcc`/`ptxas` output is blocked on D1;** the spike reports
only what the tools *accept*.
Exit: committed spike report; the fallback rule (if SM121 is not accepted, ISA statements stay [secondary], SM120
is used only for what it can express, and block-scaled FP4/FP6 may be lost).

**Gate 0b: PTX ISA inventory (free, needs D2).** From NVIDIA's public PTX ISA documentation extract, for
sm_120/sm_121: `mma` shapes and type combinations (including `.block_scale` and `kind::` forms), `cp.async`,
`ldmatrix`, `shfl`, `bar`, atomics and `.approx` error bounds, as a machine-readable inventory with clause
citations.
Exit: inventory committed; every Gate 1/2 item traces to a clause.

**Gate 1: spec-first numeric oracles (free).** Rewrite and extend `tensor.rs` for the Gate 0b formats. Element
oracles: **ml_dtypes and gfloat cross-checked against each other**; scale-factor layout read from CUTLASS source.
Expected outputs and hashes committed before the emulator runs. Bit-exact where the spec defines the result;
otherwise a **numeric tolerance band per format with a PTX ISA clause as rationale**; `.approx` operations use the
documented max-error as the band. Block-scaled forms only if Gate 0b shows PTX documents them for the target.
Exit: oracle suite green in CI; per-format table of bit-exact vs banded and why.

**Gate 2 spike (hours to 2 days).** Probe ptoxide for `mma`/`shfl`; then add one shape (`mma.sync` m16n8k16) on top
of it and measure the cost.
Decision rule [reviewer's, my call]: if one shape takes more than 2 days, **rescope Gate 2 to numerics-only**
(Gate 1 plus single-instruction PTX tests, no whole-kernel executor). Otherwise decide fork vs fresh.

**Gate 2: functional compute path (free).** Execute a **hand-written PTX kernel corpus** (no NVIDIA binaries): the
exact list is committed with a sizes table: GEMM per Gate 1 format, reduction, softmax, layernorm, shared-memory
transpose, atomics, async copy. Inputs from a committed manifest (zeros, subnormals, max, NaN/Inf, fixed RNG
seed). References are numpy/CPU, **committed before the emulator runs**. Memcheck checks on by default. Triton's
CPU interpreter may cross-check Triton-written kernels (debug tool, not a hardware oracle).
Exit: every corpus kernel matches its reference within the Gate 1 band on the manifest; unsupported instructions
stop with a line number; label **spec-verified**.
**Known gap, documented not hidden:** hand-written PTX never exercises real PyTorch/Triton code paths. Triton-emitted
PTX is a possible later corpus source, only after confirming no NVIDIA SDK component produced the text (D1).
**Quantised-model sanity:** one tiny MXFP4/NVFP4 GEMM or layer validated against torchao/ml_dtypes.

**Gate 3: calibration data and outside-tester path (free).**
(a) Versioned dataset with provenance (source URL, model, quantisation, **stack and version**, build, kernel, depth,
metric). Candidate sources: llama.cpp #16578 (only about 30 of about 135 rows are Spark, builds mixed) [verified];
Spark Arena (<https://huggingface.co/sparkarena>; vLLM, SGLang, TRT-LLM, llama.cpp; fixed PP2048/TG128; results.csv
and recipe.yaml with runtime versions) [secondary]; LMSYS DGX Spark review, NVIDIA blog, JetsonHacks tables, the NVIDIA
forum same-harness thread, SiliconBench (arXiv 2609.19169) [secondary]. Estimated 15-25 distinct models and 100-300
points overall [my estimate]. **Count independent (model, quantisation, stack) cells; depth rows of one model are
correlated and count once. Never pool across software stacks.**
(b) A harness that runs on the CI emulator and whose JSON output validates against a committed schema; owners can
run it on a Spark and return the JSON.
(c) Apply for free hardware access (NVIDIA LaunchPad, partner programs); approval never assumed.
(d) Noise floor from public data alone: group rows by identical (model, quantisation, stack, depth); sigma_between =
median absolute deviation of ln(x) across submitters (build as a fixed effect where known); sigma_within from
llama-bench "±" values. Interval = [x e^(-1.96 sigma), x e^(+1.96 sigma)], x the pooled median.
Exit: dataset, schema and noise-floor method committed; harness passes on the emulator in CI; outcomes of (b) and
(c) recorded whatever they are. **No exit depends on any outside party responding.**

**Gate 4b: full performance model.**
**Amended by D3 (2026-11-18):** the unfitted-formula test is replaced by leave-one-model-out testing of a calibrated
model. Efficiency is fitted **per stack and per format** (NVFP4 and Ollama paths measured clearly below the rest),
with its spread carried into the prediction as an interval. Until the sample-size condition below is met every result
is labelled **research-only**. The exploratory result in section 15 (median 8% with one global factor, heavy tail
up to 136%) is a motivation, not evidence for this gate.
Terms [reviewer's list, my estimate, to be validated]:
- Decode bytes per token = active-expert weights at each tensor's **real** bits per weight (MXFP4 about 4.25, Q8_0
  about 8.5, Q4_K_M a Q4_K/Q6_K mix near 4.9; scales, embeddings and output tensors often higher) plus KV-cache
  read at depth d (`2 x layers x kv_heads x head_dim x ctx x bytes`, minus sliding-window layers, adjusted for
  GQA/MLA); an activated-expert coverage term for MoE at batch above 1.
- Decode time = bytes / (BW_eff x stack_eff) + per-layer launch overhead (matters for 4B-class models).
  **BW_eff is fitted, prior 0.75-0.95 of peak.**
- Prefill time = FLOPs / (peak_fmt x eff) + attention quadratic term + weight-read floor. Peaks are dense, never the
  sparse 1 PFLOP figure.
- Per-stack and per-format efficiency terms fitted separately; a batch/concurrency term.
Validation: leave-one-model-out; compare against Gate 4a's naive roofline.
Provisional targets [my call, revised from reviewer 3; revisit after the noise floor is measured]: decode median
absolute error <= 20% per stack and **at least 2x better than the naive roofline**; prefill <= 35%. "Error" is 0
inside the noise-adjusted interval, otherwise the relative distance to its nearest edge; raw error is also
reported.
**Sample-size condition:** the label **bounded-verified (public data)** requires >= 3 held-out models,
>= 1 held-out quantisation, and >= 20 held-out **independent** cells. Below that, results are **research-only**.
Reachable for models and quantisation; marginal for 20 cells. If it is not reached, that is the honest outcome.
Published-accuracy context (reported from search snippets, not read; the earlier "15% vs 30% over 42 models"
figure is **unconfirmed and replaced**): vendor-profiled learned models report single-digit to low-teens MdAPE on
many GPUs with one stack, while a naive FP16 roofline is reported at about 30% (server) and 80% (offline) in one
study (arXiv 2607.02391, reported). Our setting (one device, mixed stacks, few models) is harder.
**Biggest risk:** format/stack efficiency is not transferable. NVFP4/MXFP4 paths on SM121 are reportedly immature, so
a bandwidth roofline can predict a 4-bit model 2x faster than the software delivers. Gate 4a's format check exposes
this early.
**Proxy data** (RTX 5090 / PRO 6000, Strix Halo, Apple M-series): use only for compute- and format-side term shapes;
never for BW_eff, which differs by memory controller and backend.

**Cut from v1:** Gate 5 (Arm CPU half) and `spark-sass-lab`.

## 10. What is left to the builder

Crate boundaries, IR details, file formats, internal naming, exit-code numbering, CI details, fork-vs-fresh for the
PTX executor (after the spike).

## 11. Risks and unknowns

1. D1 (NVIDIA licence) could constrain any use of `nvcc` output; PTX-first limits exposure but does not remove it
   for Gate 0a.
2. The PyPI toolchain may lack what nvcc needs, or not accept SM121.
3. `docs.nvidia.com` unreachable (D2): Gate 0b cannot start without the PTX ISA text.
4. `mma.sync` accumulation order and internal precision are unspecified; some results can only be banded.
5. No hardware oracle; ceiling is spec-verified until an outside Spark owner reproduces results.
6. Memory system, clocks and thermals are only terms in the performance model; throttling reports conflict.
7. Public data is noisy and confounded (build, OS, kernel, quantisation, stack); few independent cells.
8. Gate 2 may collapse to numerics-only if the spike cost is high.
9. Gate 4b may not reach its sample-size condition; Gate 4a may show the model is not worth refining.
10. Hand-written PTX does not cover real Triton/PyTorch paths.

## 12. Owner decisions

- **D1, resolved 2026-11-18 by the owner: hand-written PTX only.** The owner does not hold the NVIDIA toolchain
  licence, so v1 uses **no NVIDIA toolchain**: no `nvcc`, `ptxas` or `cuobjdump`, and no NVIDIA-built output is
  executed, disassembled or committed. All kernels are written by hand in PTX from NVIDIA's public documentation.
  Consequences: **Gate 0a is cut**; `spark-sass-lab` stays cut; the PyPI wheels are not used. (Not legal advice; this
  records the owner's decision.)
  - New risk: with no `ptxas`, hand-written PTX **cannot be validated against NVIDIA's assembler**. "spec-verified"
    therefore means "matches the public documentation", not "real `ptxas` would accept it". Mitigations: every
    instruction use cites a PTX ISA clause (Gate 0b inventory); read CUTLASS/Triton source for PTX idioms (reading
    source, not running NVIDIA output); and Gate 3's outside-tester harness includes a **"does `ptxas` accept this
    PTX"** check that a Spark owner (who has the licence) runs and returns.
- **D2, resolved 2026-11-18 by the owner: hosts allowed.** The "Dgx" environment now allows `docs.nvidia.com`,
  `jetsonhacks.com`, `huggingface.co`, `forums.developer.nvidia.com`, `lmsys.org` and `arxiv.org`. Gate 0b and the
  Gate 3/4a data work are unblocked.

- **D3, resolved 2026-11-18 by the owner: Option 2, an open amendment.** The pre-registered Gate 4a rule fired
  (median error 55% against a 40% limit, section 15) and **stays on record as fired**. The owner chose to amend the
  rule: Gate 4b is kept, tested **leave-one-model-out** instead of by the unfitted formula, and built as a
  one-factor-per-stack/format calibrated roofline with explicit uncertainty, labelled **research-only** until its
  sample-size condition is met. This is an amendment, not a pass of the original rule.

## 13. Review log and my calls

**Round 1** (builder check; adversarial technical/licence): decoder integer/control-flow coverage was overstated;
`BSSY/BSYNC` are no-ops; encodings are SM86/SM61 only; `ir::Function` is graphics-only; `tensor.rs` beyond FP16 is a
rewrite; verification tools do not transfer; the 25% noise figure is confounded; "spec-verified" was too strong for
SASS; `sm_121a`/`sm_120a` needed; PyPI wheel contents unverified; CUDA EULA risk unaddressed; untestable exit
criteria; stale references; missing licences.

**Round 2** (performance/calibration; prior-art/scope): the decode model lacked real bits-per-weight, MoE coverage,
KV, per-stack terms, and treated 273 GB/s and 1 PFLOP as givens; the data sources were richer than assumed (Spark
Arena etc.) but need independent-cell counting and per-stack pooling; a concrete noise-floor method and an
"error against interval" definition were supplied; **ptoxide** (permissive Rust PTX VM) existed, undermining "new
compute IR from scratch"; ml_dtypes and gfloat give a two-oracle numeric check; CUTLASS is to be read, not run;
GenZ/llm-analysis make a roofline baseline nearly free; a memcheck checker and quantised-model validation were
missing; Gate 5, `spark-sass-lab` and the label taxonomy looked over-engineered; Gate 4's cheapest form should come
first; my earlier "15% vs 30%" citation did not match the reviewer's reading and was replaced.

**Unverified across all four reviews:** current EULA text, PTX shape tables, CUDA 13 SM121 notes, SM121 acceptance by the
wheels, the no-`tcgen05` claim beyond forums, and most performance-paper numbers (arXiv was unreachable).

**My calls:** working name `ounce-spark`; MIT; PTX-first; labels *spec-verified* and *inferred*;
convergence-barrier warp model; Rust with forbidden unsafe; the order (4a first, then 0a/0b/1/Gate 2 spike in
parallel); fork-vs-fresh decided by the spike; Gate 2 rescoped to numerics-only if one shape costs over 2 days;
Gate 4a stop/continue thresholds (40% / 25%); provisional Gate 4b targets (decode <=20% and 2x better than naive,
prefill <=35%) with the sample-size condition; Gate 5 and `spark-sass-lab` cut from v1; hashes and inventories only
committed (no SASS text, no binaries); unlicensed third-party SASS tools read-only; Triton interpreter as
cross-check only; nothing from `ounce-nx2`.

## 14. Gate 4a result, first pass (SUPERSEDED by section 15)

Files: `spikes/gate4a/gate4a.py`, `spikes/gate4a/output.txt`. Formula fixed before looking at data:
decode tok/s = 0.9 x 273 GB/s / bytes per token. No fitted parameters.

Data actually obtained: **8 readings, 4 independent (model, format, stack) cells**, against the 15-20 rows planned.
Only 3 readings are [verified] (llama.cpp #16578, one model); the rest are [secondary] search snippets, and several
inputs (active-parameter fractions, two model sizes) are [memory]. jetsonhacks.com is blocked in this container
and the llama.cpp thread exposes only three readable Spark rows.

| Result | Value |
|---|---|
| Median absolute error, zero-parameter formula | **51%** (mean 61%) |
| Direction | **All 8 readings over-predicted** (measured/predicted 0.46-0.78, median 0.66) |
| Per-cell medians (120B MoE MXFP4, 20B MoE MXFP4, 8B FP8, 70B FP8) | 0.62, 0.70, 0.67, 0.78 |
| Same data at BW = 177 / 200 / 262 GB/s | median measured/predicted 0.92 / 0.82 / 0.62 |

Pre-registered rule (section 9.3): median error above 40% means stop. Applied literally it fires. **I am
calling it inconclusive instead, as my judgement and open to overruling:** the rule's precondition (15-20 rows)
was not met, and the error is one-sided and consistent across a 15x range of model size and two formats, which a
single fitted efficiency factor would absorb. Fitting it on these same 8 points proves nothing; it must be
validated on held-out cells. The two-format check was **not run** (no same-model, two-format data found).


## 15. Gate 4a result, second pass (2026-11-18, throwaway spike)

Files: `spikes/gate4a/gate4a_v2.py`, `output_v2.txt`, `loo_exploratory.py`, `output_loo_exploratory.txt`,
`field_notes_results.jsonl` (copied from an MIT-licensed repo, attribution below). Same formula as before, fixed
in advance: decode tok/s = 0.9 x 273 GB/s / (active params x bits per weight / 8). Single-stream, batch 1,
speculative-decoding rows excluded.

New data after hosts were allowed: the field notes of one Spark operator
(<https://github.com/sergioamsilva/dgx-spark-field-notes>, MIT, (c) 2026 Sergio Silva; one machine, one operator,
same harness across Ollama and vLLM, versions pinned). [verified: read from the repo files; the measurements are the
author's.] Not obtained: Spark Arena results (the Hugging Face org page is a landing page with no datasets, results
are on `spark-arena.com`, a host not allowed here) and the JetsonHacks table (the URL returns 404).
Inputs marked [memory]: active-parameter counts and bits per weight (MXFP4 4.25, NVFP4 4.5, Q4_K_M 4.85); Gemma's
4B active comes from its "A4B" name. Excluded: qwen3-coder-next and GLM-4.7-Flash (active parameters unknown).

| Result | Value |
|---|---|
| Readings / independent (model, format, stack) cells | 15 / 11 (meets the planned 15-20 readings) |
| **Median absolute error, zero-parameter formula** | **55%** (mean 89%) |
| Direction | **All 15 over-predicted**; measured/predicted 0.28-0.78, median 0.64 |
| Per format (cell medians) | MXFP4 0.64, FP8 0.72, Q4_K_M 0.59, **NVFP4 0.44 (0.29-0.58)** |
| Per stack (cell medians, not pooled) | SGLang 0.72, llama.cpp 0.64, vLLM 0.58 (0.29-0.67), Ollama 0.49 |
| Two-format check (Qwen3-30B-A3B, vLLM, FP8 ~53 to NVFP4 77.1 tok/s) | measured speedup 1.45 vs 1.78 predicted from bytes: -18%, inside the 30% rule (the FP8 figure is approximate in the source) |

**Verdict against the pre-registered rule: it fires.** Median error is above 40% and the sample now meets the
precondition, so I no longer call it inconclusive. The unfitted formula does not predict Spark decode speed:
real decode reaches roughly 0.64 of it. This is decision D3 for the owner (section 12).

**Exploratory, post-hoc, run after seeing the results, not the pre-registered test:** leave-one-model-out with one
global efficiency factor (0.61-0.66) gives median error **8%**, 11 of 15 readings within 25%, but a heavy tail
(p75 32%, max 136%). The tail comes from Gemma-4 on vLLM NVFP4 (30.3-32.6 tok/s, 0.28-0.30) and Ollama (0.49), and
Ollama gpt-oss-120B (0.46), that is from format/stack/kernel efficiency, not from bandwidth. The forum author
attributes weak NVFP4 speed to missing kernels on SM12x (vLLM falls back, or uses FlashInfer CUTLASS kernels), so
the earlier-flagged risk (format and stack efficiency does not transfer) is **visible in the data**.

**Caveats:** one operator supplies 10 of 15 readings; same-hardware noise is not measured by them; ten readings
depend on [memory] inputs; vendor and review-site rows have unknown stacks; model cells are correlated, so leave-
one-model-out with only 6 models is a weak test.

## 16. Gate 0b early findings from the PTX ISA (2026-11-18)

Source: NVIDIA's public PTX ISA documentation, version 9.5, fetched with `curl` from `docs.nvidia.com` (the page
title reads "PTX ISA 9.5 documentation"). The HTML is **not committed** (NVIDIA documentation terms); only
citations and extracted facts go in the repo. Text was read after tag-stripping, so each item below needs a
visual check against the page in Gate 0b. [verified from the text unless stated]

- Targets `sm_120`/`sm_120a`/`sm_120f` and `sm_121`/`sm_121a`/`sm_121f` exist.
- **Maximum statically allocated shared memory is 100 KB on `sm_120a`/`sm_121a`**, versus 228 KB on
  `sm_90a`/`sm_100a`/`sm_110a`. `spark-mem` must enforce 100 KB, not a datacenter figure.
- The block-scaled `mma` qualifiers `.kind::mxf4nvf4` and `.kind::mxf4` are listed as supported on `sm_120a` and
  `sm_121a`. So block-scaled FP4 `mma` **is documented for the Spark's target**, closing an earlier open question.
- `.acc::f16` and the `e4m3/e5m2` `mma` forms are listed for `sm_120a`/`sm_121a`.
- **`cvt.rn.satfinite{.relu}.{e2m1x2/e2m3x2/e3m2x2/ue8m0x2}.f32` and `cvt.rn{.relu}.f16x2.{e2m1x2/e2m3x2/e3m2x2}`:**
  the architecture-specific list names `sm_100a`, `sm_101a` (renamed `sm_110a`) and `sm_120a`, and **does not name
  `sm_121a`**; the family-specific lists name `sm_120f` or higher in the same family (from PTX 8.8). A forum post
  says SM121 lacks `cvt.e2m1x2`, which removes the FP4 compute speedup but not the bandwidth saving. The
  documentation neither clearly confirms nor refutes this; **Gate 0b must resolve it from the page itself**,
  because it decides how FP4 conversion is modelled.
- `cvt` with `.s2f6x2` is listed for `sm_120a` and `sm_121a`.
