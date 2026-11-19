# ounce-spark

A free, evidence-first stand-in for the NVIDIA DGX Spark: PTX-level functional emulation of GPU kernels
on an ordinary CPU, spec-first numeric oracles, and a calibrated performance model. It is **not** a
cycle-accurate emulator and supplies no GPU power. Every claim carries a maturity label
(see `docs/LIMITS.md`); without Blackwell hardware, GPU claims cap at **spec-verified**.

Design: `docs/superpowers/specs/2026-11-18-ounce-spark-design.md`.
Plan: `docs/superpowers/plans/`. Progress: `PLAN.md`.

No NVIDIA toolchain is used: kernels are hand-written PTX from NVIDIA's public documentation.
