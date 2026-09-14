# TrustNet Evaluation Protocol

## 1. Purpose

This document defines how TrustNet is evaluated as an evidence-first credibility-analysis system.

The evaluation is designed to measure:

- final verification performance,
- retrieval and evidence quality,
- uncertainty behavior,
- failure modes,
- latency,
- comparison against simpler baselines.

This document distinguishes between:

- **Completed evaluation** — experiments actually executed and preserved in the repository.
- **Development evaluation** — experiments used during system development and diagnosis.
- **Planned evaluation** — desirable measurements that were specified but not necessarily completed during the final project cycle.

Results must never be reported as completed unless corresponding experiment artifacts exist.

---

## 2. Primary Benchmark

TrustNet uses **AVeriTeC** as its primary fact-verification benchmark.

The benchmark ground-truth label must not be supplied to TrustNet during inference.

The evaluation sequence is:

```text
Benchmark claim
    |
    v
TrustNet independently analyzes claim
    |
    v
TrustNet prediction produced
    |
    v
Prediction mapped to benchmark label
    |
    v
Compare with AVeriTeC ground truth