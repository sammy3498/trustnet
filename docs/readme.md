# TrustNet Documentation

This directory contains the methodological and research documentation for TrustNet.

TrustNet is an evidence-first credibility analysis system that combines evidence retrieval, structured LLM reasoning, source assessment, deterministic multi-factor scoring, and uncertainty handling.

---

## Documentation Index

### `dataset_strategy.md`

Describes the benchmark and dataset strategy used for TrustNet development and evaluation.

It covers the role of datasets such as AVeriTeC and the separation between development data and final evaluation data.

---

### `evaluation_protocol.md`

Defines the TrustNet evaluation methodology.

It covers:

- verification metrics,
- benchmark protocol,
- retrieval evaluation,
- evidence quality,
- citation validity,
- confidence and uncertainty,
- ablations,
- error analysis,
- latency,
- qualitative evaluation.

The document contains both:

1. evaluation procedures that were actually completed during the project, and
2. planned or recommended measurements that were not necessarily completed in the final project cycle.

Final reports must distinguish measured results from proposed evaluation work.

---

### `source_hierarchy.md`

Defines the source-quality and provenance hierarchy used by TrustNet when assessing retrieved evidence.

It helps guide:

- source reliability,
- provenance scoring,
- evidence quality assessment,
- treatment of primary and secondary sources.

---

## Final Evaluation Record

The primary final TrustNet v2 benchmark was performed using AVeriTeC.

Final validation:

```text
N = 80
Accuracy = 0.4000
Macro-F1 = 0.3873