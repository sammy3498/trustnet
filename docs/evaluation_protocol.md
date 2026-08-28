# TrustNet Evaluation Protocol

## 1. Primary Objective

Evaluate whether TrustNet provides accurate, evidence-grounded,
transparent credibility assessments and whether it improves over
simpler LLM-based approaches.

## 2. Primary Benchmark

AVeriTeC is the primary benchmark.

The dataset ground-truth labels must never be provided to the system
during evaluation.

TrustNet first generates its prediction independently. The prediction
is then compared with the benchmark ground truth.

## 3. Systems to Compare

### Baseline A — Plain LLM

Claim
→ LLM
→ Prediction

No external evidence retrieval.

### Baseline B — Search + LLM

Claim
→ Web Search
→ Retrieved Evidence
→ LLM
→ Prediction

No TrustNet multi-factor framework.

### System C — Full TrustNet

Claim
→ Claim Processing
→ Hybrid Retrieval
→ Reranking
→ Evidence Validation
→ Evidence Analysis
→ Five-Factor Scoring
→ Uncertainty Gate
→ Final Report

## 4. AVeriTeC Label Mapping

TrustNet produces continuous credibility scores while AVeriTeC uses
categorical verification labels.

For benchmark comparison, convert TrustNet output into an evaluation
label.

Initial mapping:

TrustNet 60–100
→ Supported

TrustNet 0–39
→ Refuted

TrustNet 40–59
→ Not Enough Evidence

TrustNet uncertainty gate triggered
→ Not Enough Evidence

If the retrieved high-quality evidence contains substantial unresolved
supporting and contradicting evidence:
→ Conflicting Evidence

This mapping is an initial experimental mapping and may be calibrated
using development data.

Do not tune thresholds using the final test set.

## 5. Verification Metrics

Measure:

- Accuracy
- Macro F1
- Per-class precision
- Per-class recall
- Per-class F1
- Confusion matrix

Macro F1 is important because benchmark classes may be imbalanced.

## 6. Retrieval Metrics

Where gold evidence is available, evaluate retrieval separately from
final prediction.

Measure:

- Recall@5
- Recall@10
- evidence coverage
- relevant evidence retrieved

This helps distinguish retrieval failures from reasoning failures.

## 7. Evidence Quality Evaluation

Measure:

- evidence relevance
- evidence adequacy
- source quality
- evidence independence
- whether evidence actually supports the generated reasoning

## 8. Citation Validity

For every citation returned by TrustNet, check:

- URL is syntactically valid
- URL is reachable
- page corresponds to the cited source
- cited evidence is actually present or supported by the page

Report citation validity rate.

## 9. Confidence Evaluation

Compare TrustNet confidence against correctness.

The system should ideally:

- have high confidence when correct and strongly evidenced
- lower confidence when evidence is weak
- trigger uncertainty when evidence is insufficient

Measure confidence calibration where practical.

## 10. Factor Consistency

Check whether the five factor scores behave consistently with their
definitions.

Special attention should be given to:

- Evidence Strength
- Factual Consistency

because they account for 75% of the final score.

## 11. Ablation Experiments

Compare Full TrustNet against reduced versions.

Examples:

### No Reranker
Hybrid Retrieval
→ directly use retrieved evidence

### No Provenance Analysis
Remove source provenance processing.

### No Uncertainty Gate
Always force a credibility verdict.

### No Multi-Factor Framework
Use retrieved evidence with a simple LLM verdict.

These experiments identify which TrustNet components actually improve
performance.

## 12. Error Analysis

Categorize failures into:

- claim extraction failure
- query-generation failure
- retrieval failure
- reranking failure
- irrelevant evidence
- source-quality/provenance failure
- reasoning failure
- incorrect stance classification
- scoring failure
- uncertainty failure
- citation failure

Record representative examples from each category.

## 13. Performance Metrics

Measure:

- median latency (p50)
- p95 latency
- number of retrieved documents
- number of LLM calls
- token usage
- estimated cost per request

Normal Mode target:
approximately 5–10 seconds where practical.

Deep Mode may take substantially longer.

## 14. Qualitative Evaluation

Human participants may be used only to evaluate the quality of
TrustNet's completed outputs during research benchmarking.

They do NOT participate in the operational credibility verification
pipeline.

Possible evaluation criteria:

- reasoning clarity
- usefulness
- transparency
- evidence adequacy
- actionability

Use a 1–5 rating scale.

Blind preference testing may compare TrustNet reports against baseline
reports without telling evaluators which system generated each result.

## 15. Evaluation Order

1. Test retrieval independently.
2. Test evidence analysis.
3. Test final verification performance.
4. Compare against baselines.
5. Run ablations.
6. Perform error analysis.
7. Measure latency and cost.
8. Conduct final qualitative evaluation.

## 16. Minimum Successful Evaluation

The project must at minimum produce:

- AVeriTeC benchmark results
- Plain LLM baseline results
- Search + LLM baseline results
- Full TrustNet results
- Accuracy and Macro F1
- Retrieval Recall@k
- Citation validity
- Latency measurements
- Error analysis

Optional experiments must not delay the core evaluation.