# TrustNet Dataset Strategy

TrustNet does not train a large language model from scratch.

Datasets are primarily used for:
- evaluation
- benchmarking
- retrieval testing
- evidence-selection testing
- prompt development
- error analysis
- comparison against baselines

## 1. AVeriTeC — Primary Benchmark

Priority: Essential

Domain:
News, political, and real-world claims.

Role in TrustNet:
- Main end-to-end evaluation dataset
- Claim verification testing
- Evidence retrieval evaluation
- Evidence adequacy evaluation
- Comparison of TrustNet against baselines
- Error analysis

AVeriTeC is the primary benchmark for the project.

## 2. LIAR — Supporting Benchmark

Priority: Useful

Domain:
Political claims.

Role:
- Additional claim-level robustness testing
- Testing performance on short political statements
- Secondary verification experiments

Limitation:
LIAR is mainly useful for claim/veracity evaluation and does not provide the same evidence-oriented evaluation structure as AVeriTeC.

## 3. MultiFC — Supporting Multi-Domain Benchmark

Priority: Useful if time permits

Domain:
Real-world claims from multiple fact-checking sources.

Role:
- Cross-domain robustness testing
- Testing whether the framework generalizes beyond the primary AVeriTeC benchmark

It is not required for the minimum viable evaluation.

## 4. SciFact — Secondary Domain Benchmark

Priority: Optional but recommended

Domain:
Scientific claims.

Role:
- Test whether TrustNet generalizes from news/political claims to scientific claims
- Evidence retrieval testing
- Scientific claim verification
- Domain-transfer evaluation

Use only after the primary AVeriTeC pipeline is working.

## 5. Climate-FEVER — Additional Scientific/Climate Benchmark

Priority: Optional

Domain:
Climate claims.

Role:
- Additional domain-transfer testing
- Climate-specific evidence verification
- Robustness evaluation

Use only if project time permits.

# Priority Order

1. AVeriTeC
2. LIAR
3. SciFact
4. MultiFC
5. Climate-FEVER

# Minimum Required Evaluation

If development time becomes limited, TrustNet must at minimum complete:

- AVeriTeC evaluation
- Plain-LLM baseline
- Search + LLM baseline
- Full TrustNet evaluation
- Evidence retrieval evaluation
- Citation validity evaluation
- Error analysis

Additional datasets must not delay completion of the core TrustNet system.

# Important Principle

Dataset labels are benchmark ground truth.

They must not be supplied to TrustNet during normal evaluation.

The system receives the claim and permitted evidence/retrieval inputs, generates its prediction, and only afterward is its output compared with the dataset ground truth.