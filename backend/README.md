# TrustNet Backend

The TrustNet backend is the core credibility-analysis service.

It is implemented with FastAPI and provides the pipeline used by both the Windows desktop application and the browser extension.

---

## 1. Responsibilities

The backend handles:

- claim preparation,
- search-query generation,
- web evidence retrieval,
- lexical and semantic ranking,
- reranking,
- evidence deduplication,
- source/provenance processing,
- structured LLM evidence analysis,
- deterministic five-factor credibility scoring,
- uncertainty handling,
- API response generation.

The browser extension and desktop frontend do not implement credibility scoring themselves. They communicate with this backend.

---

## 2. Main Pipeline

```text
Input Claim
    |
    v
Claim Preparation
    |
    v
Search Query Generation
    |
    v
Evidence Retrieval
    |
    v
BM25 + Semantic Ranking
    |
    v
Reranking / Deduplication
    |
    v
Evidence Quality Processing
    |
    v
LLM Evidence Analysis
    |
    v
Five-Factor Scoring
    |
    v
Uncertainty Gate
    |
    v
Structured API Response