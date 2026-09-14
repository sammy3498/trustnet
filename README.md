# TrustNet

**TrustNet** is an evidence-first credibility analysis system that evaluates factual claims using web evidence, retrieval-augmented generation, source assessment, structured LLM reasoning, deterministic scoring, and uncertainty handling.

The project provides:

- a FastAPI backend,
- hybrid evidence retrieval and reranking,
- structured LLM-based evidence analysis,
- a deterministic five-factor credibility framework,
- a Windows desktop application,
- and a Manifest V3 browser extension for analyzing selected webpage text.

Current release: **v0.1.0**

---

## 1. Project Objective

TrustNet is designed to assess the credibility of factual claims without relying on human reviewers in the operational verification pipeline.

Human evaluation may be used for research benchmarking of completed outputs, but humans do not perform or assist individual credibility checks during normal system operation.

The system follows an evidence-first principle:

> Strong evidence is primary. Secondary credibility factors must not override clear supporting or contradicting evidence.

---

## 2. Credibility Framework

TrustNet uses five weighted credibility factors:

| Factor | Weight |
|---|---:|
| Evidence Strength | 40% |
| Factual Consistency | 35% |
| Source Provenance | 10% |
| Bias & Framing | 8% |
| Incentive / Conflict | 7% |

The overall credibility score is calculated deterministically from these factors.

Operational score labels are:

| Score | Label |
|---|---|
| 80-100 | Highly Credible |
| 60-79 | Moderately Credible |
| 40-59 | Low Credibility |
| 0-39 | Not Credible |

An uncertainty gate can override the normal score label when the available evidence is insufficient or unreliable.

---

## 3. System Architecture

The primary TrustNet pipeline is:

```text
User Claim
    |
    v
Claim Preparation
    |
    v
Search Query Generation
    |
    v
Web Evidence Retrieval
    |
    v
Hybrid Ranking / Deduplication
    |
    v
Evidence Quality + Provenance Processing
    |
    v
LLM Evidence Analysis
    |
    v
Five-Factor Deterministic Scoring
    |
    v
Uncertainty Gate
    |
    v
Structured Credibility Report
