# TrustNet Browser Extension

The TrustNet browser extension provides a lightweight interface for analyzing factual claims selected directly from webpages.

It is a Chrome/Edge Manifest V3 extension that communicates with the TrustNet desktop application's local FastAPI backend.

Current version: **0.1.0**

---

## 1. Purpose

The extension allows a user to:

- select a factual claim on a webpage,
- open the TrustNet extension,
- automatically load the selected text,
- send the claim to the local TrustNet backend,
- view the resulting credibility assessment.

The extension does not perform credibility analysis itself.

Retrieval, evidence analysis, scoring, and uncertainty handling are performed by the TrustNet backend.

---

## 2. Architecture

```text
Webpage
   |
   v
Selected Claim
   |
   v
TrustNet Popup
   |
   v
Manifest V3 Service Worker
   |
   v
http://127.0.0.1:8000
   |
   v
TrustNet Backend
   |
   v
Credibility Result