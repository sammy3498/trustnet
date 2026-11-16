"""Evidence-seeking searches, not a table of pre-decided claim verdicts.

A claim about two measurements needs searches for both measurements.  Rules in
this module choose *where to look*, never whether the claim is true.
"""
from __future__ import annotations

import re

# Broad source categories can be extended without changing the verdict logic.
FACET_HINTS = (
    (r"\b(population|census|demograph|birth rate|mortality)\b", "population statistics census official data"),
    (r"\b(happiness|life satisfaction|wellbeing|well-being)\b", "world happiness report life evaluation country rankings"),
    (r"\b(vot(e|es|ed|ing|er|ers)|election|ballot|primar(y|ies)|caucus)\b", "voter eligibility election law state primary authoritative guidance"),
    (r"\b(tax|deduction|gst|tariff|income.tax)\b", "tax authority government notification law effective date"),
    (r"\b(inflation|gdp|unemployment|interest rate)\b", "official statistical release central bank data"),
    (r"\b(vaccine|medication|treatment|clinical|disease|drug)\b", "medical guideline systematic review original study"),
    (r"\b(climate|emissions|temperature|warming|co2)\b", "original climate observations scientific assessment"),
    (r"\b(planet|orbit|asteroid|nasa|space|moon|sun)\b", "space agency astronomical observations scientific explanation"),
    (r"\b(law|court|constitution|amendment|legal|illegal|regulation)\b", "official statute court ruling jurisdiction date"),
)

STOP_NAMES = {"The", "A", "An", "No", "None", "Every", "All", "Any", "If", "In", "On", "At", "But", "And", "As", "Has", "Have", "Do", "Does", "Can", "Is", "Are"}


def _entities(claim: str) -> str:
    # Surface entity names verbatim; do not hallucinate missing countries/people.
    names = re.findall(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,})){0,2}\b", claim)
    unique = []
    months = {"January","February","March","April","May","June","July","August","September","October","November","December"}
    for name in names:
        parts = [word for word in name.split() if word not in STOP_NAMES and word not in months]
        name = " ".join(parts)
        if not name or name in unique:
            continue
        unique.append(name)
    # Accommodate the common dotted abbreviation U.S.
    if re.search(r"\bU\.S\.\b|U\.S\.", claim) and "US" not in unique:
        unique.append("US")
    return " ".join(unique[:5])


def atomic_search_facets(claim: str) -> list[str]:
    """Find overt conjunctive clauses without asserting their truth.

    This is deliberately syntactic: evidence, not a keyword rule, decides
    whether the entire claim is supported.  Keep only clauses that are
    independently assessable; don't split noun phrases such as 'red and
    processed meat'.
    """
    claim = " ".join(claim.split()).strip().rstrip(" .")
    if not claim:
        return []
    pieces = re.split(r"[,;]?\s+\b(?:and|but|whereas|while)\b\s+", claim, flags=re.I)
    if len(pieces) == 1:
        return [claim]
    clauses = []
    for piece in pieces:
        piece = piece.strip(" .,;")
        # Require a plausible subject/predicate, not merely a coordinated
        # phrase: 'red and processed meat' should remain an atomic entity.
        words = re.findall(r"[a-zA-Z]+", piece)
        verb = re.search(r"\b(is|are|was|were|has|have|had|can|cannot|do|does|did|takes|take|rotates|orbit|orbits|increased|decreased|ranks|ranked|classifies|classified|reduces|reduce|proved|stated|said|announced|causes|cause|shows|show|boils|weighs|weigh|costs|cost|allows|allow|bans|ban)\b",piece,re.I)
        if len(words) >= 4 and verb:
            clauses.append(piece)
    return clauses if len(clauses) >= 2 else [claim]


def _quote_search(claim: str, entities: str, years: list[str]) -> list[str]:
    """Seek the original documented finding for attributed research claims."""
    date = " ".join(years[:2])
    # Limit topical wording so search engines don't simply repeat the claim's
    # potentially false conclusion verbatim.
    content = re.sub(r"\b(conclusively|never|always|proved|proves|all|no one|none|every)\b", "", claim, flags=re.I)
    terms = [t for t in re.findall(r"[a-zA-Z]{4,}", content) if t.lower() not in
             {"that","were","have","from","there","their","these","under","because","review","study","analysis","stated","statement","announced"}]
    topic = " ".join(dict.fromkeys(x.lower() for x in terms))[:100]
    if re.search(r"\b(review|study|research|analysis|journal|report)\b", claim,re.I):
        return [f"{entities} {date} {topic} original review conclusion limitations authors correction".strip()]
    if re.search(r"\b(stated|said|announced|declared|quoted|reported|warned)\b", claim,re.I):
        return [f"{entities} {date} {topic} original statement press release transcript".strip()]
    return []


def plan_verification_queries(claim: str, max_queries: int = 6) -> list[str]:
    """Search literal claim, distinct conjuncts, and original attributed sources.

    Preserve at least one query for each overt independently verifiable facet
    before using searches on the whole statement.  No claim-specific answers
    or trusted domains are hardcoded.
    """
    claim = " ".join(claim.split()).strip()
    if not claim or max_queries <= 0:
        return []
    entities = _entities(claim)
    years = list(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b",claim)))
    date = " ".join(years[:2])
    clauses = atomic_search_facets(claim)
    queries = []
    if len(clauses) > 1:
        for clause in clauses[:min(3,max_queries)]:
            matching = [hint for regex,hint in FACET_HINTS if re.search(regex,clause,re.I)]
            queries.append(f"{clause} {matching[0] if matching else 'official original evidence'}")
    # For a compound comparison a single conjunct can contain two different
    # measurable dimensions; keep topical facets after clause reservations.
    topics = [hint for regex,hint in FACET_HINTS if re.search(regex,claim,re.I)]
    if len(clauses) == 1:
        for topic in topics[:max_queries-1]:
            queries.append(f"{entities} {date} {topic}".strip())
    queries.extend(_quote_search(claim,entities,years))
    if re.search(r"\b(no one|nobody|never|none|every|all|always|any|only|impossible)\b",claim,re.I):
        stem = re.sub(r"\b(no one|nobody|never|none|every|all|always|any|only|impossible)\b", "",claim,flags=re.I)
        queries.append(f"{stem.strip(' .')} exceptions counterexamples original evidence")
    if re.search(r"\b(said|stated|claimed|quoted|announced|tweeted|supported|promised)\b",claim,re.I):
        queries.append(f"{entities} {date} original statement transcript context".strip())
    queries.append(claim)
    queries.append(f"{claim} official source fact check")
    unique,seen=[],set()
    for q in queries:
        q=" ".join(q.split())[:260]
        if q and q.casefold() not in seen:
            unique.append(q)
            seen.add(q.casefold())
    if len(unique)<=max_queries:
        return unique
    # Preserve the literal statement even when facet queries fill the budget.
    head=unique[:max_queries-1]
    if claim not in head:
        head.append(claim)
    else:
        head.append(unique[max_queries-1])
    return head
