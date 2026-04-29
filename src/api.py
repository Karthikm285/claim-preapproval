from dotenv import load_dotenv
load_dotenv()  

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
import os, json, re, time

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import httpx

# Optional deps for Build C
try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None

app = FastAPI(title="Claims Triage API", version="0.4")

# ----------------------------
# Config / Versions
# ----------------------------
LOG_PATH = "logs/triage_events.jsonl"

MODEL_FRAUD_VERSION = "fraud_svm_v1"
MODEL_AUTO_VERSION  = "auto_svm_v1"
RAG_VERSION         = "guidelines_synth_tfidf_v1"
DATA_VERSION        = "claims_synth_v1"
API_VERSION         = "0.4"

# Guardrails
AMOUNT_BLOCK = 8000
REVIEW_RISK_BLOCK = 0.8

# Qdrant (host vs docker)
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")

# LLM (OpenAI)
OPENAI_API_KEY  = os.getenv("")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_TIMEOUT_S   = float(os.getenv("LLM_TIMEOUT_S", "15"))

# ----------------------------
# Request schema
# ----------------------------
class Claim(BaseModel):
    claim_id: str = "UNKNOWN"

    member_age: float | None = None
    member_gender: str = ""
    state: str = ""
    plan_type: str = ""
    member_tenure_months: float | None = None

    provider_specialty: str = ""
    place_of_service: str = ""

    inpatient_flag: bool = False
    out_of_network: bool = False
    prior_auth_required: bool = False

    prior_claims_90d: float | None = None
    ed_visits_180d: float | None = None
    chronic_index: float | None = None

    claim_amount: float | None = None
    allowed_amount: float | None = None

    fraud_signal_score: float | None = None
    review_risk_score: float | None = None

    clinical_notes: str = ""
    call_center_notes: str = ""

class ClaimRequest(BaseModel):
    enable_llm: bool = False
    claim: Claim

# ----------------------------
# Globals loaded at startup
# ----------------------------
fraud_model = None
auto_model = None
rag_vec = None
rag_doc_matrix = None
rag_docs = None
router_cutoffs = None

# Build C (optional)
qdrant = None
bm25 = None
rag_meta = None  # expected: {"collection": "...", "docs": [...]}

# ----------------------------
# Feature schema (must match training)
# ----------------------------
NUM_COLS = [
    "member_age",
    "prior_claims_90d",
    "ed_visits_180d",
    "chronic_index",
    "member_tenure_months",
    "claim_amount",
    "allowed_amount",
    "fraud_signal_score",
    "review_risk_score",
]

CAT_COLS = [
    "member_gender",
    "state",
    "plan_type",
    "provider_specialty",
    "place_of_service",
    "inpatient_flag",
    "out_of_network",
    "prior_auth_required",
]

TEXT_COL = "combined_notes"
ALL_EXPECTED = NUM_COLS + CAT_COLS + [TEXT_COL]

# ----------------------------
# Startup load
# ----------------------------
@app.on_event("startup")
def load_artifacts():
    """
    Load core artifacts (required).
    Load Build C artifacts (optional).
    """
    global fraud_model, auto_model, rag_vec, rag_doc_matrix, rag_docs, router_cutoffs
    global qdrant, bm25, rag_meta

    # Core artifacts (required)
    try:
        router_cutoffs = joblib.load("artifacts/router_cutoffs.joblib")
        fraud_model = joblib.load("artifacts/fraud_model.joblib")
        auto_model = joblib.load("artifacts/auto_model.joblib")
        rag_vec = joblib.load("artifacts/rag_vectorizer.joblib")
        rag_doc_matrix = joblib.load("artifacts/rag_doc_matrix.joblib")
        rag_docs = joblib.load("artifacts/rag_docs.joblib")
    except Exception as e:
        raise RuntimeError(f"Failed to load core artifacts: {e}")

    # Build C artifacts (optional)
    try:
        rag_meta = joblib.load("artifacts/rag_qdrant_meta.joblib")
    except Exception:
        rag_meta = None

    try:
        bm25_pack = joblib.load("artifacts/bm25_index.joblib")
        bm25 = bm25_pack.get("bm25")
        if rag_meta is None and "docs" in bm25_pack:
            rag_meta = {"collection": "", "docs": bm25_pack["docs"]}
    except Exception:
        bm25 = None

    try:
        if QdrantClient is not None:
            qdrant = QdrantClient(url=QDRANT_URL)
            _ = qdrant.get_collections()
        else:
            qdrant = None
    except Exception:
        qdrant = None

# ----------------------------
# Health
# ----------------------------
@app.get("/health")
def health():
    core_ok = all(x is not None for x in [fraud_model, auto_model, router_cutoffs, rag_vec, rag_doc_matrix, rag_docs])
    build_c_possible = bool(rag_meta and rag_meta.get("docs")) and (bm25 is not None or (qdrant is not None and rag_meta.get("collection")))
    return {
        "status": "ok" if core_ok else "not_ready",
        "core_ok": core_ok,
        "build_c_ok": build_c_possible,
        "qdrant_url": QDRANT_URL,
        "ts_utc": datetime.utcnow().isoformat() + "Z",
    }

# ----------------------------
# Helpers
# ----------------------------
def log_event(event: Dict[str, Any]) -> None:
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")

def blocked_for_auto(claim: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if bool(claim.get("out_of_network", False)):
        reasons.append("out_of_network")
    if bool(claim.get("inpatient_flag", False)):
        reasons.append("inpatient_flag")
    if bool(claim.get("prior_auth_required", False)):
        reasons.append("prior_auth_required")

    amt = float(claim.get("claim_amount", 0) or 0)
    if amt >= AMOUNT_BLOCK:
        reasons.append("high_claim_amount")

    rrs = float(claim.get("review_risk_score", 0) or 0)
    if rrs >= REVIEW_RISK_BLOCK:
        reasons.append("high_review_risk_score")

    return (len(reasons) > 0), reasons

def normalize_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(claim)
    clinical = str(claim.get("clinical_notes", "") or "")
    call = str(claim.get("call_center_notes", "") or "")
    out[TEXT_COL] = (clinical + " " + call).strip()

    for c in NUM_COLS:
        out.setdefault(c, np.nan)

    for c in CAT_COLS:
        if c in ["inpatient_flag", "out_of_network", "prior_auth_required"]:
            out.setdefault(c, False)
        else:
            out.setdefault(c, "")

    for c in ALL_EXPECTED:
        if c not in out:
            out[c] = np.nan

    return out

def build_query(claim: Dict[str, Any]) -> str:
    parts = [
        claim.get("provider_specialty", ""),
        claim.get("place_of_service", ""),
        claim.get("plan_type", ""),
        claim.get("state", ""),
        claim.get("clinical_notes", ""),
        claim.get("call_center_notes", ""),
    ]
    return " ".join([str(p) for p in parts if p is not None]).strip()

# ----------------------------
# Retrieval A: TF-IDF baseline
# ----------------------------
def retrieve_tfidf_diverse(query: str, top_k: int = 3, pool_k: int = 30):
    q = rag_vec.transform([query])
    sims = cosine_similarity(q, rag_doc_matrix).ravel()
    idx_sorted = sims.argsort()[::-1][:pool_k]

    raw = []
    for i in idx_sorted:
        d = rag_docs[i]
        raw.append((d["doc_id"], d["title"], d["text"], float(sims[i])))

    chosen, used_titles, used_categories = [], set(), set()
    for doc_id, title, text, score in raw:
        if title in used_titles:
            continue
        category = title.split()[0] if title else "misc"
        if category in used_categories and len(used_categories) < top_k:
            continue
        used_titles.add(title)
        used_categories.add(category)
        chosen.append((doc_id, title, text, score))
        if len(chosen) == top_k:
            break

    if len(chosen) < top_k:
        for doc_id, title, text, score in raw:
            if title in used_titles:
                continue
            used_titles.add(title)
            chosen.append((doc_id, title, text, score))
            if len(chosen) == top_k:
                break

    return chosen

# ----------------------------
# Retrieval C: Hybrid (Qdrant +/or BM25) with RRF
# ----------------------------
def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())

def retrieve_qdrant(query: str, pool_k: int = 30):
    if qdrant is None or rag_meta is None or rag_vec is None or not rag_meta.get("collection"):
        return []
    v = rag_vec.transform([query]).toarray().astype(np.float32)[0].tolist()
    res = qdrant.query_points(
        collection_name=rag_meta["collection"],
        query=v,
        limit=pool_k,
        with_payload=True,
    )
    out = []
    for p in res.points:
        payload = p.payload or {}
        out.append((payload.get("doc_id"), payload.get("title"), payload.get("text", ""), float(p.score), "qdrant"))
    return out

def retrieve_bm25(query: str, pool_k: int = 30):
    if bm25 is None or rag_meta is None or "docs" not in rag_meta:
        return []
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    idx = np.argsort(scores)[::-1][:pool_k]
    docs = rag_meta["docs"]
    out = []
    for i in idx:
        d = docs[int(i)]
        out.append((d.get("doc_id"), d.get("title"), d.get("text", ""), float(scores[int(i)]), "bm25"))
    return out

def fuse_rrf(results: List[Tuple[str, str, str, float, str]], k: int = 60) -> List[Tuple[str, str, str, float]]:
    fusion: Dict[str, Dict[str, Any]] = {}
    by_source: Dict[str, List[Tuple[str, str, str]]] = {}
    for doc_id, title, text, score, src in results:
        if not doc_id:
            continue
        by_source.setdefault(src, []).append((doc_id, title, text))

    for src, lst in by_source.items():
        for rank, (doc_id, title, text) in enumerate(lst, start=1):
            add = 1.0 / (k + rank)
            if doc_id not in fusion:
                fusion[doc_id] = {"doc_id": doc_id, "title": title, "text": text, "score": 0.0}
            fusion[doc_id]["score"] += add

    fused = list(fusion.values())
    fused.sort(key=lambda x: x["score"], reverse=True)
    return [(x["doc_id"], x["title"], x["text"], float(x["score"])) for x in fused]

def diversify(docs: List[Tuple[str, str, str, float]], top_k: int = 3) -> List[Tuple[str, str, str, float]]:
    chosen, used_titles = [], set()
    for doc_id, title, text, score in docs:
        t = title or ""
        if t in used_titles:
            continue
        used_titles.add(t)
        chosen.append((doc_id, title, text, score))
        if len(chosen) == top_k:
            break
    if len(chosen) < top_k:
        for doc_id, title, text, score in docs:
            if (doc_id, title) in [(c[0], c[1]) for c in chosen]:
                continue
            chosen.append((doc_id, title, text, score))
            if len(chosen) == top_k:
                break
    return chosen

def retrieve_hybrid(query: str, top_k: int = 3, pool_k: int = 30):
    q_hits = retrieve_qdrant(query, pool_k=pool_k)
    b_hits = retrieve_bm25(query, pool_k=pool_k)

    # If only one retriever is available, return it directly
    if q_hits and not b_hits:
        fused = [(d, t, x, float(s),) for (d, t, x, s, _src) in q_hits]
        fused.sort(key=lambda r: r[3], reverse=True)
        return diversify(fused, top_k=top_k)

    if b_hits and not q_hits:
        fused = [(d, t, x, float(s),) for (d, t, x, s, _src) in b_hits]
        fused.sort(key=lambda r: r[3], reverse=True)
        return diversify(fused, top_k=top_k)

    if not q_hits and not b_hits:
        return []

    fused = fuse_rrf(q_hits + b_hits, k=60)
    return diversify(fused, top_k=top_k)

# ----------------------------
# Routing + LLM
# ----------------------------
def decide_route(fraud_score: float, auto_score: float, blocked_auto: bool) -> str:
    fc = float(router_cutoffs["fraud_cutoff"])
    ac = float(router_cutoffs["auto_cutoff"])
    if fraud_score >= fc:
        return "SIU_REVIEW"
    if (auto_score >= ac) and (not blocked_auto):
        return "AUTO_APPROVE"
    return "CLINICIAN_REVIEW"

def llm_enabled() -> bool:
    return bool(OPENAI_API_KEY)

def call_llm_summary(
    claim: Dict[str, Any],
    route: str,
    fraud_score: float,
    auto_score: float,
    blocked: bool,
    reasons: List[str],
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not llm_enabled():
        return {"enabled": False, "provider": "openai", "model": OPENAI_MODEL, "summary": ""}

    ev_lines = [
        f"- {e.get('doc_id','')} | {e.get('title','')} | sim={float(e.get('similarity',0.0)):.3f} | {e.get('snippet','')}"
        for e in evidence
    ]

    prompt = f"""
You are a claims triage assistant. Write a 1-2 sentence justification for the routing decision.
Rules:
- Only use the provided claim facts + evidence snippets.
- Do NOT invent codes, diagnoses, or missing fields.
- Mention guardrail blocks if any.
- Be concise.

Claim facts:
state={claim.get('state','')}, plan={claim.get('plan_type','')}, specialty={claim.get('provider_specialty','')}, pos={claim.get('place_of_service','')}
claim_amount={claim.get('claim_amount', None)}, out_of_network={claim.get('out_of_network', False)}, inpatient={claim.get('inpatient_flag', False)}, prior_auth_required={claim.get('prior_auth_required', False)}
fraud_score={fraud_score:.3f}, auto_score={auto_score:.3f}, route={route}
blocked_for_auto={blocked}, reasons={reasons}

Evidence:
{chr(10).join(ev_lines)}
""".strip()

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a careful, compliance-minded assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 120,
    }

    # Retry ONLY for transient 5xx/503 etc. If quota exhausted (429 quota), return immediately.
    max_attempts = 3
    base_sleep = 1.0

    try:
        with httpx.Client(timeout=LLM_TIMEOUT_S) as client:
            last_err = ""
            for attempt in range(1, max_attempts + 1):
                resp = client.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload)

                if resp.status_code == 200:
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    return {"enabled": True, "provider": "openai", "model": OPENAI_MODEL, "summary": text}

                # 429: quota/billing -> do NOT retry (it won't help)
                if resp.status_code == 429:
                    body = resp.text[:400]
                    return {
                        "enabled": True,
                        "provider": "openai",
                        "model": OPENAI_MODEL,
                        "error": f"HTTP 429 (quota/billing). Fix billing/limits, then retry. Body: {body}",
                        "summary": "",
                    }

                # Retry transient server errors
                if resp.status_code in (500, 502, 503, 504):
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    if attempt == max_attempts:
                        break
                    sleep_s = min(10.0, base_sleep * (2 ** (attempt - 1))) + (0.1 * attempt)
                    time.sleep(sleep_s)
                    continue

                # Other non-retryable errors
                return {
                    "enabled": True,
                    "provider": "openai",
                    "model": OPENAI_MODEL,
                    "error": f"HTTP {resp.status_code}: {resp.text[:400]}",
                    "summary": "",
                }

        return {
            "enabled": True,
            "provider": "openai",
            "model": OPENAI_MODEL,
            "error": f"OpenAI request failed after retries. Last error: {last_err}",
            "summary": "",
        }

    except Exception as e:
        msg = str(e).replace(OPENAI_API_KEY, "***REDACTED***") if OPENAI_API_KEY else str(e)
        return {"enabled": True, "provider": "openai", "model": OPENAI_MODEL, "error": msg, "summary": ""}

# ----------------------------
# Endpoint
# ----------------------------
@app.post("/triage")
def triage(payload: ClaimRequest):
    required = [fraud_model, auto_model, rag_vec, rag_doc_matrix, rag_docs, router_cutoffs]
    if any(x is None for x in required):
        raise HTTPException(status_code=500, detail="Artifacts not loaded. Check /health.")

    claim = payload.claim.model_dump()
    claim_id = payload.claim.claim_id

    norm = normalize_claim(claim)
    X = pd.DataFrame([norm])[ALL_EXPECTED]

    try:
        fraud_score = float(fraud_model.decision_function(X)[0])
        auto_score = float(auto_model.decision_function(X)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model input error: {e}")

    blocked, reasons = blocked_for_auto(claim)
    route = decide_route(fraud_score, auto_score, blocked)

    query = build_query(claim)

    # Retrieval: Hybrid first; fallback to TF-IDF
    evidence_rows = retrieve_hybrid(query, top_k=3, pool_k=30)
    if not evidence_rows:
        evidence_rows = retrieve_tfidf_diverse(query, top_k=3, pool_k=30)

    evidence_pack = [
        {"doc_id": d[0], "title": d[1], "snippet": (d[2] or "")[:220] + "...", "similarity": float(d[3])}
        for d in evidence_rows
    ]

    audit = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "api_version": API_VERSION,
        "data_version": DATA_VERSION,
        "model_versions": {"fraud": MODEL_FRAUD_VERSION, "auto": MODEL_AUTO_VERSION},
        "rag_version": RAG_VERSION,
        "model_fraud": "LinearSVC(decision_function)",
        "model_auto": "LinearSVC(decision_function)",
        "routing_policy": {
            "fraud_cutoff": float(router_cutoffs["fraud_cutoff"]),
            "auto_cutoff": float(router_cutoffs["auto_cutoff"]),
            "SIU_PCT": float(router_cutoffs.get("SIU_PCT", 0.10)),
            "AUTO_PCT": float(router_cutoffs.get("AUTO_PCT", 0.05)),
        },
        "notes": "Quantile routing cutoffs loaded from artifacts/router_cutoffs.joblib",
    }

    result = {
        "claim_id": claim_id,
        "route": route,
        "risk": {"fraud_score": fraud_score, "auto_score": auto_score},
        "guardrails": {"blocked_for_auto": blocked, "reasons": reasons},
        "decision": {
            "recommended_action": (
                "ESCALATE_TO_SIU" if route == "SIU_REVIEW"
                else "AUTO_APPROVE" if route == "AUTO_APPROVE"
                else "SEND_TO_CLINICIAN"
            ),
            "reason": "Two-model router + safety guardrails + evidence pack",
        },
        "evidence_pack": evidence_pack,
        "audit": audit,
    }

    # LLM only if requested
    if payload.enable_llm:
        llm = call_llm_summary(
            claim=claim,
            route=route,
            fraud_score=fraud_score,
            auto_score=auto_score,
            blocked=blocked,
            reasons=reasons,
            evidence=evidence_pack,
        )
        result["llm"] = llm
        if llm.get("enabled") and llm.get("summary"):
            audit["llm_provider"] = llm.get("provider")
            audit["llm_model"] = llm.get("model")
    else:
        result["llm"] = {"enabled": False, "provider": "openai", "model": OPENAI_MODEL, "summary": ""}

    log_event({
        "timestamp_utc": audit["timestamp_utc"],
        "claim_id": claim_id,
        "route": route,
        "risk": result["risk"],
        "guardrails": result["guardrails"],
        "evidence_doc_ids": [e["doc_id"] for e in evidence_pack],
        "api_version": audit["api_version"],
        "data_version": audit["data_version"],
        "model_versions": audit["model_versions"],
        "rag_version": audit["rag_version"],
        "routing_policy": audit["routing_policy"],
        "llm": result["llm"],
    })

    return result