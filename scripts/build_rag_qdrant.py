import os, re, json, joblib
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import networkx as nx
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)

COLLECTION = "guidelines_v1"
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())

def build_kg(docs: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Small, practical KG:
      policy_type -> doc_id
      specialty -> doc_id
    Derived from titles like: "Rx policy for Onc", "Imaging policy for MSK"
    """
    G = nx.DiGraph()
    for d in docs:
        doc_id = d["doc_id"]
        title = d.get("title","")
        text  = d.get("text","")

        # policy_type: first token (Rx/Imaging/Labs/etc.)
        policy_type = title.split()[0] if title else "misc"
        G.add_node(doc_id, kind="doc", title=title)
        G.add_node(policy_type, kind="policy_type")
        G.add_edge(policy_type, doc_id, rel="HAS_DOC")

        # specialty-ish: after "for X"
        m = re.search(r"\bfor\s+([A-Za-z0-9]+)", title)
        if m:
            spec = m.group(1)
            G.add_node(spec, kind="specialty")
            G.add_edge(spec, doc_id, rel="HAS_DOC")

        # simple entity hooks from text (optional)
        if "prior auth" in (text or "").lower():
            G.add_node("prior_auth", kind="concept")
            G.add_edge("prior_auth", doc_id, rel="MENTIONS")

    return G

def main():
    # You already have rag_docs.joblib from earlier (synthetic guideline corpus)
    docs = joblib.load("artifacts/rag_docs.joblib")  # list[{doc_id,title,text}]
    texts = [d["title"] + " " + d["text"] for d in docs]

    # 1) Embeddings
    model = SentenceTransformer(EMBED_MODEL_NAME)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)

    # 2) Qdrant upsert
    client = QdrantClient(url="http://localhost:6333")
    dim = emb.shape[1]

    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )

    points = []
    for i, d in enumerate(docs):
        payload = {
            "doc_id": d["doc_id"],
            "title": d.get("title",""),
            "text": d.get("text",""),
        }
        points.append(qm.PointStruct(id=i, vector=emb[i].tolist(), payload=payload))

    client.upsert(collection_name=COLLECTION, points=points)

    # 3) BM25 index
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    joblib.dump({"bm25": bm25, "tokenized": tokenized}, ART_DIR / "bm25_index.joblib")

    # 4) Knowledge Graph
    G = build_kg(docs)
    joblib.dump(G, ART_DIR / "kg.joblib")

    # 5) Save embed config
    joblib.dump(
        {"collection": COLLECTION, "embed_model": EMBED_MODEL_NAME, "dim": dim},
        ART_DIR / "rag_qdrant_meta.joblib"
    )

    print("✅ Built: Qdrant collection, BM25 index, KG, meta")

if __name__ == "__main__":
    main()