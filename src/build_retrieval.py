import joblib, re
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
import numpy as np

# We reuse your existing TF-IDF vectorizer + docs (already in artifacts/)
VEC_PATH = "artifacts/rag_vectorizer.joblib"
DOCS_PATH = "artifacts/rag_docs.joblib"

OUT_META = "artifacts/rag_qdrant_meta.joblib"
OUT_BM25 = "artifacts/bm25_index.joblib"

COLLECTION = "guidelines_v1"

def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", (text or "").lower())

def main():
    vec = joblib.load(VEC_PATH)
    docs = joblib.load(DOCS_PATH)

    # TF-IDF vectors (dense for Qdrant)
    X = vec.transform([d["text"] for d in docs])
    X = X.toarray().astype(np.float32)
    dim = X.shape[1]

    qdrant = QdrantClient(url="http://127.0.0.1:6333")

    # recreate collection
    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = []
    for i, d in enumerate(docs):
        payload = {
            "doc_id": d["doc_id"],
            "title": d["title"],
            "text": d["text"],
        }
        points.append(PointStruct(id=i, vector=X[i].tolist(), payload=payload))

    qdrant.upsert(collection_name=COLLECTION, points=points)

    # BM25 index
    corpus_tokens = [tokenize(d["text"]) for d in docs]
    bm25 = BM25Okapi(corpus_tokens)

    joblib.dump({"collection": COLLECTION, "dim": dim}, OUT_META)
    joblib.dump({"bm25": bm25, "docs": docs}, OUT_BM25)

    print(f"✅ Qdrant loaded: {len(docs)} docs into {COLLECTION} (dim={dim})")
    print("✅ BM25 index saved")

if __name__ == "__main__":
    main()

