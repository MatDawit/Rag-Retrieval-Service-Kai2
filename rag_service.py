import os
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from fastembed import TextEmbedding
from pymilvus import connections, Collection


# ---- Required env vars (set in Render dashboard) ----
MILVUS_URI = os.environ.get("MILVUS_URI", "").strip()
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "").strip()
MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", "rag_chunks").strip()

# Shared secret between ask.php and this service
RAG_API_KEY = os.environ.get("RAG_API_KEY", "").strip()

# Embedding model name for fastembed (use one you know works on your machine)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5").strip()


app = FastAPI()

# Load embedder once (warm containers reuse it)
try:
    embedder = TextEmbedding(model_name=EMBED_MODEL)
except Exception as e:
    embedder = None
    EMBED_ERR = str(e)


class SearchReq(BaseModel):
    text: str
    top_k: int = 12


def require_env():
    if not MILVUS_URI or not MILVUS_TOKEN:
        raise HTTPException(status_code=500, detail="Missing MILVUS_URI or MILVUS_TOKEN")
    if embedder is None:
        raise HTTPException(status_code=500, detail=f"Embedding model failed to load: {globals().get('EMBED_ERR','unknown')}")


def auth(x_api_key: Optional[str]):
    # If you set RAG_API_KEY, enforce it. If blank, allow open (not recommended).
    if RAG_API_KEY and x_api_key != RAG_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


HEALTH_VERSION = "health-head-v3"

@app.get("/health")
def health():
    return {
        "ok": True,
        "embed_model": EMBED_MODEL,
        "milvus_collection": MILVUS_COLLECTION,
        "version": HEALTH_VERSION,
    }


@app.post("/search")
def search(req: SearchReq, x_api_key: Optional[str] = Header(default=None)):
    auth(x_api_key)
    require_env()

    q = (req.text or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing text")

    top_k = max(1, min(int(req.top_k), 30))

    # ---- Embed ----
    try:
        vec = next(embedder.embed([q])).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # ---- Milvus search ----
    try:
        connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
        col = Collection(MILVUS_COLLECTION)
        col.load()

        res = col.search(
            data=[vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["doc_id", "title", "chunk_index", "text"],
        )

        hits: List[Dict[str, Any]] = []
        for h in res[0]:
            hits.append({
                "score": float(h.score),
                "doc_id": h.entity.get("doc_id"),
                "title": h.entity.get("title"),
                "chunk_index": int(h.entity.get("chunk_index")),
                "text": h.entity.get("text"),
            })

        return {"hits": hits}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus search failed: {e}")
