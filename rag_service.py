import os
from threading import Lock
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel

from fastembed import TextEmbedding
from pymilvus import connections, Collection


MILVUS_URI = os.environ.get("MILVUS_URI", "").strip()
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "").strip()
MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", "rag_chunks").strip()
RAG_API_KEY = os.environ.get("RAG_API_KEY", "").strip()
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5").strip()

HEALTH_VERSION = "health-v4"

app = FastAPI()

# ---- Embedder (load once per container) ----
EMBED_ERR = None
try:
    embedder = TextEmbedding(model_name=EMBED_MODEL)
except Exception as e:
    embedder = None
    EMBED_ERR = str(e)

# ---- Milvus (lazy singleton per container) ----
_milvus_lock = Lock()
_col = None
_connected = False

def require_env():
    if not MILVUS_URI or not MILVUS_TOKEN:
        raise HTTPException(status_code=500, detail="Missing MILVUS_URI or MILVUS_TOKEN")
    if embedder is None:
        raise HTTPException(status_code=500, detail=f"Embedding model failed to load: {EMBED_ERR or 'unknown'}")
    if not RAG_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing RAG_API_KEY")

def auth(x_api_key: Optional[str]):
    if x_api_key != RAG_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def get_collection() -> Collection:
    global _col, _connected
    with _milvus_lock:
        if not _connected:
            connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
            _connected = True
        if _col is None:
            _col = Collection(MILVUS_COLLECTION)
            _col.load()
        return _col

class SearchReq(BaseModel):
    text: str
    top_k: int = 12

@app.get("/")
def root():
    return {"ok": True, "service": "rag-retrieval"}

@app.get("/health")
def health():
    # must be super-light + always fast
    return {
        "ok": True,
        "embed_model": EMBED_MODEL,
        "milvus_collection": MILVUS_COLLECTION,
        "version": HEALTH_VERSION,
    }

@app.head("/health", include_in_schema=False)
def health_head():
    return Response(status_code=200)

@app.get("/ready")
def ready():
    # slightly deeper: verifies embedder loaded and env configured
    require_env()
    return {"ok": True}

@app.post("/search")
def search(req: SearchReq, x_api_key: Optional[str] = Header(default=None)):
    auth(x_api_key)
    require_env()

    q = (req.text or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing text")
    if len(q) > 2000:
        raise HTTPException(status_code=413, detail="Query too long")

    top_k = max(1, min(int(req.top_k), 30))

    try:
        vec = next(embedder.embed([q])).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    try:
        col = get_collection()
        res = col.search(
            data=[vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["doc_id", "title", "chunk_index", "text"],
        )

        hits = []
        for h in res[0]:
            hits.append({
                "score": float(h.score),
                "doc_id": h.entity.get("doc_id"),
                "title": h.entity.get("title"),
                "chunk_index": int(h.entity.get("chunk_index")),
                "text": h.entity.get("text"),
            })

        return {"hits": hits}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus search failed: {e}")
