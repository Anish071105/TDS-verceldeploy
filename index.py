import os
import base64
import io
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai

client = openai.OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

EMBEDDING_MODEL_NAME  = "text-embedding-3-small"
GENERATION_MODEL_NAME = "gpt-3.5-turbo"
EMBEDDING_FILE        = Path(__file__).parent / "embedding.npz"

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    image:   Optional[str] = None  # base64 Data‑URI

class LinkResponse(BaseModel):
    url:  str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links:  List[LinkResponse]

# global storage
embeddings_data: Optional[np.ndarray] = None
chunks_metadata: Optional[List[Dict]]  = None

# ——— 1) IMAGE CAPTIONING using GPT-3.5-TURBO ———
async def get_image_description(b64_data_uri: str) -> Optional[str]:
    # Strip off any prefix like "data:image/...;base64,"
    _, _, b64data = b64_data_uri.partition("base64,")
    try:
        # We include the first 1000 chars to avoid payload too large
        sample = b64data[:1000]
        prompt = (
            "You are an AI assistant that describes images.\n"
            "Provide a concise (1–2 sentence) factual description of the image.\n"
            "The image is provided as base64 below; you do not need to decode it yourself.\n"
            "Only describe what you see; do not guess.\n"
            "Base64 sample:\n" + sample
        )
        resp = await asyncio.to_thread(lambda: client.chat.completions.create(
            model=GENERATION_MODEL_NAME,
            messages=[{"role":"user","content":prompt}]
        ))
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Image description error:", e)
        return None

# ——— 2) TEXT EMBEDDING ———
async def get_text_embedding(text: str) -> Optional[np.ndarray]:
    try:
        r = await asyncio.to_thread(lambda: client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=text
        ))
        return np.array(r.data[0].embedding)
    except Exception as e:
        print("❌ Text embed error:", e)
        return None

# ——— 3) COMBINED EMBEDDING ———
async def get_combined_embedding(question: str, image_b64: Optional[str]) -> Optional[np.ndarray]:
    # 3.a) prepend image description if present
    if image_b64:
        desc = await get_image_description(image_b64)
        if desc:
            question = f"[Image description: {desc}]\n{question}"
    # 3.b) embed the enriched question
    return await get_text_embedding(question)

# ——— 4) FIND BEST CHUNK ———
async def find_best_chunk(
    question: str,
    image_b64: Optional[str],
    embeddings: np.ndarray,
    metadata:   List[Dict]
) -> Optional[Dict]:
    if embeddings is None or embeddings.size == 0:
        return None

    emb = await get_combined_embedding(question, image_b64)
    if emb is None:
        return None

    dots  = np.dot(embeddings, emb)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
    sims  = dots / np.where(norms==0, 1e-9, norms)

    idx = int(np.argmax(sims))
    score = float(sims[idx])
    if score < 0.3:
        return None

    chunk = metadata[idx].copy()
    chunk["similarity"] = score
    return chunk

# ——— 5) GENERATE ANSWER ———
async def generate_answer(question: str, chunk: Dict) -> str:
    ctx = chunk.get("text", "")
    if not ctx:
        return "I don't know."
    messages = [
        {"role":"system","content":"You are a helpful assistant. Answer only from the context; otherwise say 'I don't know'."},
        {"role":"user","content":f"Context:\n{ctx}\n\nQuestion: {question}"}
    ]
    try:
        r = await asyncio.to_thread(lambda: client.chat.completions.create(
            model=GENERATION_MODEL_NAME,
            messages=messages
        ))
        return r.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Generation error:", e)
        return "I don't know."

# ——— 6) STARTUP: load or build embeddings ———
@app.on_event("startup")
async def startup_event():
    global embeddings_data, chunks_metadata

    try:
        EMBED_PATH = Path(__file__).parent / "embedding.npz"
        if not EMBED_PATH.exists():
            print("📦 Downloading embedding...")
            url = "https://github.com/Anish071105/TDS-verceldeploy/releases/download/v1.0/embedding.zip"
            zip_path = Path(__file__).parent / "embedding.zip"
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(Path(__file__).parent)

        data = np.load(EMBED_PATH, allow_pickle=True)
        embeddings_data = np.array(data["vectors"])
        chunks_metadata = list(data["metadata"])
        print(f"✅ Loaded {len(embeddings_data)} embeddings")

    except Exception as e:
        print("❌ Failed loading embeddings:", e)
        embeddings_data = np.array([])
        chunks_metadata = []
        
# ——— 7) API ROUTE ———
@app.post("/api/", response_model=QueryResponse)
async def api_handler(payload: QueryRequest) -> QueryResponse:
    q = payload.question.strip()
    if not q:
        raise HTTPException(400, "Question is empty")

    best = await find_best_chunk(q, payload.image, embeddings_data, chunks_metadata)
    if not best:
        return QueryResponse(answer="I don't know.", links=[])

    ans = await generate_answer(q, best)
    url = best.get("main_url") or best.get("post_url") or ""
    snippet = best.get("text", "")
    links = [LinkResponse(url=url, text=snippet)] if url else []
    return QueryResponse(answer=ans, links=links)

@app.get("/")
async def root():
    return {"message":"TDS Virtual TA API is running","endpoints":["/api/","/health"]}

@app.get("/health")
async def health():
    return {
        "status":"ok",
        "embeddings_loaded": "embeddings_loaded": bool(embeddings_data is not None and embeddings_data.size > 0),
        "num_embeddings": embeddings_data.shape[0] if embeddings_data is not None else 0
    }
@app.get("/debug-files")
def debug_files():
    try:
        return {"files": os.listdir(Path(__file__).parent)}
    except Exception as e:
        return {"error": str(e)}
