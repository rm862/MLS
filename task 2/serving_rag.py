import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn
from pydantic import BaseModel
import threading, queue, time

app = FastAPI()

# Example documents in memory
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

# 1. Load embedding model
# Modify model path accordingly
EMBED_MODEL_NAME = "/content/drive/MyDrive/MLSystems/edin-mls-25-spring/task-2/e5_model"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

# Basic Chat LLM
# Modify model path accordingly
chat_pipeline = pipeline("text-generation", model="/content/drive/MyDrive/MLSystems/edin-mls-25-spring/task-2/qwen_model", truncation=True)

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

# ----------------- Batch Processing Mechanism for Batched Requests -----------------

# Create a global request queue
request_queue = queue.Queue()

# Define batch processing parameters
MAX_BATCH_SIZE = 4      # Maximum number of requests per batch
MAX_WAITING_TIME = 1    # Maximum waiting time (in seconds) before processing a batch

def batch_worker():
    """
    Background worker function that continuously monitors the request queue.
    It aggregates incoming requests into a batch (up to MAX_BATCH_SIZE or waits until MAX_WAITING_TIME),
    processes the batch by calling the LLM pipeline in batch mode, and then sets each request's result.
    """
    while True:
        batch = []
        try:
            # Wait for the first request; block for up to MAX_WAITING_TIME seconds
            item = request_queue.get(timeout=MAX_WAITING_TIME)
            batch.append(item)
            batch_start = time.time()
        except queue.Empty:
            continue

        # Accumulate requests until reaching MAX_BATCH_SIZE or time exceeds MAX_WAITING_TIME
        while len(batch) < MAX_BATCH_SIZE:
            elapsed = time.time() - batch_start
            remaining = MAX_WAITING_TIME - elapsed
            if remaining <= 0:
                break
            try:
                item = request_queue.get(timeout=remaining)
                batch.append(item)
            except queue.Empty:
                break

        # Build prompts for each request in the batch
        prompts = []
        for payload, future in batch:
            query = payload.get("query")
            k_val = payload.get("k", 2)
            query_emb = get_embedding(query)
            retrieved_docs = retrieve_top_k(query_emb, k_val)
            context = "\n".join(retrieved_docs)
            prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
            prompts.append(prompt)

        try:
            # Call the pipeline in batch mode with explicit truncation
            batch_results = chat_pipeline(prompts, max_length=50, do_sample=True, truncation=True)
        except Exception as e:
            # If error occurs, set error message for all requests in the batch
            for _, future in batch:
                future["result"] = f"Error: {e}"
                future["event"].set()
            continue

        # Process each result; each element in batch_results may be a list or a dict
        for (payload, future), res in zip(batch, batch_results):
            if isinstance(res, list) and len(res) > 0:
                sub_res = res[0]
                if isinstance(sub_res, dict):
                    result_text = sub_res.get("generated_text", "")
                else:
                    result_text = "Error: unexpected sub_res format"
            elif isinstance(res, dict):
                result_text = res.get("generated_text", "")
            else:
                result_text = "Error: unexpected result format"
            future["result"] = result_text
            future["event"].set()

# Start the background worker thread (daemon mode)
worker_thread = threading.Thread(target=batch_worker, daemon=True)
worker_thread.start()

# --------------------------------------------------------------------------------------------

def get_embedding(text: str) -> np.ndarray:
    """
    Compute an average-pool embedding for the input text.
    """
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Precompute document embeddings
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])

def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """
    Retrieve top-k documents via dot-product similarity.
    """
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]

def rag_pipeline(query: str, k: int = 2) -> str:
    """
    Process a single query by computing its embedding, retrieving
    top-k documents, constructing the prompt, and generating the answer.
    (This function remains for debugging; batched requests are handled by batch_worker.)
    """
    query_emb = get_embedding(query)
    retrieved_docs = retrieve_top_k(query_emb, k)
    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
    generated = chat_pipeline(prompt, max_length=50, do_sample=True, truncation=True)[0]["generated_text"]
    return generated

# ----------------- Endpoints -----------------

# Define request model for FastAPI
class QueryRequest(BaseModel):
    query: str
    k: int = 2

@app.post("/rag")
def predict(payload: QueryRequest):
    """
    Endpoint that uses the request queue and batch processing.
    """
    # Using payload.dict() here; if using Pydantic V2, replace with model_dump()
    future = {"event": threading.Event(), "result": None}
    request_queue.put((payload.dict(), future))
    future["event"].wait()  # Block until result is set by the batch worker
    return {
        "query": payload.query,
        "result": future["result"],
    }

@app.post("/rag_no_batch")
def predict_no_batch(payload: QueryRequest):
    """
    Endpoint that directly processes the request without using a request queue or batcher.
    """
    return {
        "query": payload.query,
        "result": rag_pipeline(payload.query, payload.k)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)