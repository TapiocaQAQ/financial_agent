# app/embeddings.py
import os
import torch
from chromadb.utils import embedding_functions

MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# chromadb 內建的 SentenceTransformer wrapper
EMB = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=MODEL_NAME,
    device=DEVICE,
    normalize_embeddings=True,   
)
