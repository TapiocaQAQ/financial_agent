# app/retriever.py
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

EMB = SentenceTransformer("BAAI/bge-m3")
CROSS = CrossEncoder("BAAI/bge-reranker-large")

# ingest.py 完全相同的持久化路徑與 collection 名稱
client = chromadb.PersistentClient(path="./index/chroma")
DB = client.get_collection("kb_main")

def hybrid_search(query: str, k=4):
    # 基本向量檢索（若庫是空的，這裡會拿不到文件）
    vec = DB.query(
        query_embeddings=EMB.encode([query], normalize_embeddings=True).tolist(),
        n_results=max(k*6, 8),
        include=["documents","metadatas"]
    )
    if not vec["documents"] or not vec["documents"][0]:
        return []
    cands = list(zip(vec["documents"][0], vec["metadatas"][0]))
    pairs = [[query, d] for d,_ in cands]
    scores = CROSS.predict(pairs)
    ranked = [x for _,x in sorted(zip(scores, cands), key=lambda t: t[0], reverse=True)][:k]
    return [{"text": d, "source": m["source"]} for d,m in ranked]
