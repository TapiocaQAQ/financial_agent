# app/ingest.py
import os, glob
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from app.embeddings import EMB

# EMB = SentenceTransformer("BAAI/bge-m3")

# ✅ 使用持久化客戶端，讓不同行程共用同一套資料
client = chromadb.PersistentClient(path="./index/chroma")
DB = client.get_or_create_collection(name="kb_main", embedding_function=EMB)

def embed(texts: list[str]) -> list[list[float]]:
    """
    Chroma 的 SentenceTransformerEmbeddingFunction 是可呼叫物件，
    直接 EMB(texts) 即可；回傳可能是 list 或 numpy array，這裡統一轉成純 list。
    """
    vecs = EMB(texts)  # ❗不要用 .encode()
    # 統一成純 Python list[ list[float] ]
    try:
        # numpy -> list
        return [list(map(float, v)) for v in vecs]
    except Exception:
        return vecs

def ingest_data(data_dir="data"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    ids, metas, texts = [], [], []
    for fp in glob.glob(os.path.join(data_dir, "*.md")):
        with open(fp, "r", encoding="utf-8") as f:
            raw = f.read()
        base = os.path.basename(fp)
        for i, ch in enumerate(splitter.split_text(raw)):
            ids.append(f"{base}-{i}")
            metas.append({"source": base})
            texts.append(ch)



    if texts:
        DB.add(documents=texts, ids=ids, metadatas=metas, embeddings=embed(texts))
    
    print(f"Ingested {len(texts)} chunks from {data_dir}")
    print("Collections:", [c.name for c in client.list_collections()])
    print("Count in kb_main:", DB.count())  # ✅ 應該 > 0

if __name__ == "__main__":
    ingest_data()
