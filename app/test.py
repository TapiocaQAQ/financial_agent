from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse

import os, glob
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

EMB = SentenceTransformer("BAAI/bge-m3")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}