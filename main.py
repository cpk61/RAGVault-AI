from fastapi import FastAPI
from pydantic import BaseModel
from core import RAGVault

app = FastAPI(title='RAGVault AI')
rag = RAGVault()

class Doc(BaseModel):
    doc_id: str
    text: str

class Q(BaseModel):
    question: str
    k: int = 3

@app.post('/documents')
def add(d: Doc):
    rag.add_document(d.doc_id, d.text)
    return {'chunks': len(rag.chunks)}

@app.post('/query')
def query(q: Q):
    return rag.query(q.question, q.k)
