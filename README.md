# RAGVault-AI — Remote GenAI / RAG Intern Portfolio

**Local retrieval-augmented QA service with chunking, TF-IDF retrieval, citations, confidence thresholds and container-ready deployment notes.**

This repository is tailored for remote GenAI/RAG internships that ask for practical evidence rather than only coursework.

## What is implemented
- FastAPI ingest + query API
- Word-based chunking
- TF-IDF + cosine-similarity retrieval
- Source citations with chunk IDs and similarity scores
- Confidence threshold with insufficient-evidence fallback
- Unit tests
- Dockerfile for reproducible deployment

## Why this matters
RAG systems fail when retrieval is weak but the generation layer answers confidently anyway. This project keeps retrieval evidence visible and adds a confidence gate before returning an answer.

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload
pytest -q
```

## Run with Docker
```bash
docker build -t ragvault-ai .
docker run -p 8000:8000 ragvault-ai
```

## Architecture
`Documents -> Chunker -> TF-IDF Index -> Query -> Similarity Ranking -> Confidence Gate -> Answer + Citations`

## Production extension
- embedding model + vector DB behind the same retrieval interface
- versioned document ingestion
- evaluation set for retrieval precision and groundedness
- optional authorized LLM generation stage that must preserve citations

## Integrity note
The current project is local and API-key free. It demonstrates retrieval and evidence handling; it does not claim production deployment or real customer data.