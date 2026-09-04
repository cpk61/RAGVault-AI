# RAGVault-AI

**Local retrieval-augmented QA service with chunking, TF-IDF retrieval, citations and confidence thresholds.**

## Why this project matters
This project demonstrates a self-contained RAG-style retrieval service that works without paid model APIs. It focuses on evidence retrieval, citations, confidence gating and deterministic behavior.

## Features
- FastAPI document ingest + query API
- Word-based chunking
- TF-IDF + cosine similarity retrieval
- Source citations with chunk IDs and scores
- Confidence threshold with insufficient-evidence fallback
- Unit tests

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger UI.

## Test
```bash
pytest -q
```

## Architecture
```text
Documents -> Chunker -> TF-IDF Index -> Query -> Similarity Ranking -> Confidence Gate -> Answer + Citations
```

## Portfolio note
The core demo is local and uses no external API keys. In production, replace the extractive answer stage with an authorized LLM only after preserving citation and confidence safeguards.
