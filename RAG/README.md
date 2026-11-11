# HUFLIT RAG Robot

The RAG application, crawler, Elasticsearch indexer, and production knowledge
base live in this folder. The knowledge base includes the HUFLIT sources and
the supplementary results PDF for HIRA Paper 315 in `documents/`.

## Run the API

Use Python 3.12 and Docker Desktop. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
docker compose up -d
.\.venv\Scripts\python.exe huflit_es_indexer.py
```

Start LM Studio, load a chat model, and enable its local server on port 1234.
Then start the RAG API:

```powershell
.\.venv\Scripts\python.exe api_server.py
```

Open `http://localhost:8000`. See [LM_STUDIO.md](LM_STUDIO.md) for model server
settings. The API also loads offline voice models from `models/`: a
`faster-whisper-large-v3-turbo` model directory and the Piper files
`vi_VN-vais1000-medium.onnx` plus its matching `.json`. Model weights and
virtual environments are excluded from Git and must be installed separately.

Stop Elasticsearch with `docker compose down`.

## PDF knowledge source

`huflit_rag_db_production.json` already contains extracted text and retrieval
chunks from `documents/HIRA_Paper315_Supplementary_ResultsOnly.pdf`. To rebuild
that entry after replacing the PDF, run:

```powershell
.\.venv\Scripts\python.exe ingest_pdf_to_rag.py
.\.venv\Scripts\python.exe huflit_es_indexer.py
```

The indexer recreates its Elasticsearch indices from the complete JSON corpus.
The PDF text is indexed as a research supplement; table layout is not retained
by text extraction.
