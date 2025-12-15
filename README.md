# e-Consultation Analysis Platform

AI-powered platform for analyzing stakeholder comments on government drafts.

## Setup
- Install Ollama and pull llama3:8b.
- pip install -r requirements.txt
- Run backend: uvicorn backend.main:app --reload
- Run frontend: streamlit run frontend/app.py

## Features
- Upload comments CSV/Excel.
- AI processing with LLaMA 3:8B (sentiment, summary, keywords, recommendations).
- Multi-language translation.
- Topic clustering and visualizations.
- Reports in PDF/Excel.
- Scalable batch processing.