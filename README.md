# Resume Reviewer (AI-Resume-Checker)

An AI-powered system that analyzes resumes against job descriptions, computes a Resume–Job Match Score, and generates strengths, weaknesses, and improvement suggestions using Groq and Hugging Face APIs.

This repository is implemented in three parts:
- Part 1: Extracting Resume Text (PDF, DOCX, TXT)
- Part 2: Computing Resume–Job Similarity
- Part 3: Generating a Detailed Resume Review

## Quickstart (Docker Compose)

1) Edit `env.txt` and set your keys for later parts:

```
GROQ_API_KEY=changeme
HF_API_KEY=changeme
```

2) Build and run:

```bash
docker compose up --build
```

3) API will be at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Tests

```bash
pytest -q
```

## Part 1 API

POST `/extract` (multipart/form-data)
- file: uploaded resume (`.pdf`, `.docx`, `.txt`)

Response:

```json
{
  "filename": "resume.pdf",
  "content_type": "application/pdf",
  "num_characters": 1234,
  "text": "...extracted text..."
}
```

## Part 2 API

POST `/similarity` (application/json)

Request:

```json
{
  "resume_text": "I am a Python developer with FastAPI experience",
  "job_text": "We need a Python developer skilled in FastAPI"
}
```

Response:

```json
{ "score": 0.72 }
```

## Roadmap
- Part 2: Add similarity scoring endpoint `/similarity` ✅
- Part 3: Add review generation endpoint `/review` ✅

## Part 3 API

POST `/review` (application/json)

Request:

```json
{
  "resume_text": "Python developer with FastAPI and NLP experience",
  "job_text": "We need a Python engineer with FastAPI skills"
}
```

Response:

```json
{
  "match_score": 0.73,
  "strengths": ["Strong Python skills", "FastAPI experience"],
  "weaknesses": ["Limited cloud exposure"],
  "suggestions": ["Gain AWS experience", "Quantify project impact"]
}
```

Environment variables (set in `.env` for Docker or your shell locally):

```
# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

Ensure you have Ollama installed and the model pulled locally:

```bash
brew install ollama # macOS
ollama serve &
ollama pull llama3.1
```