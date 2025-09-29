from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.services.extraction import extract_text_from_file
from app.services.similarity import compute_similarity
from app.services.review import generate_review


app = FastAPI(title="Resume Reviewer API", version="0.1.0")


class ExtractResponse(BaseModel):
    filename: str
    content_type: str
    num_characters: int
    text: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        text = extract_text_from_file(file_bytes=file_bytes, filename=file.filename)
        return ExtractResponse(
            filename=file.filename or "uploaded",
            content_type=file.content_type or "application/octet-stream",
            num_characters=len(text),
            text=text,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to extract text")


class SimilarityRequest(BaseModel):
    resume_text: str
    job_text: str


class SimilarityResponse(BaseModel):
    score: float


@app.post("/similarity", response_model=SimilarityResponse)
def similarity(payload: SimilarityRequest):
    try:
        score = compute_similarity(payload.resume_text, payload.job_text)
        return SimilarityResponse(score=round(float(score), 6))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to compute similarity")


class ReviewRequest(BaseModel):
    resume_text: str
    job_text: str
    debug: bool | None = False


class ReviewResponse(BaseModel):
    match_score: float
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]
    raw_content: str | None = None
    parsing_note: str | None = None


@app.post("/review", response_model=ReviewResponse, response_model_exclude_none=True)
async def review(payload: ReviewRequest):
    try:
        score = compute_similarity(payload.resume_text, payload.job_text)
        result = await generate_review(payload.resume_text, payload.job_text, score)
        # Ensure rounding consistency on score in the response
        result["match_score"] = round(float(result.get("match_score", score)), 6)
        if not payload.debug:
            result.pop("raw_content", None)
            result.pop("parsing_note", None)
        return ReviewResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate review")


@app.get("/ollama/health")
async def ollama_health():
    import os
    import httpx
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{base}/api/tags")
            r.raise_for_status()
            data = r.json()
        return {"status": "ok", "base": base, "models": [m.get("name") for m in data.get("models", [])]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama unreachable at {base}: {e}")

