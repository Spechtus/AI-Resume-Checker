import json
import os
from typing import Any, Dict, List
import re

import httpx

# Ollama local API
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"


def _build_prompt(resume_text: str, job_text: str, similarity: float) -> List[Dict[str, str]]:
    system = (
        "You are an expert resume reviewer. You analyze a resume against a job description. "
        "Return STRICT JSON only with keys: strengths (array of strings), weaknesses (array of strings), suggestions (array of strings). "
        "Do not include extra commentary, no markdown, no code fences, no preface. Output must be a single JSON object."
    )
    user = (
        f"Resume text:\n{resume_text}\n\n"
        f"Job description:\n{job_text}\n\n"
        f"Similarity score (0-1): {similarity:.4f}\n"
        "Provide concise, specific bullet points."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


async def generate_review(
    resume_text: str, job_text: str, similarity: float
) -> Dict[str, Any]:
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    messages = _build_prompt(resume_text, job_text, similarity)

    payload = {
        "model": model,
        "messages": messages,
        "options": {"temperature": 0.2},
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            OLLAMA_CHAT_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    # Ollama chat response: { "message": { "content": "..." }, ... }
    content = data.get("message", {}).get("content", "{}")
    parsing_note = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract the first JSON object from mixed content (e.g., with prose or code fences)
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                parsed = json.loads(match.group(0))
                parsing_note = "Extracted JSON object from mixed content"
            except json.JSONDecodeError:
                parsed = {"strengths": [], "weaknesses": [], "suggestions": []}
                parsing_note = "Failed to parse extracted JSON; returned empty arrays"
        else:
            parsed = {"strengths": [], "weaknesses": [], "suggestions": []}
            parsing_note = "No JSON object found in model output; returned empty arrays"

    # Normalize fields
    strengths = parsed.get("strengths") or []
    weaknesses = parsed.get("weaknesses") or []
    suggestions = parsed.get("suggestions") or []

    return {
        "match_score": float(similarity),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "raw_content": content,
        "parsing_note": parsing_note,
    }

