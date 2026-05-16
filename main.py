"""
SHL Assessment Recommender - FastAPI Service
Uses Google Gemini (free tier) for LLM calls.
Retrieval: Hybrid BM25 + TF-IDF.
"""

import json
import os
import re
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Gemini client ─────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ── Data models ───────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

# ── Catalog ───────────────────────────────────────────────────────────────────

CATALOG_PATH = Path(__file__).parent / "catalog.json"

def load_catalog() -> list[dict]:
    with open(CATALOG_PATH) as f:
        return json.load(f)

CATALOG: list[dict] = load_catalog()
CATALOG_URL_SET: set[str] = {item["url"] for item in CATALOG}

# ── Retrieval ─────────────────────────────────────────────────────────────────

def _make_doc_text(item: dict) -> str:
    parts = [
        item["name"],
        item.get("description", ""),
        " ".join(item.get("keywords", [])),
        " ".join(item.get("job_levels", [])),
        " ".join(item.get("test_type", [])),
    ]
    return " ".join(p for p in parts if p).lower()

_DOCS = [_make_doc_text(item) for item in CATALOG]
_TOKENIZED_DOCS = [doc.split() for doc in _DOCS]
_BM25 = BM25Okapi(_TOKENIZED_DOCS)
_TFIDF_VEC = TfidfVectorizer(ngram_range=(1, 2))
_TFIDF_MATRIX = _TFIDF_VEC.fit_transform(_DOCS)
logger.info(f"Index built over {len(CATALOG)} catalog entries.")


def retrieve(query: str, k: int = 10,
             test_type_filter: Optional[list[str]] = None,
             level_filter: Optional[list[str]] = None) -> list[dict]:
    q_lower = query.lower()
    bm25_scores = np.array(_BM25.get_scores(q_lower.split()))
    tfidf_scores = cosine_similarity(_TFIDF_VEC.transform([q_lower]), _TFIDF_MATRIX).flatten()

    def _norm(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn + 1e-9)

    combined = 0.6 * _norm(bm25_scores) + 0.4 * _norm(tfidf_scores)
    results = []
    for idx in np.argsort(combined)[::-1]:
        item = CATALOG[idx]
        if test_type_filter and not set(item.get("test_type", [])).intersection(test_type_filter):
            continue
        if level_filter:
            lvl = " ".join(item.get("job_levels", [])).lower()
            if not any(lf.lower() in lvl for lf in level_filter):
                continue
        results.append({**item, "_score": float(combined[idx])})
        if len(results) >= k:
            break
    return results


# ── Context extraction ────────────────────────────────────────────────────────

def extract_context(messages: list[Message]) -> dict:
    text = " ".join(m.content for m in messages).lower()
    ctx: dict = {"level": None, "test_types": []}

    if any(w in text for w in ["entry", "junior", "graduate", "fresher", "campus", "trainee", "intern"]):
        ctx["level"] = ["Entry-Level"]
    elif any(w in text for w in ["mid", "associate", "2 year", "3 year", "4 year", "intermediate"]):
        ctx["level"] = ["Mid-Professional"]
    elif any(w in text for w in ["senior", "5+ year", "lead", "principal"]):
        ctx["level"] = ["Professional", "Senior Manager"]
    elif any(w in text for w in ["manager", "management", "team lead", "head of"]):
        ctx["level"] = ["Manager", "Senior Manager"]
    elif any(w in text for w in ["director", "vp", "c-suite", "ceo", "cto", "executive"]):
        ctx["level"] = ["Director", "Executive", "Senior Manager"]

    if any(w in text for w in ["personality", "behaviour", "opq", "trait"]):
        ctx["test_types"].append("P")
    if any(w in text for w in ["cognitive", "aptitude", "reasoning", "ability", "numerical", "verbal"]):
        ctx["test_types"].append("A")
    if any(w in text for w in ["knowledge", "technical", "coding test", "programming"]):
        ctx["test_types"].append("K")
    if any(w in text for w in ["situational", "sjt", "scenario", "behavioural test"]):
        ctx["test_types"].append("B")
    if any(w in text for w in ["simulation", "practical", "hands-on"]):
        ctx["test_types"].append("S")
    return ctx


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the SHL Assessment Recommender — a helpful assistant for hiring managers and recruiters.

RULES:
1. Only recommend assessments from the CATALOG below. Never invent names or URLs.
2. Stay on-topic: SHL assessments and hiring needs only. Refuse general HR/legal/salary questions.
3. Every URL must exactly match a URL from the catalog.

BEHAVIORS:
- CLARIFY: If the first message is vague (no role/context), ask ONE clarifying question. Do not recommend yet.
- RECOMMEND: Once you have role + context, recommend 1-10 assessments with name and URL.
- REFINE: If user changes constraints, update shortlist without starting over.
- COMPARE: Use only catalog data.

OUTPUT FORMAT — RETURN VALID JSON ONLY, NO OTHER TEXT BEFORE OR AFTER:
{
  "reply": "<your message to the user>",
  "recommendations": [
    {"name": "<exact name from catalog>", "url": "<exact url from catalog>", "test_type": "<code>"}
  ],
  "end_of_conversation": false
}

Notes:
- recommendations = [] when clarifying or refusing
- recommendations has 1-10 items when giving a shortlist
- end_of_conversation = true only when user is fully satisfied and done
- test_type codes: A=Ability, P=Personality, K=Knowledge, B=Behavioural/SJT, S=Simulation

CATALOG:
{catalog}
"""

def build_system_prompt(candidates: list[dict]) -> str:
    lines = []
    for item in candidates:
        types = ",".join(item.get("test_type", []))
        levels = ", ".join(item.get("job_levels", []))
        lines.append(
            f"Name: {item['name']}\n"
            f"URL: {item['url']}\n"
            f"Type: {types} | Levels: {levels} | Duration: {item.get('duration_minutes','?')}min\n"
            f"Description: {item['description']}\n"
        )
    return SYSTEM_PROMPT.replace("{catalog}", "\n---\n".join(lines))


# ── Gemini call ───────────────────────────────────────────────────────────────

def call_gemini(system: str, messages: list[Message]) -> str:
    if not gemini_client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    # Build history (all except last message)
    history = []
    for m in messages[:-1]:
        role = "user" if m.role == "user" else "model"
        history.append(types.Content(role=role, parts=[types.Part(text=m.content)]))

    last_msg = messages[-1].content

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=history + [types.Content(role="user", parts=[types.Part(text=last_msg)])],
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            max_output_tokens=900,
        ),
    )
    return response.text


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_response(raw: str) -> ChatResponse:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        return ChatResponse(reply=raw.strip(), recommendations=[], end_of_conversation=False)
    try:
        obj = json.loads(cleaned[start:end])
    except json.JSONDecodeError:
        logger.warning(f"JSON parse failed: {cleaned[start:start+200]}")
        return ChatResponse(reply=raw.strip(), recommendations=[], end_of_conversation=False)

    recs = []
    for r in obj.get("recommendations", [])[:10]:
        url = r.get("url", "")
        if url in CATALOG_URL_SET:
            recs.append(Recommendation(
                name=r.get("name", ""),
                url=url,
                test_type=r.get("test_type", ""),
            ))

    return ChatResponse(
        reply=obj.get("reply", raw.strip()),
        recommendations=recs,
        end_of_conversation=bool(obj.get("end_of_conversation", False)),
    )


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    messages = request.messages[-8:]
    ctx = extract_context(messages)
    user_text = " ".join(m.content for m in messages if m.role == "user").strip()
    query = user_text or "general assessment"

    candidates = retrieve(query=query, k=15,
                          test_type_filter=ctx["test_types"] or None,
                          level_filter=ctx["level"])
    if len(candidates) < 6:
        extra = retrieve(query=query, k=12)
        seen = {c["name"] for c in candidates}
        for e in extra:
            if e["name"] not in seen:
                candidates.append(e)
                seen.add(e["name"])

    system = build_system_prompt(candidates[:15])

    try:
        raw = call_gemini(system=system, messages=messages)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    return parse_response(raw)
