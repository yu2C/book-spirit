"""LLM query planner: rewrite user questions into retrieval-friendly queries."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from core.config import OLLAMA_MODEL, OLLAMA_URL
from core.retrieval import is_overview_question, retrieval_top_k

USE_QUERY_PLANNER = os.getenv("USE_QUERY_PLANNER", "false").lower() in ("1", "true", "yes")

PLANNER_PROMPT = """你是檢索查詢規劃器，不回答問題本身。
根據使用者問題與書籍資訊，輸出 JSON（不要 markdown）：
{
  "intent": "overview|fact|person|event|quote|other",
  "queries": ["檢索用短查詢1", "檢索用短查詢2"],
  "top_k": 3,
  "notes_for_generator": "給回答模型的簡短提示（繁中）"
}
規則：
- queries 1～3 條，適合向量/BM25 搜尋，短而具體
- 英文書必須至少一條英文查詢
- overview 類加上 introduction prologue summary timeline narrative
- 不要捏造書中未出現的專有名詞
- top_k：overview 用 8，具體事實用 3～5
"""


@dataclass
class QueryPlan:
    intent: str = "other"
    queries: List[str] = field(default_factory=list)
    top_k: Optional[int] = None
    notes_for_generator: str = ""
    raw: str = ""

    def effective_queries(self, fallback: str) -> List[str]:
        cleaned = [q.strip() for q in self.queries if q and q.strip()]
        return cleaned if cleaned else [fallback]


def _parse_plan_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def plan_query(
    question: str,
    *,
    book_id: str = "",
    book_title: str = "",
    language_hint: str = "en",
    ollama_model: str = OLLAMA_MODEL,
    ollama_url: str = OLLAMA_URL,
    timeout: int = 60,
) -> QueryPlan:
    """Call Ollama to produce retrieval queries. On failure returns empty plan."""
    user_block = (
        f"book_id: {book_id}\n"
        f"book_title: {book_title}\n"
        f"language_hint: {language_hint}\n"
        f"user_question: {question}\n"
    )
    prompt = f"{PLANNER_PROMPT}\n\n{user_block}\nJSON:"
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
    except requests.RequestException:
        return QueryPlan()

    data = _parse_plan_json(raw)
    top_k = data.get("top_k")
    if isinstance(top_k, (int, float)):
        top_k = int(top_k)
    else:
        top_k = None

    return QueryPlan(
        intent=str(data.get("intent") or "other"),
        queries=[str(q) for q in (data.get("queries") or []) if q],
        top_k=top_k,
        notes_for_generator=str(data.get("notes_for_generator") or ""),
        raw=raw,
    )


def build_query_plan(
    question: str,
    book_id: Optional[str],
) -> Optional[QueryPlan]:
    if not USE_QUERY_PLANNER:
        return None
    from ingest.books_registry import get_book

    entry = get_book(book_id) if book_id and book_id != "all" else None
    title = entry.book_title if entry else ""
    lang = "en" if title and sum(1 for c in title if ord(c) < 128) > len(title) // 2 else "zh"
    plan = plan_query(question, book_id=book_id or "", book_title=title, language_hint=lang)
    if not plan.queries and is_overview_question(question):
        plan.queries = [
            "introduction prologue summary main narrative",
            question,
        ]
    if plan.top_k is None:
        plan.top_k = retrieval_top_k(question, 3)
    return plan
