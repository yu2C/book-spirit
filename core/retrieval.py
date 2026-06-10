"""Retrieval helpers: overview detection, query expansion, boilerplate filtering."""

from __future__ import annotations

import re
from typing import Dict, List

_OVERVIEW_RE = re.compile(
    r"(大綱|敘事|講什麼|講甚麼|在講|主題|概況|摘要|總覽|整本|全書|"
    r"主要內容|故事講|overview|summary|what is this book|main narrative|"
    r"table of contents|outline|big picture)",
    re.IGNORECASE,
)

_THEMATIC_RE = re.compile(
    r"(理想|主義|觀點|立場|分歧|概念|idealism|greed|versus|vs\.|community|"
    r"describe|perspective|debate|觀念)",
    re.IGNORECASE,
)

_BOILERPLATE_RE = re.compile(
    r"(acknowledg(e)?ments|discover more|about the author|notes\s*$|"
    r"tap here to learn|explore book giveaways|sneak peeks|to my parents|"
    r"thank you for listening|favorite authors|book giveaways|"
    r"I\.?\s*F\.?\s*STONE|BENJAMIN C\.?\s*BRADLEE|grateful for the opportunity|"
    r"artists to\s+produce|scanning, uploading, and distribution|"
    r"seller\.\s+He wrote the book|enrich our culture)",
    re.IGNORECASE,
)


def is_overview_question(question: str) -> bool:
    return bool(_OVERVIEW_RE.search(question))


def is_thematic_question(question: str) -> bool:
    return bool(_THEMATIC_RE.search(question))


def retrieval_top_k(question: str, default: int) -> int:
    if is_overview_question(question):
        return max(default, 8)
    if is_thematic_question(question):
        return max(default, 6)
    return max(default, 5)


def expand_retrieval_queries(question: str) -> List[str]:
    """Extra search strings for multi-query merge (esp. unread-book overviews)."""
    queries = [question.strip()]
    if is_overview_question(question):
        queries.extend(
            [
                "introduction prologue summary main narrative story",
                "table of contents outline what happened",
                "idealism greed lies cryptocurrency craze beginning",
            ]
        )
        if re.search(r"[\u4e00-\u9fff]", question):
            queries.append("introduction chapter 1 summary events characters timeline")
    elif is_thematic_question(question):
        queries.extend(
            [
                "idealism greed libertarian cypherpunk Ethereum community",
                "ideals versus profit motives cryptocurrency",
                "different views titles hierarchy professional Geth",
            ]
        )
        if re.search(r"[\u4e00-\u9fff]", question):
            queries.extend(
                [
                    "idealism greed lies",
                    "Ethereum community beliefs",
                ]
            )
    seen: set[str] = set()
    out: List[str] = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


def is_boilerplate_chunk(text: str) -> bool:
    if not text or len(text.strip()) < 40:
        return True
    if _BOILERPLATE_RE.search(text):
        return True
    # PDF 轉檔常見破碎短行
    letters = sum(1 for c in text if c.isalpha())
    if letters < 80 and "。" not in text and text.count(".") < 2:
        if _BOILERPLATE_RE.search(text[:200]):
            return True
    return False


def filter_boilerplate_docs(docs: List[Dict]) -> List[Dict]:
    return [d for d in docs if not is_boilerplate_chunk(d.get("text") or "")]


def merge_docs_by_best_score(doc_lists: List[List[Dict]]) -> List[Dict]:
    merged: Dict[str, Dict] = {}
    for docs in doc_lists:
        for doc in docs:
            key = str(doc.get("point_id") or f"{doc.get('book_id')}:{doc.get('chunk_id')}")
            prev = merged.get(key)
            if prev is None or doc.get("score", 0) > prev.get("score", 0):
                merged[key] = doc
    return sorted(merged.values(), key=lambda d: d.get("score", 0), reverse=True)
