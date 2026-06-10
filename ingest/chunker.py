"""
第二步：Markdown → 分塊 + 提取元數據
實驗不同的分塊大小，提取章節結構
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64

CHAPTER_PART_RE = re.compile(r"^第[一二三四五六七八九十百千\d]+部分\b")
CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百千\d]+章\b")
HEADING_MISC_RE = re.compile(
    r"^(推荐序|序　|关于本书|附录|致谢|额外推荐|埃里克的笔记|纳瓦尔亲述|进一步了解纳瓦尔)"
)


@dataclass
class Chunk:
    """分塊單位"""

    text: str
    page: int = 0
    heading: str = ""
    chapter: str = ""
    chunk_id: int = 0


_BACK_MATTER_MARKERS = (
    "\nAcknowledgments\n",
    "\nAcknowledgements\n",
    "\nAbout the Author\n",
    "\nAbout the author\n",
    "\nNotes\n",
    "\nBibliography\n",
    "\nIndex\n",
    "\nDiscover More\n",
)


def normalize_md_text(md_text: str) -> str:
    """清理 PDF 轉換殘留；去掉重複目錄與英文書常見後記。"""
    md_text = md_text.replace("\f", "")
    marker = "\nTable of Contents\n"
    idx = md_text.find(marker)
    if idx > len(md_text) * 0.5:
        md_text = md_text[:idx]
    # 後記、致謝多在全書後段，避免被「大綱類」問題檢索到
    cut_at = len(md_text)
    for marker in _BACK_MATTER_MARKERS:
        pos = md_text.find(marker)
        if pos > len(md_text) * 0.55:
            cut_at = min(cut_at, pos)
    if cut_at < len(md_text):
        md_text = md_text[:cut_at]
    return md_text


def parse_line_structure(
    line: str, allow_short_headings: bool = True
) -> Tuple[str | None, str | None]:
    """
    解析單行是否為章節/小節標題。
    Returns:
        (chapter, heading) — 僅非 None 的欄位需要更新
    """
    stripped = line.strip()
    if not stripped:
        return None, None

    if line.startswith("# "):
        return line.replace("# ", "").strip(), None
    if line.startswith("## "):
        return None, line.replace("## ", "").strip()
    if CHAPTER_PART_RE.match(stripped) or CHAPTER_RE.match(stripped):
        return stripped, ""
    if HEADING_MISC_RE.match(stripped):
        return None, stripped

    # 短行標題（正文才啟用，避免目錄區污染章節標籤）
    if re.search(r"\[\d+\]", stripped) or "。" in stripped or "，" in stripped:
        return None, None

    if allow_short_headings and (
        len(stripped) <= 20
        and len(re.findall(r"[\u4e00-\u9fff]", stripped)) >= 2
        and not re.search(r"[。！？.，,；;：:]$", stripped)
        and not stripped[0].isdigit()
        and not stripped.startswith("·")
        and "http" not in stripped
    ):
        return None, stripped

    return None, None


def extract_structure(md_text: str) -> tuple[List[str], List[str]]:
    """
    提取 Markdown 結構：標題和章節

    Returns:
        (章節列表, 標題列表)
    """
    md_text = normalize_md_text(md_text)
    lines = md_text.split("\n")
    chapters = []
    headings = []

    body_started = False
    for line in lines:
        stripped = line.strip()
        if (
            stripped == "背景"
            or stripped.startswith("#")
            or CHAPTER_PART_RE.match(stripped)
            or CHAPTER_RE.match(stripped)
        ):
            body_started = True
        chapter_update, heading_update = parse_line_structure(
            line, allow_short_headings=body_started
        )
        if not body_started:
            chapter_update, heading_update = None, None
        if chapter_update is not None:
            chapters.append(chapter_update)
        if heading_update:
            headings.append(heading_update)

    return chapters, headings


def _chunk_markdown_native(md_text: str, chunk_size: int = 512, overlap: int = 64) -> List[Chunk]:
    """自研分塊：字數上限 + 句號斷句 + 章節 metadata。"""
    md_text = normalize_md_text(md_text)
    lines = md_text.split("\n")
    chunks = []
    current_text = ""
    current_heading = ""
    current_chapter = ""
    chunk_id = 0
    page = 0

    body_started = False
    for line in lines:
        stripped = line.strip()
        if (
            stripped == "背景"
            or stripped.startswith("#")
            or CHAPTER_PART_RE.match(stripped)
            or CHAPTER_RE.match(stripped)
        ):
            body_started = True
        chapter_update, heading_update = parse_line_structure(
            line, allow_short_headings=body_started
        )
        if not body_started:
            chapter_update, heading_update = None, None
        if chapter_update is not None:
            current_chapter = chapter_update
            if heading_update == "":
                current_heading = ""
        if heading_update is not None and heading_update != "":
            current_heading = heading_update

        # 累積文本
        current_text += line + "\n"

        # 檢查是否達到分塊大小
        if len(current_text) >= chunk_size:
            # 創建分塊（不要在中間斷句）
            chunk_text = current_text[:chunk_size]

            # 找到最後一個句子的結尾
            last_period = max(
                chunk_text.rfind("。"),
                chunk_text.rfind("！"),
                chunk_text.rfind("？"),
                chunk_text.rfind(".\n"),
            )

            if last_period > 0:
                chunk_text = chunk_text[: last_period + 1]

            chunks.append(
                Chunk(
                    text=chunk_text.strip(),
                    page=page,
                    heading=current_heading,
                    chapter=current_chapter,
                    chunk_id=chunk_id,
                )
            )

            # 重疊部分
            current_text = (
                current_text[len(chunk_text) - overlap :] if len(chunk_text) > overlap else ""
            )
            chunk_id += 1

            # 簡單的頁數估算（假設每 2000 字 ~ 1 頁）
            page = chunk_id // (2000 // chunk_size) + 1

    # 最後的分塊
    if current_text.strip():
        chunks.append(
            Chunk(
                text=current_text.strip(),
                page=page,
                heading=current_heading,
                chapter=current_chapter,
                chunk_id=chunk_id,
            )
        )

    return chunks


def chunk_markdown(md_text: str, chunk_size: int = 512, overlap: int = 64) -> List[Chunk]:
    """
    分塊 Markdown。CHUNKER_MODE 見 core/config.py：
    - semantic（預設，BGE 語意切分）
    - native（固定字數 + 章節 regex）
    """
    from core.config import CHUNKER_MODE, CHUNKER_SEMANTIC, SUPPORTED_CHUNKER_MODES

    if CHUNKER_MODE not in SUPPORTED_CHUNKER_MODES:
        raise ValueError(f"不支援 CHUNKER_MODE={CHUNKER_MODE}，可用: {SUPPORTED_CHUNKER_MODES}")

    if CHUNKER_MODE == CHUNKER_SEMANTIC:
        from ingest.chunker_semantic import chunk_markdown_semantic

        return chunk_markdown_semantic(md_text, chunk_size=chunk_size, overlap=overlap)
    return _chunk_markdown_native(md_text, chunk_size=chunk_size, overlap=overlap)
