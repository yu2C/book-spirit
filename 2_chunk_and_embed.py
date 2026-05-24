"""
第二步：Markdown → 分塊 + 提取元數據
實驗不同的分塊大小，提取章節結構

安裝：pip install langchain
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64

CHAPTER_PART_RE = re.compile(r'^第[一二三四五六七八九十百千\d]+部分\b')
CHAPTER_RE = re.compile(r'^第[一二三四五六七八九十百千\d]+章\b')
HEADING_MISC_RE = re.compile(
    r'^(推荐序|序　|关于本书|附录|致谢|额外推荐|埃里克的笔记|纳瓦尔亲述|进一步了解纳瓦尔)'
)

@dataclass
class Chunk:
    """分塊單位"""
    text: str
    page: int = 0
    heading: str = ""
    chapter: str = ""
    chunk_id: int = 0

def normalize_md_text(md_text: str) -> str:
    """清理 PDF 轉換殘留，並移除末尾重複目錄。"""
    md_text = md_text.replace('\f', '')
    marker = '\nTable of Contents\n'
    idx = md_text.find(marker)
    if idx > len(md_text) * 0.5:
        md_text = md_text[:idx]
    return md_text

def parse_line_structure(line: str, allow_short_headings: bool = True) -> Tuple[str | None, str | None]:
    """
    解析單行是否為章節/小節標題。
    Returns:
        (chapter, heading) — 僅非 None 的欄位需要更新
    """
    stripped = line.strip()
    if not stripped:
        return None, None

    if line.startswith('# '):
        return line.replace('# ', '').strip(), None
    if line.startswith('## '):
        return None, line.replace('## ', '').strip()
    if CHAPTER_PART_RE.match(stripped) or CHAPTER_RE.match(stripped):
        return stripped, ""
    if HEADING_MISC_RE.match(stripped):
        return None, stripped

    # 短行標題（正文才啟用，避免目錄區污染章節標籤）
    if (
        re.search(r'\[\d+\]', stripped)
        or '。' in stripped
        or '，' in stripped
    ):
        return None, None

    if allow_short_headings and (
        len(stripped) <= 20
        and len(re.findall(r'[\u4e00-\u9fff]', stripped)) >= 2
        and not re.search(r'[。！？.，,；;：:]$', stripped)
        and not stripped[0].isdigit()
        and not stripped.startswith('·')
        and 'http' not in stripped
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
    lines = md_text.split('\n')
    chapters = []
    headings = []
    
    body_started = False
    for line in lines:
        if line.strip() == '背景':
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

def chunk_markdown(md_text: str, chunk_size: int = 512, overlap: int = 64) -> List[Chunk]:
    """
    分塊 Markdown 文本
    
    Args:
        md_text: Markdown 文本
        chunk_size: 每個分塊的字符數
        overlap: 分塊之間的重疊字符數
        
    Returns:
        Chunk 列表
    """
    md_text = normalize_md_text(md_text)
    lines = md_text.split('\n')
    chunks = []
    current_text = ""
    current_heading = ""
    current_chapter = ""
    chunk_id = 0
    page = 0
    
    body_started = False
    for line in lines:
        if line.strip() == '背景':
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
        current_text += line + '\n'
        
        # 檢查是否達到分塊大小
        if len(current_text) >= chunk_size:
            # 創建分塊（不要在中間斷句）
            chunk_text = current_text[:chunk_size]
            
            # 找到最後一個句子的結尾
            last_period = max(
                chunk_text.rfind('。'),
                chunk_text.rfind('！'),
                chunk_text.rfind('？'),
                chunk_text.rfind('.\n'),
            )
            
            if last_period > 0:
                chunk_text = chunk_text[:last_period + 1]
            
            chunks.append(Chunk(
                text=chunk_text.strip(),
                page=page,
                heading=current_heading,
                chapter=current_chapter,
                chunk_id=chunk_id
            ))
            
            # 重疊部分
            current_text = current_text[len(chunk_text) - overlap:] if len(chunk_text) > overlap else ""
            chunk_id += 1
            
            # 簡單的頁數估算（假設每 2000 字 ~ 1 頁）
            page = chunk_id // (2000 // chunk_size) + 1
    
    # 最後的分塊
    if current_text.strip():
        chunks.append(Chunk(
            text=current_text.strip(),
            page=page,
            heading=current_heading,
            chapter=current_chapter,
            chunk_id=chunk_id
        ))
    
    return chunks

def test_different_chunk_sizes(md_text: str, chunk_sizes: List[int] = [256, 512, 1024]):
    """
    測試不同的分塊大小，比較效果
    """
    print("\n🔬 測試不同的分塊大小:")
    print("=" * 80)
    
    for size in chunk_sizes:
        chunks = chunk_markdown(md_text, chunk_size=size)
        avg_length = sum(len(c.text) for c in chunks) / len(chunks)
        
        print(f"\n📊 Chunk Size = {size}")
        print(f"   - 分塊數: {len(chunks)}")
        print(f"   - 平均長度: {avg_length:.0f}")
        print(f"   - 前 3 個分塊預覽:")
        
        for i, chunk in enumerate(chunks[:3]):
            preview = chunk.text[:100].replace('\n', ' ')
            print(f"     [{i}] [{chunk.chapter}] {preview}...")
    
    print("\n💡 建議: 根據搜尋品質選擇合適的大小")

if __name__ == "__main__":
    # 讀取轉換後的 Markdown
    md_dir = Path("outputs")
    md_files = list(md_dir.glob("*.md"))
    
    if not md_files:
        print("❌ 找不到 Markdown 檔案")
        print("   請先執行: python 1_convert_pdf_to_md.py")
        exit(1)
    
    md_path = md_files[0]
    print(f"📖 讀取: {md_path}")
    
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    # 提取結構
    chapters, headings = extract_structure(md_text)
    print(f"\n📚 結構分析:")
    print(f"   - 章數: {len(chapters)}")
    print(f"   - 小節數: {len(headings)}")
    if chapters:
        print(f"   - 第一章: {chapters[0]}")
    
    # 測試分塊
    test_different_chunk_sizes(md_text)
    
    # 最推薦的分塊
    print("\n✅ 使用推薦的分塊大小: 512")
    chunks = chunk_markdown(md_text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP)
    print(f"   總分塊數: {len(chunks)}")
    
    # 保存分塊信息
    chunks_path = md_dir / f"{md_path.stem}_chunks_info.txt"
    with open(chunks_path, 'w', encoding='utf-8') as f:
        f.write("Chunk ID | 章節 | 標題 | 文本長度 | 預覽\n")
        f.write("=" * 100 + "\n")
        for chunk in chunks:
            preview = chunk.text[:50].replace('\n', ' ')
            f.write(f"{chunk.chunk_id:3d} | {chunk.chapter:10s} | {chunk.heading:10s} | {len(chunk.text):4d} | {preview}\n")
    
    print(f"💾 分塊信息已保存到: {chunks_path}")
