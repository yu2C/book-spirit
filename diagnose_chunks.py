"""
診斷工具：檢查分塊是否切好

這個工具會幫你回答：
- 分塊邊界在哪裡？
- 是否在邏輯邊界切割？
- 每個分塊包含多少概念？
"""

from pathlib import Path
from typing import List, Dict
import re

class ChunkDiagnostics:
    """分塊診斷工具"""
    
    @staticmethod
    def analyze_chunk_boundaries(chunks: List[str], context_chars: int = 50):
        """
        分析分塊邊界，看是否在正確的地方切割
        
        正確的地方：句子結束（。！？\n）
        錯誤的地方：單詞中間、句子中間
        """
        print("\n" + "="*80)
        print("🔍 分塊邊界診斷")
        print("="*80)
        
        good_boundaries = 0
        bad_boundaries = 0
        
        for i in range(len(chunks) - 1):
            chunk1_end = chunks[i][-context_chars:]
            chunk2_start = chunks[i+1][:context_chars]
            
            # 檢查邊界是否在句子結尾
            is_sentence_boundary = (
                chunk1_end.rstrip()[-1] in '。！？\n' if chunk1_end.rstrip() else False
            )
            
            status = "✅" if is_sentence_boundary else "❌"
            
            if is_sentence_boundary:
                good_boundaries += 1
            else:
                bad_boundaries += 1
            
            # 只顯示前 5 個邊界
            if i < 5:
                print(f"\n[邊界 {i+1}] {status}")
                print(f"  上一個 chunk 結尾: ...{chunk1_end}")
                print(f"  下一個 chunk 開頭: {chunk2_start}...")
        
        total = good_boundaries + bad_boundaries
        pct = good_boundaries / total * 100 if total > 0 else 0
        
        print(f"\n📊 邊界品質統計:")
        print(f"   好的邊界（句子末尾）: {good_boundaries}/{total} ({pct:.0f}%)")
        print(f"   壞的邊界（句子中間）: {bad_boundaries}/{total}")
        
        if pct >= 80:
            print(f"   ✅ 邊界切割很好")
        elif pct >= 50:
            print(f"   △ 邊界切割有改進空間")
        else:
            print(f"   ❌ 邊界切割很差，需要改進")
        
        return good_boundaries / total if total > 0 else 0
    
    @staticmethod
    def analyze_chunk_sizes(chunks: List[str]):
        """
        分析分塊大小的分佈
        
        理想：均勻分佈，沒有極端大小
        """
        print("\n" + "="*80)
        print("📏 分塊大小分析")
        print("="*80)
        
        sizes = [len(c) for c in chunks]
        
        avg = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        
        # 計算標準差
        variance = sum((s - avg) ** 2 for s in sizes) / len(sizes)
        std_dev = variance ** 0.5
        
        print(f"\n📊 大小統計:")
        print(f"   平均: {avg:.0f} 字")
        print(f"   最小: {min_size} 字")
        print(f"   最大: {max_size} 字")
        print(f"   標準差: {std_dev:.0f}")
        
        # 分布直方圖
        bins = [0, 200, 400, 600, 800, 1000, 2000]
        bin_names = ["<200", "200-400", "400-600", "600-800", "800-1000", ">1000"]
        
        print(f"\n📈 大小分佈:")
        for i in range(len(bins) - 1):
            count = sum(1 for s in sizes if bins[i] <= s < bins[i+1])
            pct = count / len(sizes) * 100
            bar = "█" * int(pct / 5)
            print(f"   {bin_names[i]:>10}: {bar} {pct:.0f}%")
        
        # 判斷
        if std_dev < avg * 0.3:
            print(f"\n✅ 大小分佈很均勻（標準差小）")
            return "good"
        elif std_dev < avg * 0.5:
            print(f"\n△ 大小分佈可接受")
            return "ok"
        else:
            print(f"\n❌ 大小分佈不均勻（有極端值）")
            return "bad"
    
    @staticmethod
    def analyze_concept_distribution(chunks: List[str]):
        """
        分析每個分塊中的「概念數量」
        
        簡化判斷：看獨特詞彙數量
        """
        print("\n" + "="*80)
        print("💡 概念分佈分析")
        print("="*80)
        
        print("\n📝 分析思路:")
        print("   一個好的分塊應該圍繞 1-2 個核心概念")
        print("   太多概念混在一起 → 搜尋時污染")
        print("\n   簡化檢查：看是否有多個「段落」（空行分隔）")
        
        multi_paragraph = 0
        single_paragraph = 0
        
        for i, chunk in enumerate(chunks[:10]):  # 只檢查前 10 個
            paragraphs = len([p for p in chunk.split('\n\n') if p.strip()])
            
            if paragraphs > 1:
                multi_paragraph += 1
                status = "△ 多段落"
            else:
                single_paragraph += 1
                status = "✅ 單段落"
            
            print(f"\n[Chunk {i+1}] {status} ({paragraphs} 段)")
            preview = chunk[:60].replace('\n', ' ')
            print(f"   {preview}...")
        
        pct_single = single_paragraph / (multi_paragraph + single_paragraph) * 100
        print(f"\n📊 統計 (前 10 個):")
        print(f"   單段落: {single_paragraph}/{single_paragraph + multi_paragraph} ({pct_single:.0f}%)")
        print(f"\n   ✅ 如果 > 70% 單段落，說明分塊很乾淨")
    
    @staticmethod
    def sample_chunk_preview(chunks: List[str], num_samples: int = 5):
        """
        隨機顯示幾個分塊的完整內容
        """
        print("\n" + "="*80)
        print("👀 分塊樣本預覽")
        print("="*80)
        
        import random
        
        indices = random.sample(range(len(chunks)), min(num_samples, len(chunks)))
        
        for idx in indices:
            chunk = chunks[idx]
            print(f"\n[Chunk {idx}] ({len(chunk)} 字)")
            print("-" * 80)
            print(chunk)
            print("-" * 80)

def load_chunks_from_markdown(md_path: str, chunk_size: int = 512) -> List[str]:
    """從 Markdown 重新生成分塊"""
    
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    lines = md_text.split('\n')
    chunks = []
    current_text = ""
    
    for line in lines:
        current_text += line + '\n'
        
        if len(current_text) >= chunk_size:
            chunk_text = current_text[:chunk_size]
            
            # 找最後一個句子邊界
            last_period = max(
                chunk_text.rfind('。'),
                chunk_text.rfind('！'),
                chunk_text.rfind('？'),
            )
            
            if last_period > 0:
                chunk_text = chunk_text[:last_period + 1]
            
            chunks.append(chunk_text.strip())
            current_text = current_text[len(chunk_text):]
    
    if current_text.strip():
        chunks.append(current_text.strip())
    
    return chunks

if __name__ == "__main__":
    # 讀取 Markdown
    md_dir = Path("outputs")
    md_files = list(md_dir.glob("*.md"))
    
    if not md_files:
        print("❌ 找不到 Markdown 檔案")
        exit(1)
    
    md_path = md_files[0]
    print(f"📖 診斷: {md_path}")
    print(f"   使用 chunk_size=512")
    
    # 生成分塊
    chunks = load_chunks_from_markdown(str(md_path), chunk_size=512)
    print(f"\n✅ 生成 {len(chunks)} 個分塊")
    
    # 執行診斷
    print("\n" + "="*80)
    print("🔬 開始分塊診斷")
    print("="*80)
    
    # 1. 邊界分析
    boundary_score = ChunkDiagnostics.analyze_chunk_boundaries(chunks)
    
    # 2. 大小分析
    size_score = ChunkDiagnostics.analyze_chunk_sizes(chunks)
    
    # 3. 概念分佈
    ChunkDiagnostics.analyze_concept_distribution(chunks)
    
    # 4. 樣本預覽
    ChunkDiagnostics.sample_chunk_preview(chunks, num_samples=3)
    
    # 最終評分
    print("\n" + "="*80)
    print("🎯 最終診斷")
    print("="*80)
    
    if boundary_score >= 0.8:
        print(f"✅ 分塊邊界很好")
        print(f"   這會導致：搜尋到完整的邏輯單元，很少有「半句話」污染")
    elif boundary_score >= 0.5:
        print(f"△ 分塊邊界中等")
        print(f"   建議：改進分塊邏輯，確保在句子末尾切割")
    else:
        print(f"❌ 分塊邊界很差")
        print(f"   後果：搜尋會被碎片化的文本污染，最終 LLM 生成質量下降")
    
    print(f"\n💡 記住：好的搜尋品質 = 好的分塊品質")
    print(f"   如果後續 LLM 回答不好，先檢查分塊，不是檢查 LLM")
