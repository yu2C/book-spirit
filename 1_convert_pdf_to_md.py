"""
第一步：PDF → Markdown 轉換
使用 MarkItDown（微軟官方）轉換 PDF 為結構化 Markdown

安裝：pip install markitdown
"""

import sys
import os
from pathlib import Path
from markitdown import MarkItDown

def convert_pdf_to_markdown(pdf_path: str, output_dir: str = "outputs") -> str:
    """
    轉換 PDF 為 Markdown
    
    Args:
        pdf_path: PDF 檔案路徑
        output_dir: 輸出目錄
        
    Returns:
        輸出的 Markdown 檔案路徑
    """
    # 建立輸出目錄
    Path(output_dir).mkdir(exist_ok=True)
    
    # 生成輸出檔名
    pdf_name = Path(pdf_path).stem
    output_path = Path(output_dir) / f"{pdf_name}.md"
    
    print(f"🔄 開始轉換: {pdf_path}")
    print(f"📝 輸出位置: {output_path}")
    
    try:
        # 初始化 MarkItDown
        md = MarkItDown()
        
        # 轉換 PDF
        result = md.convert(pdf_path)
        
        # 寫入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result.text_content)
        
        # 統計
        lines = result.text_content.split('\n')
        chars = len(result.text_content)
        
        print(f"✅ 轉換成功！")
        print(f"   - 行數: {len(lines)}")
        print(f"   - 字數: {chars}")
        print(f"💾 已保存到: {output_path}")
        
        return str(output_path)
        
    except FileNotFoundError:
        print(f"❌ 找不到檔案: {pdf_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 轉換失敗: {e}")
        sys.exit(1)

def preview_markdown(md_path: str, max_lines: int = 50):
    """預覽 Markdown 前幾行"""
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"\n📖 預覽前 {max_lines} 行:")
    print("=" * 80)
    for i, line in enumerate(lines[:max_lines]):
        print(line.rstrip())
        if i > max_lines:
            break
    print("=" * 80)

if __name__ == "__main__":
    # 使用方法
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        # 預設找 sample_books 目錄下的第一個 PDF
        sample_dir = Path("sample_books")
        pdf_files = list(sample_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("❌ 使用方法:")
            print("   python 1_convert_pdf_to_md.py <PDF路徑>")
            print("\n   或在 sample_books/ 目錄放 PDF 檔案，直接執行本腳本")
            sys.exit(1)
        
        pdf_file = str(pdf_files[0])
        print(f"📚 找到 PDF: {pdf_file}")
    
    # 執行轉換
    md_path = convert_pdf_to_markdown(pdf_file)
    
    # 預覽結果
    preview_markdown(md_path)
