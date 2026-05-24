"""
第五步：LLM 推理 + 回答生成
把搜尋結果餵給 Ollama，生成最終回答

硬體要求（M4 Air 24GB 滿足）：
- Qwen2.5:7b-q4 需要 8-10GB VRAM
- 加上 Embedding + Qdrant，總共 12-15GB
- M4 Air 24GB 很舒適

安裝步驟：
1. brew install ollama
2. ollama pull qwen2.5:7b-q4
3. ollama serve （在另一個終端執行）
4. python 5_generate_answer_with_llm.py
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import requests

# ============================================================================
# Ollama 客戶端
# ============================================================================

class OllamaClient:
    """簡單的 Ollama 客戶端"""
    
    def __init__(self, model: str = "qwen2.5:7b-instruct-q4_K_M", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/generate"
    
    def health_check(self) -> bool:
        """檢查 Ollama 服務是否運行"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """
        調用 Ollama 生成回答
        
        Args:
            prompt: 提示詞
            stream: 是否流式輸出
            
        Returns:
            生成的文本
        """
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        try:
            response = requests.post(self.endpoint, json=payload, timeout=60)
            response.raise_for_status()
            
            if stream:
                # 流式模式
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        full_response += data.get("response", "")
                        if data.get("done", False):
                            break
                return full_response
            else:
                # 非流式模式
                data = response.json()
                return data.get("response", "")
                
        except requests.exceptions.ConnectionError:
            return "❌ 無法連接到 Ollama。請確保執行了：ollama serve"
        except Exception as e:
            return f"❌ 錯誤: {str(e)}"

# ============================================================================
# RAG 完整流程
# ============================================================================

class RAGPipeline:
    """完整的 RAG 管道"""
    
    def __init__(self, 
                 qdrant_path: str = "./qdrant_storage",
                 embedding_model: str = "BAAI/bge-small-zh-v1.5",
                 ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"):
        
        print("🔄 初始化 RAG 管道...")
        
        # 1. 初始化 Embedding
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✅ Embedding 模型: {embedding_model}")
        
        # 2. 連接 Qdrant
        self.client = QdrantClient(path=qdrant_path)
        print(f"✅ Qdrant: {qdrant_path}")
        
        # 3. 初始化 Ollama
        self.ollama = OllamaClient(model=ollama_model)
        
        # 檢查 Ollama
        if self.ollama.health_check():
            print(f"✅ Ollama: {ollama_model}")
        else:
            print(f"⚠️  Ollama 未運行。請執行：ollama serve")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        向量搜尋
        """
        # 編碼查詢
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # 搜尋
        results = self.client.query_points(
            collection_name="books",
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        
        # 格式化結果
        retrieved = []
        for result in results.points:
            retrieved.append({
                'text': result.payload.get('text', ''),
                'score': result.score,
                'chapter': result.payload.get('chapter', ''),
                'heading': result.payload.get('heading', ''),
                'page': result.payload.get('page', 0),
                'book_title': result.payload.get('book_title', ''),
            })
        
        return retrieved
    
    def build_prompt(self, query: str, context: List[Dict]) -> str:
        """
        構造提示詞
        
        這是 RAG 成敗的關鍵
        """
        
        # 格式化上下文
        context_text = ""
        for i, doc in enumerate(context, 1):
            context_text += f"\n[來源 {i}] {doc['book_title']} - {doc['chapter']} - {doc['heading']}\n"
            context_text += f"{doc['text']}\n"
        
        # 系統提示詞（中文優化）
        system_prompt = """你是一個知識助手，基於提供的文本內容回答問題。

回答規則：
1. 只基於提供的文本內容回答
2. 如果文本中沒有相關信息，直接說「文本中沒有相關信息」
3. 在回答中引用具體的文本段落
4. 用清晰的邏輯組織回答
5. 用中文回答"""

        # 構造完整提示詞
        prompt = f"""{system_prompt}

提供的文本內容：
{context_text}

問題：{query}

回答："""
        
        return prompt
    
    def generate(self, query: str, top_k: int = 3) -> Dict:
        """
        完整的 RAG 流程：檢索 + 生成
        """
        
        start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"❓ 問題：{query}")
        print(f"{'='*80}")
        
        # 1. 檢索
        print(f"\n🔍 搜尋中...")
        retrieved = self.retrieve(query, top_k=top_k)
        
        if not retrieved:
            return {
                'question': query,
                'answer': '❌ 未找到相關內容',
                'sources': [],
                'time': 0
            }
        
        # 顯示檢索結果
        print(f"✅ 找到 {len(retrieved)} 個相關段落")
        for i, doc in enumerate(retrieved, 1):
            print(f"\n  [{i}] {doc['book_title']} (相似度: {doc['score']:.3f})")
            print(f"      {doc['chapter']} → {doc['heading']}")
            preview = doc['text'][:80].replace('\n', ' ')
            print(f"      {preview}...")
        
        # 2. 生成提示詞
        prompt = self.build_prompt(query, retrieved)
        
        # 3. LLM 生成
        print(f"\n🤖 LLM 生成中（{self.ollama.model}）...")
        generate_start = time.time()
        answer = self.ollama.generate(prompt, stream=False)
        generate_time = time.time() - generate_start
        
        print(f"✅ 生成完成（{generate_time:.2f}s）")
        
        # 4. 整理結果
        result = {
            'question': query,
            'answer': answer,
            'sources': [
                {
                    'title': doc['book_title'],
                    'chapter': doc['chapter'],
                    'heading': doc['heading'],
                    'page': doc['page'],
                    'score': doc['score'],
                    'text_preview': doc['text'][:100]
                }
                for doc in retrieved
            ],
            'time': time.time() - start_time,
            'generate_time': generate_time
        }
        
        return result
    
    def print_answer(self, result: Dict):
        """格式化打印回答"""
        
        print(f"\n{'='*80}")
        print(f"📝 回答")
        print(f"{'='*80}\n")
        print(result['answer'])
        
        print(f"\n{'='*80}")
        print(f"📚 引用來源（{len(result['sources'])} 個）")
        print(f"{'='*80}\n")
        
        for i, source in enumerate(result['sources'], 1):
            print(f"[{i}] {source['title']}")
            print(f"    {source['chapter']} → {source['heading']} (第 {source['page']} 頁)")
            print(f"    相似度: {source['score']:.3f}")
            print(f"    預覽: {source['text_preview']}...\n")
        
        print(f"⏱️  耗時: {result['time']:.2f}s (LLM: {result['generate_time']:.2f}s)")

# ============================================================================
# 交互式 QA
# ============================================================================

def interactive_qa():
    """交互式問答"""
    
    # 初始化
    rag = RAGPipeline(
        qdrant_path="./qdrant_storage",
        embedding_model="BAAI/bge-small-zh-v1.5",
        ollama_model="qwen2.5:7b-instruct-q4_K_M"
    )
    
    # 檢查 Ollama
    if not rag.ollama.health_check():
        print("\n❌ Ollama 未運行！")
        print("\n請在新的終端執行：")
        print("  ollama serve")
        print("\n然後返回這個終端重試。")
        return
    
    print(f"\n{'='*80}")
    print(f"📚 書籍知識助手")
    print(f"{'='*80}")
    print(f"\n輸入 'quit' 或 'exit' 結束\n")
    
    # 交互循環
    while True:
        question = input("\n❓ 你的問題：").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 再見！")
            break
        
        if not question:
            print("請輸入問題")
            continue
        
        # 生成回答
        result = rag.generate(question)
        rag.print_answer(result)

if __name__ == "__main__":
    import sys
    
    # 檢查命令行參數
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Demo 模式：預設問題
        rag = RAGPipeline(
            qdrant_path="./qdrant_storage",
            embedding_model="BAAI/bge-small-zh-v1.5",
            ollama_model="qwen2.5:7b-instruct-q4_K_M"
        )
        
        if not rag.ollama.health_check():
            print("❌ Ollama 未運行！請執行：ollama serve")
            sys.exit(1)
        
        demo_questions = [
            "什么是专长？",
            "如何不靠运气致富？",
            "幸福是一种可以学习的技能吗？",
        ]
        
        for question in demo_questions:
            result = rag.generate(question, top_k=3)
            rag.print_answer(result)
            print("\n" + "="*80 + "\n")
    
    else:
        # 交互模式
        interactive_qa()
