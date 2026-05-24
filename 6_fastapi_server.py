"""
第六步：FastAPI 服務
把 RAG 系統部署為 REST API

運行方法：
  python 6_fastapi_server.py
  
然後訪問：
  http://127.0.0.1:8000/docs  (Swagger 文檔)
  http://127.0.0.1:8000/ask   (API 端點)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import json
from typing import List, Dict, Optional
from datetime import datetime
import logging

# 導入 RAG 系統
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import requests

# ============================================================================
# 日誌設定
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 數據模型
# ============================================================================

class AskRequest(BaseModel):
    """提問請求"""
    question: str
    top_k: int = 3
    temperature: float = 0.7

class Source(BaseModel):
    """引用來源"""
    title: str
    chapter: str
    heading: str
    page: int
    score: float
    text_preview: str

class AskResponse(BaseModel):
    """回答"""
    question: str
    answer: str
    sources: List[Source]
    time_elapsed: float
    llm_time: float

class HealthResponse(BaseModel):
    """健康檢查"""
    status: str
    ollama_available: bool
    qdrant_available: bool
    message: str

# ============================================================================
# RAG 系統
# ============================================================================

class RAGSystem:
    """封裝 RAG 邏輯"""
    
    def __init__(self, 
                 qdrant_path: str = "./qdrant_storage",
                 embedding_model: str = "BAAI/bge-small-zh-v1.5",
                 ollama_model: str = "qwen2.5:7b-instruct-q4_K_M",
                 ollama_url: str = "http://localhost:11434"):
        
        logger.info("初始化 RAG 系統...")
        
        # Embedding 模型
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"✅ Embedding 模型載入: {embedding_model}")
        except Exception as e:
            logger.error(f"❌ Embedding 模型載入失敗: {e}")
            raise
        
        # Qdrant
        try:
            self.qdrant_client = QdrantClient(path=qdrant_path)
            logger.info(f"✅ Qdrant 連接: {qdrant_path}")
        except Exception as e:
            logger.error(f"❌ Qdrant 連接失敗: {e}")
            raise
        
        # Ollama
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.ollama_endpoint = f"{ollama_url}/api/generate"
    
    def check_ollama_health(self) -> bool:
        """檢查 Ollama 是否運行"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """向量搜尋"""
        try:
            # 編碼問題
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # 搜尋
            results = self.qdrant_client.query_points(
                collection_name="books",
                query_vector=query_embedding,
                limit=top_k,
            )
            
            # 格式化
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
        
        except Exception as e:
            logger.error(f"搜尋失敗: {e}")
            return []
    
    def build_prompt(self, query: str, context: List[Dict]) -> str:
        """構造提示詞"""
        context_text = ""
        for i, doc in enumerate(context, 1):
            context_text += f"\n[來源 {i}] {doc['book_title']} - {doc['chapter']}\n"
            context_text += f"{doc['text']}\n"
        
        system_prompt = """你是一個知識助手，基於提供的文本內容回答問題。

回答規則：
1. 只基於提供的文本內容回答
2. 如果文本中沒有相關信息，直接說「文本中沒有相關信息」
3. 在回答中引用具體的文本段落
4. 用清晰的邏輯組織回答
5. 用中文回答"""

        prompt = f"""{system_prompt}

提供的文本內容：
{context_text}

問題：{query}

回答："""
        
        return prompt
    
    def generate_with_ollama(self, prompt: str) -> str:
        """呼叫 Ollama 生成回答"""
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            
            response = requests.post(self.ollama_endpoint, json=payload, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "")
        
        except requests.exceptions.Timeout:
            logger.error("Ollama 請求超時")
            return "❌ 生成超時，請重試"
        except requests.exceptions.ConnectionError:
            logger.error("無法連接到 Ollama")
            return "❌ 無法連接到 Ollama。請確保執行了：ollama serve"
        except Exception as e:
            logger.error(f"Ollama 錯誤: {e}")
            return f"❌ 生成失敗: {str(e)}"
    
    def ask(self, question: str, top_k: int = 3) -> Dict:
        """完整問答流程"""
        start_time = time.time()
        
        # 1. 搜尋
        logger.info(f"搜尋: {question}")
        retrieved = self.retrieve(question, top_k=top_k)
        
        if not retrieved:
            return {
                'question': question,
                'answer': '❌ 未找到相關內容',
                'sources': [],
                'time_elapsed': time.time() - start_time,
                'llm_time': 0
            }
        
        logger.info(f"找到 {len(retrieved)} 個相關段落")
        
        # 2. 生成
        prompt = self.build_prompt(question, retrieved)
        
        logger.info("生成回答中...")
        generate_start = time.time()
        answer = self.generate_with_ollama(prompt)
        generate_time = time.time() - generate_start
        
        logger.info(f"生成完成 ({generate_time:.2f}s)")
        
        # 3. 整理結果
        result = {
            'question': question,
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
            'time_elapsed': time.time() - start_time,
            'llm_time': generate_time
        }
        
        return result

# ============================================================================
# FastAPI 應用
# ============================================================================

app = FastAPI(
    title="Book Spirit API",
    description="本地 RAG 書籍知識助手",
    version="1.0.0",
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 RAG 系統
logger.info("啟動 Book Spirit API...")
try:
    rag = RAGSystem()
    logger.info("✅ RAG 系統初始化成功")
except Exception as e:
    logger.error(f"❌ RAG 系統初始化失敗: {e}")
    rag = None

# ============================================================================
# 路由
# ============================================================================

@app.get("/", tags=["基本"])
async def root():
    """根路由"""
    return {
        "name": "Book Spirit API",
        "version": "1.0.0",
        "docs": "/docs",
        "ask_endpoint": "/ask"
    }

@app.get("/health", tags=["基本"])
async def health() -> HealthResponse:
    """健康檢查"""
    if rag is None:
        return HealthResponse(
            status="error",
            ollama_available=False,
            qdrant_available=False,
            message="RAG 系統初始化失敗"
        )
    
    ollama_ok = rag.check_ollama_health()
    qdrant_ok = True  # 如果能初始化，Qdrant 就 OK
    
    status = "ok" if (ollama_ok and qdrant_ok) else "degraded"
    
    message_parts = []
    if not ollama_ok:
        message_parts.append("Ollama 未運行")
    if not qdrant_ok:
        message_parts.append("Qdrant 不可用")
    
    message = ", ".join(message_parts) if message_parts else "所有服務正常"
    
    return HealthResponse(
        status=status,
        ollama_available=ollama_ok,
        qdrant_available=qdrant_ok,
        message=message
    )

@app.post("/ask", response_model=AskResponse, tags=["核心"])
async def ask(request: AskRequest):
    """提問端點"""
    
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG 系統未初始化")
    
    # 檢查 Ollama
    if not rag.check_ollama_health():
        raise HTTPException(
            status_code=503,
            detail="Ollama 未運行。請執行：ollama serve"
        )
    
    try:
        result = rag.ask(request.question, top_k=request.top_k)
        
        return AskResponse(
            question=result['question'],
            answer=result['answer'],
            sources=[
                Source(
                    title=s['title'],
                    chapter=s['chapter'],
                    heading=s['heading'],
                    page=s['page'],
                    score=s['score'],
                    text_preview=s['text_preview']
                )
                for s in result['sources']
            ],
            time_elapsed=result['time_elapsed'],
            llm_time=result['llm_time']
        )
    
    except Exception as e:
        logger.error(f"處理請求失敗: {e}")
        raise HTTPException(status_code=500, detail=f"處理失敗: {str(e)}")

@app.get("/info", tags=["基本"])
async def info():
    """系統信息"""
    if rag is None:
        return {"status": "error"}
    
    return {
        "system": "Book Spirit RAG",
        "version": "1.0.0",
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "llm_model": rag.ollama_model,
        "collection": "books",
        "features": [
            "向量搜尋",
            "LLM 生成",
            "來源引用",
            "中文優化"
        ]
    }

# ============================================================================
# 啟動
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
