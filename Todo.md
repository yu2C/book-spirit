# RAG 修復 Todo

- [x] 1. 修復 Qdrant 向量維度（384 → 依模型動態設定）
- [x] 2. 更新測試問題為《納瓦爾寶典》主題
- [x] 3. 步驟 3 共用步驟 2 的分塊邏輯（含 overlap）
- [x] 4. 支援無 `#` 標題的 MD 章節偵測

## 搜尋品質工具修復

- [x] 3_build_qdrant.py 改為持久化儲存 + BGE query 前綴
- [x] 4_test_search_quality.py 實際連接 Qdrant 搜尋評估
- [x] 測試問題改為《納瓦爾寶典》
- [x] 修正錯誤的 heading 標籤（∨、[78] 碎片）
