"""
第四步：測試搜尋品質
連接 3_build_qdrant.py 建立的持久化索引，用客觀指標 + 人工評分評估分塊品質。

用法：
  python 4_test_search_quality.py              # 互動評分（預設 6 題）
  python 4_test_search_quality.py --preview    # 只顯示搜尋結果，不問評分
  python 4_test_search_quality.py --all        # 互動評分（全部題目）
  python 4_test_search_quality.py --demo       # 只顯示說明與評分標準
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# 載入步驟 3 的搜尋與索引工具
_build_spec = importlib.util.spec_from_file_location(
    "build_qdrant",
    Path(__file__).parent / "3_build_qdrant.py",
)
_build = importlib.util.module_from_spec(_build_spec)
_build_spec.loader.exec_module(_build)


class SearchQualityEvaluator:
    """搜尋品質評估器：人工判斷前 3 筆結果的相關性。"""

    def __init__(self):
        self.eval_results = []

    def evaluate_single_query(
        self,
        question: str,
        search_results: List[Dict],
    ) -> Dict:
        print(f"\n{'=' * 80}")
        print(f"❓ 問題: {question}")
        print(f"{'=' * 80}")

        ratings = []
        for rank, result in enumerate(search_results[:3], 1):
            text = result["text"][:150].replace("\n", " ")
            score = result["score"]
            chapter = result.get("chapter") or "(無)"
            heading = result.get("heading") or "(無)"

            print(f"\n[結果 {rank}] (相似度: {score:.3f})")
            print(f"   章節: {chapter} → {heading}")
            print(f"   內容: {text}...")

            rating = input("\n   👉 這個結果相關嗎？ (y/partial/n，Enter=跳過評分): ").strip().lower()
            if rating == "y":
                relevance = "relevant"
            elif rating == "partial":
                relevance = "partial"
            elif rating in ("n", "no"):
                relevance = "irrelevant"
            else:
                relevance = "skipped"

            ratings.append({
                "rank": rank,
                "score": score,
                "relevance": relevance,
                "text_preview": text,
            })

        rated = [r for r in ratings if r["relevance"] != "skipped"]
        relevant_count = sum(1 for r in rated if r["relevance"] == "relevant")
        partial_count = sum(1 for r in rated if r["relevance"] == "partial")

        if not rated:
            quality, quality_score = "skipped", None
        elif relevant_count >= 2:
            quality, quality_score = "good", 1.0
        elif relevant_count == 1 and partial_count >= 1:
            quality, quality_score = "partial", 0.5
        elif relevant_count >= 1:
            quality, quality_score = "partial", 0.5
        else:
            quality, quality_score = "bad", 0.0

        result_summary = {
            "question": question,
            "quality": quality,
            "quality_score": quality_score,
            "relevant_count": relevant_count,
            "partial_count": partial_count,
            "average_score": sum(r["score"] for r in ratings) / len(ratings),
            "detailed_ratings": ratings,
        }
        self.eval_results.append(result_summary)

        print(f"\n📊 結果統計:")
        print(f"   相關: {relevant_count}/3")
        print(f"   部分相關: {partial_count}/3")
        print(f"   平均相似度: {result_summary['average_score']:.3f}")
        if quality_score is not None:
            print(f"   評分: {quality.upper()}")

        return result_summary

    def print_summary(self):
        scored = [r for r in self.eval_results if r["quality_score"] is not None]
        if not scored:
            print("\n⚠️ 沒有完成評分的問題（全部跳過）")
            return None

        good_count = sum(1 for r in scored if r["quality"] == "good")
        partial_count = sum(1 for r in scored if r["quality"] == "partial")
        bad_count = sum(1 for r in scored if r["quality"] == "bad")
        total = len(scored)
        good_pct = good_count / total * 100

        print(f"\n{'=' * 80}")
        print("📊 總體評估結果")
        print(f"{'=' * 80}")
        print(f"\n測試問題數: {total}")
        print(f"✅ 好 (2+相關): {good_count} ({good_pct:.0f}%)")
        print(f"△ 中等: {partial_count}")
        print(f"❌ 差: {bad_count}")

        avg_quality = sum(r["quality_score"] for r in scored) / total
        avg_similarity = sum(r["average_score"] for r in scored) / total
        print(f"\n📈 指標:")
        print(f"   平均品質分數: {avg_quality:.2f} (0-1)")
        print(f"   平均相似度: {avg_similarity:.3f}")

        if good_pct >= 80:
            print("\n✅ 分塊品質良好！")
            verdict = "good"
        elif good_pct >= 50:
            print("\n△ 分塊品質中等，可試著調整 chunk_size")
            verdict = "partial"
        else:
            print("\n❌ 分塊品質不佳，請檢查分塊或 Embedding")
            verdict = "bad"
        return verdict

    def save_report(self, filename: str = "search_quality_report.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.eval_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 報告已保存: {filename}")


def prepare_test_queries() -> Dict[str, List[str]]:
    """《納瓦爾寶典》測試問題（依目前索引的書籍）。"""
    return {
        "簡單事實": [
            "什么是专长？",
            "财富和金钱有什么区别？",
            "幸福是一种技能吗？",
        ],
        "核心概念": [
            "如何不靠运气致富？",
            "如何找到自己的专长？",
            "杠杆有哪些类型？",
        ],
        "深度理解": [
            "为什么判断力比努力更重要？",
            "如何获得运气？",
            "冥想对幸福有什么帮助？",
        ],
        "邊界案例": [
            "这本书的作者是谁？",
            "纳瓦尔推荐的阅读方法是什么？",
        ],
    }


def flatten_queries(queries: Dict[str, List[str]], categories: List[str] | None = None) -> List[str]:
    items = []
    for category, questions in queries.items():
        if categories and category not in categories:
            continue
        for q in questions:
            items.append((category, q))
    return items


def print_criteria():
    print("\n📋 品質判斷標準:")
    print("=" * 80)
    print("""
相似度分數（BGE + query: 前綴）:
  • > 0.75: 高度相關
  • 0.65-0.75: 相關
  • 0.55-0.65: 邊界，需人工判斷
  • < 0.55: 可能無關

前 3 結果人工評分:
  • y: 直接回答問題
  • partial: 部分相關
  • n: 無關
""")


def run_demo():
    print("""
📚 搜尋品質評估工具

流程：
  1. 先執行 python 3_build_qdrant.py 建立索引
  2. 本腳本對每題搜尋前 3 筆結果
  3. 你判斷 y / partial / n
  4. 輸出統計與 search_quality_report.json
""")
    print_criteria()
    print("\n🎯 測試問題（《納瓦爾寶典》）:")
    for category, questions in prepare_test_queries().items():
        print(f"\n[{category}]")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")


def run_search_tests(
    preview: bool = False,
    all_questions: bool = False,
    mode: str = "vector",
    use_rerank: bool = False,
):
    if not _build.QDRANT_PATH.exists() or not _build.INDEX_META_FILE.exists():
        print("❌ 找不到 Qdrant 索引")
        print("   請先執行: python 3_build_qdrant.py")
        sys.exit(1)

    if mode == "hybrid" and not _build.BM25_CORPUS_FILE.exists():
        print("❌ hybrid 模式需要 BM25 語料")
        print("   請重新執行: python 3_build_qdrant.py")
        sys.exit(1)

    meta = _build.load_index_meta()
    print(f"📖 索引書籍: {meta['book_title']}")
    print(f"   分塊數: {meta['num_chunks']} | chunk_size={meta['chunk_size']} | overlap={meta['overlap']}")
    print(f"   儲存位置: {_build.QDRANT_PATH}")
    print(f"   檢索模式: {mode}")
    print(f"   Rerank: {'on' if use_rerank else 'off'}")

    client = _build.init_qdrant_client(recreate=False)
    model = _build.load_embedding_model(meta["embedding_model"])
    hybrid_rag = None
    if mode == "hybrid" or use_rerank:
        from rag_native import NativeRAG

        hybrid_rag = NativeRAG()

    query_map = prepare_test_queries()
    if all_questions:
        items = flatten_queries(query_map)
    else:
        # 預設每類 2 題，共 6 題，避免互動太久
        items = []
        for category, questions in query_map.items():
            for q in questions[:2]:
                items.append((category, q))

    evaluator = SearchQualityEvaluator()

    print(f"\n🔍 開始搜尋測試（共 {len(items)} 題）")
    print("=" * 80)

    for category, question in items:
        print(f"\n## [{category}]")
        if mode == "hybrid" or use_rerank:
            results = hybrid_rag.retrieve(
                question,
                top_k=3,
                mode=mode,
                use_rerank=use_rerank,
            )
        else:
            results = _build.search(client, model, question)
        if preview:
            _build.print_search_results(question, results)
        else:
            evaluator.evaluate_single_query(question, results)

    if not preview:
        evaluator.print_summary()
        evaluator.save_report()


def main():
    parser = argparse.ArgumentParser(description="RAG 搜尋品質評估")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="只顯示搜尋結果，不進行互動評分",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="評估全部 11 題（預設 6 題）",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="只顯示說明與題目列表",
    )
    parser.add_argument(
        "--mode",
        choices=("vector", "hybrid"),
        default="vector",
        help="檢索模式：vector（預設）或 hybrid（BM25 + 向量 RRF）",
    )
    parser.add_argument(
        "--use-rerank",
        action="store_true",
        help="啟用 Cross-encoder rerank（候選 M=15 → top_k）",
    )
    parser.add_argument(
        "--strategy",
        choices=("vector", "hybrid", "hybrid_rerank"),
        default=None,
        help="檢索策略（等同 API retrieval_strategy，覆寫 --mode / --use-rerank）",
    )
    args = parser.parse_args()

    strategy = args.strategy
    mode = args.mode
    use_rerank = args.use_rerank
    if strategy:
        if strategy == "vector":
            mode, use_rerank = "vector", False
        elif strategy == "hybrid":
            mode, use_rerank = "hybrid", False
        else:
            mode, use_rerank = "hybrid", True

    if args.demo:
        run_demo()
        return

    run_search_tests(
        preview=args.preview,
        all_questions=args.all,
        mode=mode,
        use_rerank=use_rerank,
    )


if __name__ == "__main__":
    main()
