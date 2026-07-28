"""
檢索品質評測（不含 LLM 生成）。

用法：
  uv run python scripts/eval.py              # 互動評分（預設 6 題）
  uv run python scripts/eval.py --preview    # 只顯示搜尋結果
  uv run python scripts/eval.py --all        # 全部題目
  uv run python scripts/eval.py --golden     # 金標自動評分（eval/test_cases.json）
  uv run python scripts/eval.py --demo       # 說明與題目列表
"""

import argparse
import json
import sys
from typing import Dict, List

from core.config import QDRANT_PATH, SearchFilters, resolve_retrieval_settings
from core.index_catalog import index_meta_path, load_index_meta
from core.retrieval_service import search_with_fallback
from eval.golden import load_test_cases, score_case, summarize_golden


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

            rating = (
                input("\n   👉 這個結果相關嗎？ (y/partial/n，Enter=跳過評分): ").strip().lower()
            )
            if rating == "y":
                relevance = "relevant"
            elif rating == "partial":
                relevance = "partial"
            elif rating in ("n", "no"):
                relevance = "irrelevant"
            else:
                relevance = "skipped"

            ratings.append(
                {
                    "rank": rank,
                    "score": score,
                    "relevance": relevance,
                    "text_preview": text,
                }
            )

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

        print("\n📊 結果統計:")
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
        print("\n📈 指標:")
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


def flatten_queries(
    queries: Dict[str, List[str]], categories: List[str] | None = None
) -> List[str]:
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


def print_search_preview(question: str, search_results: List[Dict]) -> None:
    print(f"\n❓ {question}")
    for rank, result in enumerate(search_results[:3], 1):
        text = result.get("text", "")[:150].replace("\n", " ")
        score = result.get("score", 0.0)
        chapter = result.get("chapter") or "(無)"
        heading = result.get("heading") or "(無)"
        print(f"  [{rank}] {score:.3f} | {chapter} → {heading}")
        print(f"      {text}...")


def run_demo():
    print("""
📚 搜尋品質評估工具

流程：
  1. 先執行 uv run python scripts/build_index.py 建立索引
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


def run_golden_tests(
    *,
    mode: str = "vector",
    use_rerank: bool = False,
    top_k: int = 3,
    cases_path: str | None = None,
    judge_answers: bool = False,
):
    """用 eval/test_cases.json 自動評檢索（must_contain_any，抗 chunk 重建）。"""
    if not QDRANT_PATH.exists() or not index_meta_path().exists():
        print("❌ 找不到 Qdrant 索引")
        print("   請先執行: uv run python scripts/build_index.py")
        sys.exit(1)

    from pathlib import Path

    from core.config import CHUNKER_MODE
    from core.pipeline import NativeRAG

    path = Path(cases_path) if cases_path else None
    cases = load_test_cases(path)
    meta = load_index_meta()
    books = meta.get("books") or {}
    if books:
        summary = ", ".join(f"{bid}({info.get('num_chunks', 0)})" for bid, info in books.items())
        print(f"📖 索引書籍: {summary}")
    print(f"   chunker={CHUNKER_MODE} | 檢索: {mode} | rerank: {'on' if use_rerank else 'off'}")
    print(f"🏅 金標題數: {len(cases)}（eval/test_cases.json）")
    print(f"   規則: top-{top_k} 任一块含 must_contain_any → pass")
    if judge_answers:
        print("   answer judge: on（需 Ollama；以 expected_answer 相似度做輕量基線）")

    rag = NativeRAG()
    scored: List[Dict] = []

    print(f"\n{'=' * 80}")
    for case in cases:
        book_id = case.get("book_id")
        filters = SearchFilters(book_id=book_id) if book_id else None
        results = search_with_fallback(
            rag,
            case["question"],
            top_k=top_k,
            filters=filters,
            mode=mode,
            use_rerank=use_rerank,
            scope_book_id=book_id,
        ).documents
        answer = None
        if judge_answers:
            ask_result = rag.ask(
                case["question"],
                top_k=top_k,
                filters=filters,
                mode=mode,
                use_rerank=use_rerank,
                book_id=book_id,
            )
            answer = ask_result.get("answer")
        row = score_case(case, results, top_k=top_k, answer=answer)
        scored.append(row)
        mark = "✅" if row["pass"] else "❌"
        print(f"\n{mark} [{case.get('category')}] {case['question']}")
        if case.get("expected_answer"):
            print(f"   📎 參考答案: {case['expected_answer']}")
        print(
            f"   hit@{top_k}={row['hit_at_k']}  precision@{top_k}={row['precision_at_k']:.2f}",
            end="",
        )
        if row["chapter_ok"] is not None:
            print(f"  chapter_ok={row['chapter_ok']}", end="")
        if row.get("answer_evaluated"):
            print(
                f"  answer_pass={row['answer_pass']}  answer_score={row['answer_score']:.2f}"
                if row.get("answer_score") is not None
                else f"  answer_pass={row['answer_pass']}"
            )
        else:
            print()
        for pr in row["per_rank"]:
            m = "✓" if pr["matched"] else "·"
            print(
                f"   [{pr['rank']}] {m} {pr['score']:.3f} | {pr['chapter'][:24]} | {pr['preview']}..."
            )

    summary = summarize_golden(scored)
    print(f"\n{'=' * 80}")
    print("📊 金標總結")
    print(f"   通過: {summary['passed']}/{summary['total']} ({summary['pass_rate'] * 100:.0f}%)")
    print(f"   平均 precision@{top_k}: {summary['avg_precision_at_k']:.2f}")
    print(f"   平均 MRR: {summary['avg_mrr']:.2f}")
    if summary['refusal_total']:
        print(
            f"   拒答通過: {summary['refusal_passed']}/{summary['refusal_total']} "
            f"({summary['refusal_rate'] * 100:.0f}%)"
        )
    if summary["answer_total"]:
        print(
            f"   answer baseline: {summary['answer_passed']}/{summary['answer_total']} "
            f"({summary['answer_pass_rate'] * 100:.0f}%)"
        )
        print(f"   hallucination baseline: {summary['hallucination_rate'] * 100:.0f}%")

    from core.ingest_hints import golden_ingest_warnings

    case_book_ids = [c.get("book_id") for c in cases if c.get("book_id")]
    ingest_warnings = golden_ingest_warnings(
        meta,
        pass_rate=summary["pass_rate"],
        case_book_ids=case_book_ids,
    )
    for w in ingest_warnings:
        print(f"\n{w}")

    report_path = "golden_eval_report.json"
    report_payload = {
        "summary": summary,
        "cases": scored,
        "ingest_warnings": ingest_warnings,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"💾 報告: {report_path}")
    return summary


def run_search_tests(
    preview: bool = False,
    all_questions: bool = False,
    mode: str = "vector",
    use_rerank: bool = False,
):
    if not QDRANT_PATH.exists() or not index_meta_path().exists():
        print("❌ 找不到 Qdrant 索引")
        print("   請先執行: uv run python scripts/build_index.py")
        sys.exit(1)

    from core.hybrid import resolve_bm25_paths

    if mode == "hybrid" and not resolve_bm25_paths(None):
        print("❌ hybrid 模式需要 BM25 語料")
        print("   請重新執行: uv run python scripts/build_index.py --all")
        sys.exit(1)

    meta = load_index_meta()
    books = meta.get("books") or {}
    if books:
        summary = ", ".join(f"{bid}({info.get('num_chunks', 0)})" for bid, info in books.items())
        print(f"📖 索引書籍: {summary}")
    else:
        print(f"📖 索引書籍: {meta.get('book_title', '(legacy)')}")
    from core.config import CHUNKER_MODE

    print(
        f"   chunker={CHUNKER_MODE} | chunk_size={meta.get('chunk_size')} | overlap={meta.get('overlap')}"
    )
    print(f"   儲存位置: {QDRANT_PATH}")
    print(f"   檢索模式: {mode}")
    print(f"   Rerank: {'on' if use_rerank else 'off'}")

    from core.pipeline import NativeRAG

    rag = NativeRAG()

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
        results = search_with_fallback(
            rag,
            question,
            top_k=3,
            mode=mode,
            use_rerank=use_rerank,
        ).documents
        if preview:
            print_search_preview(question, results)
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
        help="評估全部 10 題（預設每類 2 題）",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="只顯示說明與題目列表",
    )
    parser.add_argument(
        "--golden",
        action="store_true",
        help="金標自動評分（eval/test_cases.json，must_contain_any）",
    )
    parser.add_argument(
        "--cases",
        metavar="PATH",
        default=None,
        help="金標 JSON 路徑（預設 eval/test_cases.json）",
    )
    parser.add_argument(
        "--mode",
        choices=("vector", "hybrid"),
        default=None,
        help="檢索模式（未指定時依 RETRIEVAL_STRATEGY / env，預設 hybrid_rerank）",
    )
    parser.add_argument(
        "--use-rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否 rerank（未指定時依 env；預設 on）",
    )
    parser.add_argument(
        "--judge-answers",
        action="store_true",
        help="golden 時連 ask 一起跑，做 answer-level 輕量基線（需 Ollama）",
    )
    parser.add_argument(
        "--strategy",
        choices=("vector", "hybrid", "hybrid_rerank"),
        default=None,
        help="檢索策略（等同 API retrieval_strategy，覆寫 --mode / --use-rerank）",
    )
    args = parser.parse_args()

    if args.strategy:
        if args.strategy == "vector":
            mode, use_rerank = "vector", False
        elif args.strategy == "hybrid":
            mode, use_rerank = "hybrid", False
        else:
            mode, use_rerank = "hybrid", True
    else:
        mode, use_rerank, _ = resolve_retrieval_settings(
            retrieval_mode=args.mode,
            use_rerank=args.use_rerank,
        )

    if args.demo:
        run_demo()
        return

    if args.golden:
        run_golden_tests(
            mode=mode,
            use_rerank=use_rerank,
            cases_path=args.cases,
            judge_answers=args.judge_answers,
        )
        return

    run_search_tests(
        preview=args.preview,
        all_questions=args.all,
        mode=mode,
        use_rerank=use_rerank,
    )


if __name__ == "__main__":
    main()
