# RAG 上线前五项检查

这份清单的目标不是做 demo，而是把 `Book Spirit` 往「可观测、可监控、可维护」的方向推进。

## 1. 功能检查

目标：确认系统在真实问题和坏案例上按预期工作。

在 `Book Spirit` 里，至少要覆盖：
- 正常检索：`/search` 能返回相关段落
- 正常问答：`/ask` 在有命中的情况下返回答案和引用
- 空结果：没有命中时走 `empty_retrieval_answer`，而不是乱答
- 边缘案例：overview 问题、跨书问题、错误 `book_id`、过短问题

建议做法：
- 把 bad cases 固定到 `tests/` 和 `eval/test_cases.json`
- 每次改检索策略或 prompt 前后都重跑
- API 测试不要只断言 200，要断言 `sources`、拒答、错误格式都符合预期

## 2. 性能检查

目标：知道慢在哪里，而不是只知道总耗时。

至少拆成这几段：
- 路由与请求解析
- 检索：vector / hybrid / rerank / fallback
- Prompt 组装
- LLM 生成
- 日志写入

建议指标：
- `search_latency_ms`
- `ask_total_latency_ms`
- `retrieve_latency_ms`
- `llm_latency_ms`
- `fallback_step_count`

当前项目里，`/search` 和 `/ask` 已有 `stage_timings`；`/ask` 也会记录 `token_usage` 与 structured logs。下一步是把这些日志接到长期监控面板。

## 3. 质量检查

目标：区分「检索没捞到」和「模型乱说」。

至少看三类指标：
- 检索质量：MRR、Recall@k、golden pass rate
- 生成质量：幻觉率
- 拒答质量：答不出来时是否正确拒答

对 `Book Spirit` 的落地建议：
- 保留 `golden eval` 作为检索主指标
- 新增一小批「应拒答」问题，统计拒答率
- 用 `uv run python scripts/eval.py --golden --judge-answers` 跑 answer-level 基线
- 人工抽样检查引用是否真能支撑答案

原则：
- 不要只看回答是否漂亮
- 先确认 retrieval 对，再谈 generation

## 4. 安全检查

目标：避免把不该回答、不能看的内容直接吐出去。

这个项目现在是单用户书籍 RAG，但上线思维仍应先补三层：
- 查询过滤：拦截明显异常或恶意输入
- 文档访问控制：未来若支持多用户或多书权限，需要按 `book_id` / ACL 过滤
- Prompt 兜底：明确要求模型只根据提供来源作答，不知道就说不知道

最小可做版本：
- 对异常长输入、空输入、非法 `book_id` 做早期拒绝
- 对 `/ask` 加入更明确的 grounded-answer system instruction
- 记录被拒绝请求的原因，方便后续审计

## 5. 成本检查

目标：知道一次请求到底花了什么。

对 `Book Spirit`，成本不一定是云账单，也可能是：
- embedding 生成时间
- rerank 成本
- Ollama 推理时间
- Qdrant 检索资源消耗

建议至少统计：
- 每次 `/ask` 的 prompt token / completion token
- rerank 是否启用
- 命中的 chunk 数
- 是否进入 fallback

现在 `Book Spirit` 已能从 Ollama 响应提取 `prompt_eval_count` / `eval_count`，写入 API 回应、structured log 与 `query_logs.token_usage`。

如果未来切到云模型，再把这些乘上单价，就能得到：
- 单次请求成本
- 每日 / 每月成本
- 缓存能省多少

## 对这个项目的优先顺序

如果只能先补三件事，顺序应是：
1. 功能 bad-case 固化
2. 检索 / 生成分阶段打点
3. 质量与拒答的最小指标

原因很简单：
- 没有功能基线，就不知道系统是不是坏了
- 没有分阶段耗时，就找不到瓶颈
- 没有质量指标，就会把「能答」误当成「答得对」

## 对应到 Book Spirit Next

这份清单和 `data/book-spirit-next.md` 对齐后，接下来最值得补的是：
- deployment path：让系统稳定起得来
- logging / timing：让问题能被定位
- integration tests：让坏案例不会回归
- thin demo UI：让检查结果能被快速展示
