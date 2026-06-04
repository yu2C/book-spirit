from unittest.mock import patch

from core.query_planner import QueryPlan, _parse_plan_json, plan_query


def test_parse_plan_json():
    raw = 'noise {"queries": ["DAO hack"], "top_k": 8, "intent": "event"} end'
    data = _parse_plan_json(raw)
    assert data["queries"] == ["DAO hack"]
    assert data["top_k"] == 8


def test_plan_query_mock():
    fake = '{"queries": ["Ethereum DAO timeline"], "top_k": 5, "intent": "overview"}'
    with patch("core.query_planner.requests.post") as post:
        post.return_value.json.return_value = {"response": fake}
        post.return_value.raise_for_status = lambda: None
        plan = plan_query("這本書在講什麼", book_title="Cryptopians", language_hint="en")
    assert "Ethereum" in plan.queries[0]
    assert plan.top_k == 5


def test_query_plan_effective_queries():
    plan = QueryPlan(queries=["a", "b"])
    assert plan.effective_queries("fallback") == ["a", "b"]
    assert QueryPlan().effective_queries("fallback") == ["fallback"]
