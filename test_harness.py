import unittest
from unittest.mock import patch

import eval as harness_eval
from llm_judge_level_34 import judge_unordered_set_overall
from metric_router import MetricRoute, clear_route_cache, route_metric


class MetricRouterTests(unittest.TestCase):
    def setUp(self):
        clear_route_cache()

    def test_routes_each_question_once(self):
        calls = []

        def completion(_prompt):
            calls.append(_prompt)
            return '{"metric":"unordered_set","reason":"Eight teams; order is irrelevant."}'

        first = route_metric(
            "List the eight qualifying teams in any order.",
            ["AG.AL", "T1"],
            level=4,
            question_id="ewc",
            completion=completion,
        )
        second = route_metric(
            "List the eight qualifying teams in any order.",
            ["AG.AL", "T1"],
            level=4,
            question_id="ewc",
            completion=completion,
        )
        self.assertEqual(first.metric, "unordered_set")
        self.assertEqual(second.metric, "unordered_set")
        self.assertEqual(len(calls), 1)

    def test_contract_conflict_skips_router(self):
        route = route_metric(
            r"Return only \boxed{Yes} or \boxed{No}.",
            ["Coria"],
            level=1,
            completion=lambda _: self.fail("router must not run for a conflict"),
        )
        self.assertTrue(route.fallback)
        self.assertIn("not boolean", route.contract_conflict)

    def test_router_accepts_json_markdown_fence(self):
        route = route_metric(
            "List the qualifying teams in any order.",
            ["AG.AL", "T1"],
            level=4,
            completion=lambda _: (
                "```json\n"
                '{"metric":"unordered_set","reason":"Order is irrelevant."}'
                "\n```"
            ),
        )
        self.assertEqual(route.metric, "unordered_set")

    def test_router_retries_malformed_json(self):
        responses = iter(
            [
                '{"metric":"unordered_set","reason":"unterminated}',
                '{"metric":"unordered_set","reason":"retry succeeded"}',
            ]
        )
        route = route_metric(
            "List the qualifying teams in any order.",
            ["AG.AL", "T1"],
            level=4,
            completion=lambda _: next(responses),
        )
        self.assertEqual(route.metric, "unordered_set")
        self.assertFalse(route.fallback)


class UnorderedSetTests(unittest.TestCase):
    def test_reordered_identical_sets_score_full(self):
        self.assertEqual(
            judge_unordered_set_overall(
                "Name the qualifying teams; order does not matter.",
                ["T1", "AG.AL"],
                ["AG.AL", "T1"],
            ),
            1.0,
        )

    @patch("llm_judge_level_34.get_ai_response", return_value=r"\boxed{2}")
    def test_semantic_set_match_scores_full(self, _mock_llm):
        self.assertEqual(
            judge_unordered_set_overall(
                "Name the qualifying teams; order does not matter.",
                ["Anyone's Legend", "T1"],
                ["AG.AL", "T1"],
            ),
            1.0,
        )

    @patch(
        "eval.prewarm_metric_routes",
        return_value=[MetricRoute("unordered_set", "test route")],
    )
    def test_routed_scorer_dispatches_to_unordered_metric(self, _routes):
        scores, average, routes = harness_eval.estimate_routed_scores(
            ["Name teams in any order."],
            [["AG.AL", "T1"]],
            [["T1", "AG.AL"]],
            [None],
            levels=[4],
        )
        self.assertEqual(scores, [1.0])
        self.assertEqual(average, 1.0)
        self.assertEqual(routes[0].metric, "unordered_set")


if __name__ == "__main__":
    unittest.main()
