"""Route each resolved FutureX question to an existing scoring metric once."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import diskcache

from llm_config import cache_namespace, get_llm_config, get_llm_timeout


ROUTER_PROMPT_VERSION = "metric-router-v1"
ALLOWED_METRICS = {"legacy_type_a", "legacy_type_b", "unordered_set"}
_ROUTE_CACHE: Dict[str, "MetricRoute"] = {}
_DISK_CACHE = diskcache.Cache(
    f"./{cache_namespace('metric_router_diskcache', ROUTER_PROMPT_VERSION)}"
)


@dataclass(frozen=True)
class MetricRoute:
    metric: str
    reason: str
    contract_conflict: Optional[str] = None
    fallback: bool = False


def default_metric_for_level(level: Optional[int]) -> str:
    return "legacy_type_a" if level is not None and level <= 2 else "legacy_type_b"


def _canonical_gt(ground_truth: Any) -> str:
    return json.dumps(ground_truth, ensure_ascii=False, sort_keys=True, default=str)


def _cache_key(
    question: str,
    ground_truth: Any,
    question_id: Optional[str],
    level: Optional[int],
) -> str:
    payload = {
        "question_id": question_id or "",
        "question": question,
        "ground_truth": _canonical_gt(ground_truth),
        "level": level,
        "prompt_version": ROUTER_PROMPT_VERSION,
        "model": get_llm_config()[2] or "",
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def detect_contract_conflict(question: str, ground_truth: Any) -> Optional[str]:
    """Flag only contradictions that are impossible for a scorer to resolve."""
    gt_items = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    gt_tokens = {str(item).strip().casefold() for item in gt_items}
    requires_boolean = bool(
        re.search(r"\\boxed\{yes\}.*?\\boxed\{no\}", question, re.IGNORECASE | re.DOTALL)
        or re.search(r"\byes\s*/\s*no\b", question, re.IGNORECASE)
    )
    if requires_boolean and not gt_tokens.issubset({"yes", "no"}):
        return "Prompt requires Yes/No but ground truth is not boolean."
    return None


def _router_prompt(question: str, ground_truth: Any, default_metric: str) -> str:
    return f"""Classify the scoring metric for one resolved prediction question.

Choose exactly one metric:
- legacy_type_a: fixed options or ordinary label list; preserve existing scorer.
- legacy_type_b: number, text, ordered list, or other existing Level-3/4 scorer.
- unordered_set: the question asks for a collection where item order is explicitly irrelevant.

Use unordered_set only when the question does NOT request a rank, sequence, position,
or ordered output. Do not invent a new ground-truth format.

QUESTION:
{question}

GROUND_TRUTH:
{_canonical_gt(ground_truth)}

DEFAULT_METRIC_FROM_LEVEL:
{default_metric}

Return JSON only:
{{"metric":"legacy_type_a|legacy_type_b|unordered_set","reason":"brief reason"}}"""


def _openai_router_completion(prompt: str) -> str:
    from openai import OpenAI

    api_key, base_url, model = get_llm_config()
    if not api_key or not model:
        raise ValueError("Configure DEEPSEEK_API_KEY or OPENAI_API_KEY and a model.")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=get_llm_timeout())
    for _attempt in range(3):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            # Reasoning-capable models may consume the first tokens in
            # ``reasoning_content`` before emitting the constrained JSON answer.
            max_tokens=512,
        )
        content = response.choices[0].message.content or ""
        if content.strip():
            return content
    raise RuntimeError("Router model returned empty content after 3 attempts.")


def _parse_route(response: str, default_metric: str) -> MetricRoute:
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
    payload = json.loads(text)
    metric = payload.get("metric")
    if metric not in ALLOWED_METRICS:
        raise ValueError(f"Unsupported metric route: {metric!r}")
    return MetricRoute(metric=metric, reason=str(payload.get("reason", "")))


def route_metric(
    question: str,
    ground_truth: Any,
    *,
    level: Optional[int] = None,
    question_id: Optional[str] = None,
    completion: Optional[Callable[[str], str]] = None,
) -> MetricRoute:
    """Select and cache the metric for one question, independent of predictions."""
    default_metric = default_metric_for_level(level)
    conflict = detect_contract_conflict(question, ground_truth)
    if conflict:
        return MetricRoute(
            metric=default_metric,
            reason="Skipped routing because the question/GT contract conflicts.",
            contract_conflict=conflict,
            fallback=True,
        )

    key = _cache_key(question, ground_truth, question_id, level)
    if key in _ROUTE_CACHE:
        return _ROUTE_CACHE[key]
    cached = _DISK_CACHE.get(key)
    if cached:
        route = MetricRoute(**cached)
        _ROUTE_CACHE[key] = route
        return route

    router_completion = completion or _openai_router_completion
    parse_error = None
    try:
        for _attempt in range(2):
            try:
                response = router_completion(
                    _router_prompt(question, ground_truth, default_metric)
                )
                route = _parse_route(response, default_metric)
                break
            except (json.JSONDecodeError, ValueError) as exc:
                parse_error = exc
        else:
            raise parse_error
    except Exception as exc:
        route = MetricRoute(
            metric=default_metric,
            reason=f"Router fallback: {exc}",
            fallback=True,
        )
    _ROUTE_CACHE[key] = route
    _DISK_CACHE.set(key, asdict(route))
    return route


def route_questions(
    questions: Sequence[str],
    ground_truths: Sequence[Any],
    *,
    levels: Optional[Sequence[Optional[int]]] = None,
    question_ids: Optional[Sequence[Optional[str]]] = None,
    completion: Optional[Callable[[str], str]] = None,
) -> List[MetricRoute]:
    """Prewarm one cached route per question for a weekly evaluation."""
    if len(questions) != len(ground_truths):
        raise ValueError("questions and ground_truths must have equal length")
    levels = levels or [None] * len(questions)
    question_ids = question_ids or [None] * len(questions)
    if len(levels) != len(questions) or len(question_ids) != len(questions):
        raise ValueError("levels and question_ids must match questions length")
    return [
        route_metric(
            question,
            ground_truth,
            level=level,
            question_id=question_id,
            completion=completion,
        )
        for question, ground_truth, level, question_id in zip(
            questions, ground_truths, levels, question_ids
        )
    ]


def clear_route_cache() -> None:
    """Test/helper hook for resetting in-memory and persistent route caches."""
    _ROUTE_CACHE.clear()
    _DISK_CACHE.clear()
