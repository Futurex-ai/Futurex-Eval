"""Evaluate one local FutureX submission with cached metric routing."""

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from eval import estimate_routed_scores, extract_answers, prewarm_metric_routes


def load_records(path: Path):
    content = path.read_text(encoding="utf-8").strip()
    if path.suffix == ".jsonl":
        data = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        data = json.loads(content)
        if isinstance(data, dict):
            for key in ("data", "dataset", "predictions", "results"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a prediction list")
    return data


def parse_ground_truth(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except (SyntaxError, ValueError):
            return [value]
    return [value]


def record_id(record):
    for key in ("id", "session_id", "sample_id", "session"):
        if record.get(key) is not None:
            return str(record[key])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    gt_df = pd.read_excel(args.ground_truth).replace({float("nan"): None})
    predictions = {
        rid: record
        for record in load_records(args.prediction)
        if (rid := record_id(record)) is not None
    }

    rows = [
        row
        for _, row in gt_df.iterrows()
        if row.get("ground_truth") is not None and str(row["id"]) in predictions
    ]
    if not rows:
        raise ValueError("No prediction IDs overlap ground-truth IDs")

    questions = [str(row.get("prompt") or row.get("title") or row.get("en_title")) for row in rows]
    truths = [parse_ground_truth(row["ground_truth"]) for row in rows]
    levels = [int(row["level"]) for row in rows]
    ids = [str(row["id"]) for row in rows]
    stds = [row.get("std") for row in rows]
    events = [dict(predictions[id_]) for id_ in ids]
    parsed_predictions = extract_answers(questions, events)

    prewarm_metric_routes(questions, truths, levels=levels, question_ids=ids)
    scores, average, routes = estimate_routed_scores(
        questions,
        truths,
        parsed_predictions,
        stds,
        levels=levels,
        question_ids=ids,
    )

    report_rows = [
        {
            "id": id_,
            "level": level,
            "ground_truth": truth,
            "prediction": prediction,
            "score": score,
            "metric": route.metric,
            "route_reason": route.reason,
            "route_fallback": route.fallback,
            "contract_conflict": route.contract_conflict,
        }
        for id_, level, truth, prediction, score, route in zip(
            ids, levels, truths, parsed_predictions, scores, routes
        )
    ]
    report = {
        "ground_truth": str(args.ground_truth),
        "prediction": str(args.prediction),
        "events_evaluated": len(report_rows),
        "overall_score": average,
        "metric_counts": Counter(route.metric for route in routes),
        "router_fallbacks": sum(route.fallback for route in routes),
        "contract_conflicts": [
            {"id": id_, "detail": route.contract_conflict}
            for id_, route in zip(ids, routes)
            if route.contract_conflict
        ],
        "rows": report_rows,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
