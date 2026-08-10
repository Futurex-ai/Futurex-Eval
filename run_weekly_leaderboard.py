"""Score every local submission for one resolved FutureX week."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from eval import estimate_routed_scores, extract_answers, prewarm_metric_routes
from run_single_submission import load_records, parse_ground_truth, record_id


LEVEL_WEIGHTS = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}


def parse_submission_identity(path: Path):
    """Extract public leaderboard fields from the standard submission filename."""
    match = re.match(
        r"^org-(?P<org>.*?)-agent-(?P<agent>.*?)-model[-_]"
        r"(?P<model>.*?)(?:\s+org-.*)?\.(?:json|jsonl)$",
        path.name,
    )
    if match:
        return (
            match.group("model").strip(),
            match.group("agent").strip(),
            match.group("org").strip(),
        )
    return path.stem, "Unknown", "Unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--details-output", required=True, type=Path)
    parser.add_argument(
        "--gt-confidence",
        choices=["high", "low"],
        help="Only evaluate GT rows at this confidence level.",
    )
    args = parser.parse_args()

    dataframe = pd.read_excel(args.ground_truth)
    rows = [
        row for _, row in dataframe.iterrows()
        if pd.notna(row.get("ground_truth"))
    ]
    if args.gt_confidence:
        if "gt_confidence" not in dataframe.columns:
            raise ValueError("--gt-confidence requires a gt_confidence column")
        rows = [
            row for row in rows
            if str(row.get("gt_confidence", "")).strip().casefold()
            == args.gt_confidence
        ]
    questions = [
        str(row.get("prompt") or row.get("title") or row.get("en_title"))
        for row in rows
    ]
    truths = [parse_ground_truth(row["ground_truth"]) for row in rows]
    levels = [int(row["level"]) for row in rows]
    ids = [str(row["id"]) for row in rows]
    stds = [row.get("std") for row in rows]
    level_totals = Counter(levels)

    # This is intentionally called once. Subsequent calls in
    # estimate_routed_scores hit the process-local route cache.
    routes = prewarm_metric_routes(
        questions, truths, levels=levels, question_ids=ids
    )

    leaderboard = []
    details = []
    prediction_paths = sorted(
        path for path in args.prediction_dir.iterdir()
        if path.suffix in {".json", ".jsonl"}
    )
    def write_reports():
        ranked = sorted(
            leaderboard, key=lambda item: item["overallScore"], reverse=True
        )
        args.output.write_text(json.dumps(ranked, ensure_ascii=False, indent=2))
        args.details_output.write_text(
            json.dumps(
                {
                    "ground_truth": str(args.ground_truth),
                    "eventsTotal": len(ids),
                    "metricCounts": Counter(route.metric for route in routes),
                    "routerFallbacks": sum(route.fallback for route in routes),
                    "leaderboard": ranked,
                    "submissions": details,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    for file_index, path in enumerate(prediction_paths, start=1):
        print(
            f"[{file_index}/{len(prediction_paths)}] evaluating {path.name}",
            flush=True,
        )
        try:
            records = {
                rid: record
                for record in load_records(path)
                if (rid := record_id(record)) is not None
            }
            selected = [index for index, id_ in enumerate(ids) if id_ in records]
            events = [dict(records[ids[index]]) for index in selected]
            predictions = extract_answers(
                [questions[index] for index in selected], events
            )
            evaluated_scores, _, evaluated_routes = estimate_routed_scores(
                [questions[index] for index in selected],
                [truths[index] for index in selected],
                predictions,
                [stds[index] for index in selected],
                levels=[levels[index] for index in selected],
                question_ids=[ids[index] for index in selected],
            )
            level_score_raw = {level: 0.0 for level in LEVEL_WEIGHTS}
            for index, score in zip(selected, evaluated_scores):
                level_score_raw[levels[index]] += score
            for level, total in level_totals.items():
                level_score_raw[level] /= total

            model_name, agent_framework, organization = parse_submission_identity(path)
            overall_score = sum(
                LEVEL_WEIGHTS[level] * level_score_raw[level]
                for level in LEVEL_WEIGHTS
            )
            entry = {
                "submission": path.name,
                "overallScore": round(overall_score * 100, 4),
                "level1Score": round(level_score_raw[1] * 100, 4),
                "level2Score": round(level_score_raw[2] * 100, 4),
                "level3Score": round(level_score_raw[3] * 100, 4),
                "level4Score": round(level_score_raw[4] * 100, 4),
                "modelName": model_name,
                "agentFramework": agent_framework,
                "organization": organization,
                "testType": 0 if agent_framework.casefold() == "search" else 1,
                "numberOfEvents": len(ids),
                "eventsEvaluated": len(selected),
                "eventsTotal": len(ids),
                "coverage": round(len(selected) / len(ids), 4),
                "routerFallbacks": sum(route.fallback for route in evaluated_routes),
            }
            leaderboard.append(entry)
            details.append(
                {
                    **entry,
                    "scores": [
                        {"id": ids[index], "score": score}
                        for index, score in zip(selected, evaluated_scores)
                    ],
                }
            )
        except Exception as exc:
            model_name, agent_framework, organization = parse_submission_identity(path)
            leaderboard.append(
                {
                    "submission": path.name,
                    "overallScore": 0.0,
                    "level1Score": 0.0,
                    "level2Score": 0.0,
                    "level3Score": 0.0,
                    "level4Score": 0.0,
                    "modelName": model_name,
                    "agentFramework": agent_framework,
                    "organization": organization,
                    "testType": 0 if agent_framework.casefold() == "search" else 1,
                    "numberOfEvents": len(ids),
                    "eventsEvaluated": 0,
                    "eventsTotal": len(ids),
                    "coverage": 0.0,
                    "error": str(exc),
                }
            )
        write_reports()

    leaderboard.sort(key=lambda item: item["overallScore"], reverse=True)
    write_reports()
    print(
        json.dumps(
            {
                "submissions": len(prediction_paths),
                "events": len(ids),
                "metricCounts": Counter(route.metric for route in routes),
                "routerFallbacks": sum(route.fallback for route in routes),
                "leaderboard": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
