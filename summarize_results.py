#!/usr/bin/env python3
"""
Summarize experiment results for the research loop.

Reads `results.tsv` and emits:
- a machine-readable JSON frontier summary for agents
- a short Markdown brief for humans
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results.tsv"), help="Input TSV path")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("analysis_summary.json"),
        help="Output JSON summary path",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("analysis_summary.md"),
        help="Output Markdown summary path",
    )
    return parser.parse_args()


def _load_results(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for idx, row in enumerate(reader):
            if not row:
                continue
            row = dict(row)
            row["mc_acc"] = float(row["mc_acc"])
            row["bin_acc"] = float(row["bin_acc"])
            row["memory_gb"] = float(row["memory_gb"])
            row["status"] = str(row["status"]).strip().upper()
            row["row_index"] = idx
            rows.append(row)
    if not rows:
        raise ValueError(f"No experiment rows found in {path}")
    return rows


def _score_row(row: dict) -> tuple[float, float, float]:
    return (row["bin_acc"], row["mc_acc"], -row["memory_gb"])


def _summarize_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["status"].lower() for row in rows)
    return {
        "keep": counts.get("keep", 0),
        "discard": counts.get("discard", 0),
        "crash": counts.get("crash", 0),
        "total": len(rows),
    }


def _frontier_history(rows: list[dict]) -> list[dict]:
    history = [row for row in rows if row["status"] == "KEEP"]
    return [compact_row(row) for row in history]


def _current_best(rows: list[dict]) -> dict:
    kept = [row for row in rows if row["status"] == "KEEP"]
    pool = kept if kept else rows
    return max(pool, key=_score_row)


def _best_mc_side_run(rows: list[dict], current_best: dict) -> dict:
    side_rows = [row for row in rows if row["description"] != current_best["description"]]
    if not side_rows:
        return current_best
    return max(side_rows, key=lambda row: (row["mc_acc"], row["bin_acc"], -row["memory_gb"]))


def _promising_near_misses(rows: list[dict], current_best: dict, limit: int = 5) -> list[dict]:
    candidates = []
    for row in rows:
        if row["description"] == current_best["description"]:
            continue
        bin_gap = current_best["bin_acc"] - row["bin_acc"]
        if bin_gap < 0:
            continue
        row_with_gap = dict(row)
        row_with_gap["bin_gap"] = bin_gap
        candidates.append(row_with_gap)
    candidates.sort(key=lambda row: (row["bin_gap"], -row["mc_acc"], row["memory_gb"]))
    return [compact_row(row, include_gap=True) for row in candidates[:limit]]


def _next_hypothesis_hints(rows: list[dict], current_best: dict, best_mc_side: dict) -> list[str]:
    descs = [row["description"].lower() for row in rows if row["status"] != "CRASH"]
    hints = [
        (
            "Exploit the current frontier first: keep the core recipe from "
            f"{current_best['description']} and vary only one axis at a time around it."
        )
    ]

    if best_mc_side["description"] != current_best["description"]:
        hints.append(
            "Mine the best mc_acc side run "
            f"({best_mc_side['description']}) for one transferable idea, but keep the "
            "current screening frontier as the base."
        )

    if any("schednone" in desc for desc in descs):
        hints.append(
            "Avoid repeating schednone unchanged; it already regressed bin_acc in recent trials."
        )
    if any("novflip" in desc for desc in descs):
        hints.append(
            "Treat no-vertical-flip results as side-signal only unless paired with a new "
            "independent change that targets screening."
        )
    if any("binaryloss" in desc for desc in descs):
        hints.append(
            "Do not revisit grouped binary loss directly; recent runs showed screening and "
            "multiclass regressions."
        )
    if any("drop0" in desc for desc in descs) or any("drop06" in desc for desc in descs):
        hints.append(
            "Keep transfer dropout near the current frontier value; wider dropout swings have "
            "already underperformed."
        )

    return hints[:5]


def compact_row(row: dict, *, include_gap: bool = False) -> dict:
    compact = {
        "commit": row["commit"],
        "description": row["description"],
        "status": row["status"],
        "mc_acc": row["mc_acc"],
        "bin_acc": row["bin_acc"],
        "memory_gb": row["memory_gb"],
    }
    if include_gap and "bin_gap" in row:
        compact["bin_gap"] = row["bin_gap"]
    return compact


def build_summary(rows: list[dict]) -> dict:
    current_best = _current_best(rows)
    best_mc_side = _best_mc_side_run(rows, current_best)
    summary = {
        "counts": _summarize_counts(rows),
        "current_best": compact_row(current_best),
        "best_mc_side_run": compact_row(best_mc_side),
        "frontier_history": _frontier_history(rows),
        "promising_near_misses": _promising_near_misses(rows, current_best),
        "next_hypothesis_hints": _next_hypothesis_hints(rows, current_best, best_mc_side),
    }
    return summary


def render_markdown(summary: dict) -> str:
    best = summary["current_best"]
    best_mc = summary["best_mc_side_run"]
    counts = summary["counts"]
    lines = [
        "# Research Loop Analysis Summary",
        "",
        "## Current screening frontier",
        "",
        (
            f"- `{best['description']}` | bin_acc={best['bin_acc']:.6f} | "
            f"mc_acc={best['mc_acc']:.6f} | status={best['status']}"
        ),
        "",
        "## Side best mc_acc run",
        "",
        (
            f"- `{best_mc['description']}` | mc_acc={best_mc['mc_acc']:.6f} | "
            f"bin_acc={best_mc['bin_acc']:.6f} | status={best_mc['status']}"
        ),
        "",
        "## Counts",
        "",
        (
            f"- total={counts['total']}, keep={counts['keep']}, "
            f"discard={counts['discard']}, crash={counts['crash']}"
        ),
        "",
        "## Promising near misses",
        "",
    ]

    for row in summary["promising_near_misses"][:5]:
        lines.append(
            f"- `{row['description']}` | bin_gap={row['bin_gap']:.6f} | "
            f"bin_acc={row['bin_acc']:.6f} | mc_acc={row['mc_acc']:.6f} | status={row['status']}"
        )

    lines.extend(["", "## Next hypothesis hints", ""])
    for hint in summary["next_hypothesis_hints"]:
        lines.append(f"- {hint}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    rows = _load_results(args.results)
    summary = build_summary(rows)

    args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(render_markdown(summary), encoding="utf-8")

    print(
        "Wrote summary for %d experiments. Current best: %s (bin_acc=%.6f, mc_acc=%.6f)"
        % (
            summary["counts"]["total"],
            summary["current_best"]["description"],
            summary["current_best"]["bin_acc"],
            summary["current_best"]["mc_acc"],
        )
    )


if __name__ == "__main__":
    main()
