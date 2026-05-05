#!/usr/bin/env python3
"""Run and display the ASE Smart Brain multi-metric anomaly results.

The original script assumed fixed folders named ``classification``, ``data`` and
``scores``.  This version is intentionally path-configurable so the paper
artifact can be moved as a whole, or its ``Results`` / ``Classification_trend``
folders can be stored elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_METHOD = "A_GPT"
DEFAULT_SCHEMA = "naive"
DEFAULT_OUTPUT_NAME = "a_promote"
DEFAULT_FEATURE_COUNT = 4
DEFAULT_FEATURE_FILE = "test_predictions.npy"
DEFAULT_LABEL_FILE = "test_labels.npy"

NEGATIVE_METRICS = {
    "Max Latency",
    "Avg Latency",
    "Failure Count",
    "Failure Rate",
    "Retries",
    "Timeouts",
}
BUSINESS_METRICS = {
    "Request Count",
    "Success Count",
    "Success Rate",
}
INFRA_METRICS = {
    "Memory Usage",
    "CPU Usage",
}

TREND_DESC = {
    0: "stable",
    1: "slow increase",
    2: "slow decrease",
    3: "step increase",
    4: "step decrease",
}
WAVEFORM_DESC = {
    0: "normal cycle",
    1: "normal cycle",
    2: "slight clipping",
    3: "severe clipping",
    4: "cycle missing",
}
STEP_DESC = {
    0: "no local step",
    1: "local upward step",
    2: "local downward step",
}
MUTATION_DESC = {
    0: "no local mutation",
    1: "upward mutation",
    2: "downward mutation",
}


@dataclass
class CurveData:
    name: str
    group_name: str
    metric_name: str
    features: Dict[str, np.ndarray]
    labels: Dict[str, np.ndarray] = field(default_factory=dict)
    raw_curve: Optional[np.ndarray] = None

    @property
    def length(self) -> int:
        return min((len(values) for values in self.features.values()), default=0)

    def truncated_features(self, length: int) -> Dict[str, np.ndarray]:
        return {name: values[:length] for name, values in self.features.items()}


@dataclass
class GroupResult:
    name: str
    curve_count: int
    timesteps: int
    scores: np.ndarray
    indices: List[int]
    segments: List[Tuple[int, int]]
    reasons: Dict[str, int]
    score_path: Path
    indices_path: Path
    easytsad_score_path: Path


def discover_project_root(explicit_root: Optional[str]) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    search_starts = [Path.cwd(), Path(__file__).resolve().parent]
    seen: set[Path] = set()
    for start in search_starts:
        for candidate in (start, *start.parents):
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if (
                (candidate / "Classification_trend").is_dir()
                or (candidate / "Results").is_dir()
                or (candidate / "EasyTSAD").is_dir()
            ):
                return candidate
    return Path.cwd().resolve()


def resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def parse_curve_name(name: str) -> Tuple[str, str]:
    if "---" not in name:
        return "all_metrics", name.strip()
    group_name, metric_name = name.split("---", 1)
    return group_name.strip(), metric_name.strip()


def metric_kind(metric_name: str) -> str:
    if metric_name in NEGATIVE_METRICS:
        return "negative"
    if metric_name in BUSINESS_METRICS:
        return "business"
    if metric_name in INFRA_METRICS:
        return "infrastructure"
    return "unknown"


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "result"


def load_optional_array(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        return np.asarray(np.load(path)).reshape(-1)
    except Exception as exc:  # pragma: no cover - defensive for malformed npy files
        print(f"[WARN] Cannot load {path}: {exc}", file=sys.stderr)
        return None


def load_curve(
    curve_dir: Path,
    raw_data_dir: Optional[Path],
    feature_count: int,
    feature_file: str,
    label_file: str,
) -> CurveData:
    group_name, metric_name = parse_curve_name(curve_dir.name)
    features: Dict[str, np.ndarray] = {}
    labels: Dict[str, np.ndarray] = {}

    for feature_idx in range(feature_count):
        feature_name = f"feature_{feature_idx}"
        feature_dir = curve_dir / feature_name
        pred_path = feature_dir / feature_file
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing feature prediction file: {pred_path}")

        features[feature_name] = np.asarray(np.load(pred_path)).reshape(-1).astype(int)

        label_path = feature_dir / label_file
        label_values = load_optional_array(label_path)
        if label_values is not None:
            labels[feature_name] = label_values.astype(int)

    min_length = min(len(values) for values in features.values())
    features = {name: values[:min_length] for name, values in features.items()}
    labels = {name: values[:min_length] for name, values in labels.items()}

    raw_curve = None
    if raw_data_dir is not None:
        raw_curve = load_optional_array(raw_data_dir / curve_dir.name / "test.npy")
        if raw_curve is not None:
            raw_curve = raw_curve[:min_length]

    return CurveData(
        name=curve_dir.name,
        group_name=group_name,
        metric_name=metric_name,
        features=features,
        labels=labels,
        raw_curve=raw_curve,
    )


def load_curves(
    classification_dir: Path,
    raw_data_dir: Optional[Path],
    feature_count: int,
    feature_file: str,
    label_file: str,
) -> List[CurveData]:
    if not classification_dir.is_dir():
        raise FileNotFoundError(f"Classification directory does not exist: {classification_dir}")

    curves: List[CurveData] = []
    skipped: List[str] = []
    for curve_dir in sorted(path for path in classification_dir.iterdir() if path.is_dir()):
        try:
            curves.append(load_curve(curve_dir, raw_data_dir, feature_count, feature_file, label_file))
        except FileNotFoundError as exc:
            skipped.append(str(exc))

    if not curves:
        details = "\n".join(skipped[:5])
        raise RuntimeError(
            f"No valid curve folders found in {classification_dir}."
            + (f"\nFirst missing files:\n{details}" if details else "")
        )

    if skipped:
        print(f"[WARN] Skipped {len(skipped)} incomplete curve folders.")
    return curves


def group_curves(curves: Sequence[CurveData], group_by: str) -> Dict[str, List[CurveData]]:
    if group_by == "all":
        return {"all_metrics": list(curves)}

    grouped: Dict[str, List[CurveData]] = {}
    for curve in curves:
        grouped.setdefault(curve.group_name, []).append(curve)
    return dict(sorted(grouped.items()))


def contiguous_segments(indices: Sequence[int]) -> List[Tuple[int, int]]:
    if not indices:
        return []

    ordered = sorted(set(indices))
    segments: List[Tuple[int, int]] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        segments.append((start, previous))
        start = previous = index
    segments.append((start, previous))
    return segments


def add_mask_reason(target: np.ndarray, mask: np.ndarray, reasons: Dict[str, int], reason: str) -> None:
    if not mask.any():
        return
    reasons[reason] = reasons.get(reason, 0) + int(mask.sum())
    target[mask] = 1


def stable_event_mask(values: np.ndarray, event_values: Iterable[int], min_duration: int = 6) -> np.ndarray:
    event_values = set(event_values)
    mask = np.zeros(len(values), dtype=bool)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end] == values[start]:
            end += 1
        if values[start] in event_values and end - start >= min_duration:
            mask[start:end] = True
        start = end
    return mask


def analyze_group_offline(curves: Sequence[CurveData]) -> Tuple[np.ndarray, List[int], Dict[str, int]]:
    timesteps = min((curve.length for curve in curves), default=0)
    if timesteps <= 0:
        return np.zeros(0, dtype=np.int8), [], {}

    scores = np.zeros(timesteps, dtype=np.int8)
    reasons: Dict[str, int] = {}
    business_mutation_down = np.zeros(timesteps, dtype=int)
    negative_mutation_up = np.zeros(timesteps, dtype=int)

    for curve in curves:
        features = curve.truncated_features(timesteps)
        f0 = features["feature_0"]
        f1 = features["feature_1"]
        f2 = features["feature_2"]
        f3 = features["feature_3"]
        kind = metric_kind(curve.metric_name)

        waveform_mask = np.isin(f1, [2, 3, 4])
        add_mask_reason(scores, waveform_mask, reasons, "waveform_clipping_or_missing_cycle")

        if kind == "negative":
            add_mask_reason(scores, f0 == 3, reasons, "negative_metric_step_increase")
            add_mask_reason(scores, stable_event_mask(f2, [1]), reasons, "negative_metric_local_up_step")
            negative_mutation_up += (f3 == 1).astype(int)
        elif kind == "business":
            add_mask_reason(scores, f0 == 4, reasons, "business_metric_step_decrease")
            add_mask_reason(scores, stable_event_mask(f2, [2]), reasons, "business_metric_local_down_step")
            business_mutation_down += (f3 == 2).astype(int)
        elif kind == "infrastructure":
            add_mask_reason(scores, f0 == 3, reasons, "infrastructure_metric_resource_increase")

    add_mask_reason(
        scores,
        business_mutation_down >= 2,
        reasons,
        "multi_business_metrics_down_mutation",
    )
    add_mask_reason(
        scores,
        negative_mutation_up >= 2,
        reasons,
        "multi_negative_metrics_up_mutation",
    )

    indices = np.flatnonzero(scores > 0).astype(int).tolist()
    return scores, indices, reasons


def describe_curve_events(curve: CurveData, timesteps: int) -> str:
    features = curve.truncated_features(timesteps)
    event_lines: List[str] = []
    state = None
    start = 0

    for t in range(timesteps):
        current_state = tuple(int(features[f"feature_{i}"][t]) for i in range(DEFAULT_FEATURE_COUNT))
        if state is None:
            state = current_state
            start = t
            continue
        if current_state == state:
            continue
        event_lines.append(format_event_line(start, t - 1, state))
        state = current_state
        start = t

    if state is not None:
        event_lines.append(format_event_line(start, timesteps - 1, state))

    header = f"Metric '{curve.metric_name}' ({curve.name}) feature events:"
    return header + "\n" + "\n".join(event_lines)


def format_event_line(start: int, end: int, state: Tuple[int, int, int, int]) -> str:
    f0, f1, f2, f3 = state
    duration = end - start + 1
    return (
        f"- {start}-{end} ({duration} points): "
        f"{TREND_DESC.get(f0, 'unknown trend')}, "
        f"{WAVEFORM_DESC.get(f1, 'unknown waveform')}, "
        f"{STEP_DESC.get(f2, 'unknown step')}, "
        f"{MUTATION_DESC.get(f3, 'unknown mutation')}"
    )


def build_llm_prompt(curves: Sequence[CurveData], timesteps: int) -> str:
    event_text = "\n\n".join(describe_curve_events(curve, timesteps) for curve in curves)
    metric_names = ", ".join(curve.metric_name for curve in curves)
    return f"""
# Multi-metric time-series anomaly position recognition

You need to identify joint anomalies from the following metrics:
{metric_names}

Rules:
1. For negative metrics such as Max Latency, Avg Latency, Failure Count, Failure Rate,
   Retries and Timeouts, step increases and sustained local upward steps are anomalous.
2. For business metrics such as Request Count, Success Count and Success Rate, step
   decreases and sustained local downward steps are anomalous.
3. Any waveform clipping or missing cycle is anomalous.
4. Single-metric mutations are ignored. Multiple business metrics with downward
   mutation at the same timestamp are anomalous. Multiple negative metrics with
   upward mutation at the same timestamp are anomalous.
5. Local step changes lasting five or fewer timestamps are ignored.

Data:
{event_text}

Return only a Python-style integer list of anomalous timestamp indices, such as:
[10, 11, 35, 120]
""".strip()


def extract_indices_from_response(content: str, timesteps: int) -> List[int]:
    list_match = re.search(r"\[([\d,\s]+)\]", content)
    digits = re.findall(r"\b\d+\b", list_match.group(1) if list_match else content)
    indices: List[int] = []
    for digit in digits:
        value = int(digit)
        if 0 <= value < timesteps:
            indices.append(value)
    return sorted(set(indices))


def analyze_group_with_llm(
    curves: Sequence[CurveData],
    model: str,
    api_key: str,
    base_url: Optional[str],
    max_retries: int,
    retry_delay: float,
) -> Tuple[np.ndarray, List[int], Dict[str, int]]:
    timesteps = min((curve.length for curve in curves), default=0)
    if timesteps <= 0:
        return np.zeros(0, dtype=np.int8), [], {}

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("LLM mode requires the optional package: pip install openai") from exc

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    prompt = build_llm_prompt(curves, timesteps)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Only output a Python-style list of anomalous timestamp indices.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=2000,
            )
            content = response.choices[0].message.content.strip()
            indices = extract_indices_from_response(content, timesteps)
            scores = np.zeros(timesteps, dtype=np.int8)
            scores[indices] = 1
            return scores, indices, {"llm_selected_points": len(indices)}
        except Exception as exc:  # pragma: no cover - network/API dependent
            last_error = exc
            print(f"[WARN] LLM attempt {attempt + 1}/{max_retries} failed: {exc}", file=sys.stderr)
            time.sleep(retry_delay * (attempt + 1))

    raise RuntimeError(f"All LLM attempts failed. Last error: {last_error}")


def ensure_output_dirs(results_dir: Path, output_name: str) -> Dict[str, Path]:
    promote_dir = results_dir / "Promote"
    dirs = {
        "promote": promote_dir,
        "scores": promote_dir / "scores",
        "indices": promote_dir / "anomaly_indices",
        "easytsad_scores": results_dir / "Scores" / DEFAULT_METHOD / DEFAULT_SCHEMA / output_name,
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def save_group_result(
    group_name: str,
    curves: Sequence[CurveData],
    scores: np.ndarray,
    indices: List[int],
    reasons: Dict[str, int],
    output_dirs: Dict[str, Path],
) -> GroupResult:
    safe_name = safe_filename(group_name)
    score_path = output_dirs["scores"] / f"{safe_name}.npy"
    indices_path = output_dirs["indices"] / f"{safe_name}.json"
    easytsad_score_path = output_dirs["easytsad_scores"] / f"{group_name}---all_metrics.npy"

    np.save(score_path, scores.astype(np.int8))
    np.save(easytsad_score_path, scores.astype(np.int8))
    with indices_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "group": group_name,
                "curves": [curve.name for curve in curves],
                "timesteps": int(len(scores)),
                "anomaly_indices": indices,
                "segments": [{"start": start, "end": end} for start, end in contiguous_segments(indices)],
                "reasons": reasons,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return GroupResult(
        name=group_name,
        curve_count=len(curves),
        timesteps=int(len(scores)),
        scores=scores,
        indices=indices,
        segments=contiguous_segments(indices),
        reasons=reasons,
        score_path=score_path,
        indices_path=indices_path,
        easytsad_score_path=easytsad_score_path,
    )


def summarize_existing_results(results_dir: Path) -> List[Dict[str, object]]:
    eval_root = results_dir / "Evals" / DEFAULT_METHOD / DEFAULT_SCHEMA
    if not eval_root.is_dir():
        return []

    summaries: List[Dict[str, object]] = []
    for avg_path in sorted(eval_root.glob("*/avg.json")):
        try:
            with avg_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[WARN] Cannot read {avg_path}: {exc}", file=sys.stderr)
            continue

        best_f1 = data.get("best f1 under pa", {})
        event_f1 = data.get("event-based f1 under pa with mode squeeze", {})
        summaries.append(
            {
                "dataset": avg_path.parent.name,
                "best_f1": best_f1.get("f1"),
                "precision": best_f1.get("precision"),
                "recall": best_f1.get("recall"),
                "event_f1": event_f1.get("f1"),
                "event_auprc_log": data.get("event-based auprc under pa with mode log"),
                "path": str(avg_path),
            }
        )
    return summaries


def count_existing_plots(results_dir: Path) -> int:
    plot_root = results_dir / "Plots"
    if not plot_root.is_dir():
        return 0
    return sum(1 for _ in plot_root.rglob("*.pdf"))


def format_float(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "-"


def segment_preview(segments: Sequence[Tuple[int, int]], limit: int = 8) -> str:
    if not segments:
        return "none"
    pieces = [f"{start}" if start == end else f"{start}-{end}" for start, end in segments[:limit]]
    if len(segments) > limit:
        pieces.append(f"...(+{len(segments) - limit})")
    return ", ".join(pieces)


def write_summary_files(
    output_dirs: Dict[str, Path],
    classification_dir: Path,
    results_dir: Path,
    group_results: Sequence[GroupResult],
    existing_summaries: Sequence[Dict[str, object]],
    mode: str,
    overview_path: Optional[Path],
) -> Tuple[Path, Path]:
    summary_json = output_dirs["promote"] / "promote_summary.json"
    summary_md = output_dirs["promote"] / "promote_summary.md"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "classification_dir": str(classification_dir),
        "results_dir": str(results_dir),
        "overview_plot": str(overview_path) if overview_path else None,
        "groups": [
            {
                "name": result.name,
                "curve_count": result.curve_count,
                "timesteps": result.timesteps,
                "anomaly_points": len(result.indices),
                "segments": [{"start": start, "end": end} for start, end in result.segments],
                "reasons": result.reasons,
                "score_path": str(result.score_path),
                "indices_path": str(result.indices_path),
                "easytsad_score_path": str(result.easytsad_score_path),
            }
            for result in group_results
        ],
        "existing_eval_summaries": list(existing_summaries),
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = [
        "# ASE Smart Brain Result Summary",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Mode: `{mode}`",
        f"- Classification input: `{classification_dir}`",
        f"- Results output: `{results_dir}`",
    ]
    if overview_path:
        lines.append(f"- Overview plot: `{overview_path}`")
    lines.extend(
        [
            "",
            "## PROMOTE Aggregation",
            "",
            "| Group | Curves | Timesteps | Anomaly points | Segment preview | Score file |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for result in group_results:
        lines.append(
            "| {name} | {curves} | {timesteps} | {points} | {segments} | `{score}` |".format(
                name=result.name,
                curves=result.curve_count,
                timesteps=result.timesteps,
                points=len(result.indices),
                segments=segment_preview(result.segments),
                score=result.score_path,
            )
        )

    lines.extend(
        [
            "",
            "## Bundled Evaluation Results",
            "",
            "| Dataset | Best F1(PA) | Precision | Recall | Event F1(PA) | Event AUPRC(log) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if existing_summaries:
        for row in existing_summaries:
            lines.append(
                "| {dataset} | {best_f1} | {precision} | {recall} | {event_f1} | {event_auprc} |".format(
                    dataset=row["dataset"],
                    best_f1=format_float(row["best_f1"]),
                    precision=format_float(row["precision"]),
                    recall=format_float(row["recall"]),
                    event_f1=format_float(row["event_f1"]),
                    event_auprc=format_float(row["event_auprc_log"]),
                )
            )
    else:
        lines.append("| - | - | - | - | - | - |")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_json, summary_md


def render_overview_plot(group_results: Sequence[GroupResult], output_dir: Path) -> Optional[Path]:
    if not group_results:
        return None
    try:
        mpl_config_dir = output_dir / ".matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception as exc:
        print(f"[WARN] Cannot render overview plot because matplotlib is unavailable: {exc}")
        return None

    plot_path = output_dir / "promote_overview.png"
    rows = len(group_results)
    fig_height = max(2.6, 0.75 * rows + 1.2)
    fig, axes = plt.subplots(rows, 1, figsize=(12, fig_height), sharex=True)
    if rows == 1:
        axes = [axes]

    cmap = ListedColormap(["#F5F7FA", "#C62828"])
    for ax, result in zip(axes, group_results):
        ax.imshow(result.scores.reshape(1, -1), aspect="auto", interpolation="nearest", cmap=cmap)
        ax.set_yticks([])
        ax.set_ylabel(result.name, rotation=0, ha="right", va="center", labelpad=72, fontsize=9)
        ax.set_xlim(0, max(result.timesteps - 1, 1))
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.3)

    axes[-1].set_xlabel("Timestamp index")
    fig.suptitle("PROMOTE multi-metric anomaly overview", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def print_existing_results(results_dir: Path) -> None:
    summaries = summarize_existing_results(results_dir)
    plot_count = count_existing_plots(results_dir)
    print("\nBundled paper results")
    print(f"- Results directory: {results_dir}")
    print(f"- Plot PDFs: {plot_count}")
    if not summaries:
        print("- No avg.json files found under Results/Evals/A_GPT/naive")
        return

    print("\nDataset                         F1(PA)  Precision  Recall  Event-F1  Event-AUPRC")
    print("-" * 79)
    for row in summaries:
        print(
            f"{row['dataset']:<30} "
            f"{format_float(row['best_f1']):>7} "
            f"{format_float(row['precision']):>10} "
            f"{format_float(row['recall']):>7} "
            f"{format_float(row['event_f1']):>9} "
            f"{format_float(row['event_auprc_log']):>12}"
        )


def run_analysis(args: argparse.Namespace) -> int:
    project_root = discover_project_root(args.project_root)
    classification_dir = resolve_path(args.classification_dir, project_root)
    results_dir = resolve_path(args.results_dir, project_root)
    raw_data_dir = resolve_path(args.raw_data_dir, project_root) if args.raw_data_dir else None
    if raw_data_dir is not None and not raw_data_dir.exists():
        print(f"[INFO] Raw data directory not found, continuing with feature files only: {raw_data_dir}")
        raw_data_dir = None

    print(f"Project root: {project_root}")
    print(f"Classification features: {classification_dir}")
    print(f"Results directory: {results_dir}")

    curves = load_curves(
        classification_dir=classification_dir,
        raw_data_dir=raw_data_dir,
        feature_count=args.feature_count,
        feature_file=args.feature_file,
        label_file=args.label_file,
    )
    grouped = group_curves(curves, args.group_by)
    output_dirs = ensure_output_dirs(results_dir, args.output_name)

    group_results: List[GroupResult] = []
    mode = "llm" if args.use_llm else "offline-rules"

    if args.use_llm:
        if not args.api_key:
            raise RuntimeError("LLM mode requires OPENAI_API_KEY or --api-key.")
        if not args.model:
            raise RuntimeError("LLM mode requires PROMOTE_MODEL/OPENAI_MODEL or --model.")

    for group_name, group_curves_list in grouped.items():
        print(f"\nAnalyzing group `{group_name}` with {len(group_curves_list)} metric curves...")
        if args.use_llm:
            scores, indices, reasons = analyze_group_with_llm(
                curves=group_curves_list,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
            )
        else:
            scores, indices, reasons = analyze_group_offline(group_curves_list)

        result = save_group_result(
            group_name=group_name,
            curves=group_curves_list,
            scores=scores,
            indices=indices,
            reasons=reasons,
            output_dirs=output_dirs,
        )
        group_results.append(result)
        print(
            f"- anomaly points: {len(result.indices)} / {result.timesteps}; "
            f"segments: {segment_preview(result.segments, limit=5)}"
        )

    overview_path = render_overview_plot(group_results, output_dirs["promote"])
    existing_summaries = summarize_existing_results(results_dir)
    summary_json, summary_md = write_summary_files(
        output_dirs=output_dirs,
        classification_dir=classification_dir,
        results_dir=results_dir,
        group_results=group_results,
        existing_summaries=existing_summaries,
        mode=mode,
        overview_path=overview_path,
    )

    print("\nSaved outputs")
    print(f"- Summary JSON: {summary_json}")
    print(f"- Summary Markdown: {summary_md}")
    if overview_path:
        print(f"- Overview plot: {overview_path}")
    print(f"- EasyTSAD-style scores: {output_dirs['easytsad_scores']}")

    print_existing_results(results_dir)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or display ASE Smart Brain PROMOTE anomaly results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", default=os.getenv("ASE_PROJECT_ROOT"))
    parser.add_argument(
        "--classification-dir",
        default=os.getenv("ASE_CLASSIFICATION_DIR", "Classification_trend"),
        help="Folder containing feature_*/test_predictions.npy curve folders.",
    )
    parser.add_argument(
        "--raw-data-dir",
        default=os.getenv("ASE_RAW_DATA_DIR", "data"),
        help="Optional raw curve folder. Missing folders are ignored.",
    )
    parser.add_argument(
        "--results-dir",
        default=os.getenv("ASE_RESULTS_DIR", "Results"),
        help="Folder where summaries, scores and plots are read/written.",
    )
    parser.add_argument("--feature-count", type=int, default=int(os.getenv("ASE_FEATURE_COUNT", DEFAULT_FEATURE_COUNT)))
    parser.add_argument("--feature-file", default=os.getenv("ASE_FEATURE_FILE", DEFAULT_FEATURE_FILE))
    parser.add_argument("--label-file", default=os.getenv("ASE_LABEL_FILE", DEFAULT_LABEL_FILE))
    parser.add_argument("--output-name", default=os.getenv("ASE_OUTPUT_NAME", DEFAULT_OUTPUT_NAME))
    parser.add_argument("--group-by", choices=["dataset", "all"], default=os.getenv("ASE_GROUP_BY", "dataset"))
    parser.add_argument(
        "--show-existing",
        action="store_true",
        help="Only print the bundled Results/Evals and Results/Plots summary.",
    )

    llm_group = parser.add_argument_group("optional LLM mode")
    llm_group.add_argument("--use-llm", action="store_true", help="Use an OpenAI-compatible chat model.")
    llm_group.add_argument("--model", default=os.getenv("PROMOTE_MODEL") or os.getenv("OPENAI_MODEL"))
    llm_group.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    llm_group.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    llm_group.add_argument("--max-retries", type=int, default=int(os.getenv("PROMOTE_MAX_RETRIES", "3")))
    llm_group.add_argument("--retry-delay", type=float, default=float(os.getenv("PROMOTE_RETRY_DELAY", "2")))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    project_root = discover_project_root(args.project_root)
    results_dir = resolve_path(args.results_dir, project_root)

    if args.show_existing:
        print_existing_results(results_dir)
        return 0

    try:
        return run_analysis(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
