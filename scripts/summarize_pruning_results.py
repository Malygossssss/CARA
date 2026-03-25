import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_metrics(payload):
    if not payload:
        return {}
    metrics = payload.get("metrics", {})
    flat = {}
    if "semseg" in metrics and "mIoU" in metrics["semseg"]:
        flat["semseg_mIoU"] = metrics["semseg"]["mIoU"]
    if "human_parts" in metrics and "mIoU" in metrics["human_parts"]:
        flat["human_parts_mIoU"] = metrics["human_parts"]["mIoU"]
    if "sal" in metrics and "mIoU" in metrics["sal"]:
        flat["sal_mIoU"] = metrics["sal"]["mIoU"]
    if "normals" in metrics and "mean" in metrics["normals"]:
        flat["normals_mean"] = metrics["normals"]["mean"]
    if "depth" in metrics and "rmse" in metrics["depth"]:
        flat["depth_rmse"] = metrics["depth"]["rmse"]
    if "edge" in metrics:
        if "odsF" in metrics["edge"]:
            flat["edge_odsF"] = metrics["edge"]["odsF"]
        elif "loss" in metrics["edge"]:
            flat["edge_loss"] = metrics["edge"]["loss"]
    if "loss" in payload:
        flat["eval_loss"] = payload["loss"]
    return flat


def collect_rows(results_root):
    rows = []
    for summary_path in results_root.rglob("experiment_summary.json"):
        summary = load_json(summary_path)
        prompt_stats = summary.get("prompt_statistics_after", {})
        selected_eval = summary.get("eval_after_recovery") or summary.get("eval_before_recovery") or {}
        row = {
            "experiment": summary.get("experiment_name", summary_path.parent.name),
            "stage": summary.get("experiment_name", "").split("_")[2] if "_" in summary.get("experiment_name", "") else "",
            "importance_type": summary.get("importance_type"),
            "ga_aware": summary.get("importance_type") == "ga",
            "recovery": summary.get("run_recovery", False),
            "prune_ratio": summary.get("prune_ratio"),
            "total_keep_ratio": prompt_stats.get("total_keep_ratio"),
            "effective_prompt_params": prompt_stats.get("effective_prompt_params"),
            "effective_total_params": prompt_stats.get("effective_total_params"),
            "output_dir": str(summary_path.parent),
        }
        row.update(flatten_metrics(selected_eval))
        rows.append(row)
    return sorted(rows, key=lambda item: item["experiment"])


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    if not rows:
        return
    headers = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row.get(header, "")) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize UniPoRA prompt pruning experiments.")
    parser.add_argument("--results-root", type=Path, required=True, help="Root directory containing experiment outputs.")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--md-out", type=Path, default=None, help="Optional Markdown output path.")
    args = parser.parse_args()

    rows = collect_rows(args.results_root)
    if not rows:
        raise SystemExit(f"No experiment_summary.json files found under {args.results_root}")

    if args.csv_out:
        write_csv(args.csv_out, rows)
    if args.md_out:
        write_markdown(args.md_out, rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
