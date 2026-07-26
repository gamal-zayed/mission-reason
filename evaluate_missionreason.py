#!/usr/bin/env python3
"""
evaluate_missionreason.py
============================

Two modes:

  --mode dataset   Analyze the generated dataset itself (works right now,
                    no trained model needed). Produces a per-fault-code
                    recovery-directive table styled like Anderson et al.'s
                    Table 4, and an action-class breakdown.

  --mode compare    Compare a trained model's predictions against the rule-
                    engine ground truth: confusion matrix, per-class
                    precision/recall/F1, class-balanced accuracy. Requires a
                    predictions JSONL (see --predictions) with one line per
                    scenario: {"scenario_id": ..., "predicted_action": ...,
                    "predicted_directive": ... or null}.

Usage:
    python evaluate_missionreason.py --mode dataset \
        --raw-json ./missionreason_v3_dataset/missionreason_v3_raw.json

    python evaluate_missionreason.py --mode compare \
        --raw-json ./missionreason_v3_dataset/missionreason_v3_raw.json \
        --predictions ./model_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

ACTIONS = ["Observe", "Downlink", "Delay", "Skip", "Emergency"]


def load_raw(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Mode 1: dataset self-analysis (no model required)
# --------------------------------------------------------------------------

def analyze_dataset(records: List[Dict]) -> Dict:
    df = pd.DataFrame(records)
    df["action"] = df["ground_truth"].apply(lambda g: g["action"])
    df["directive"] = df["ground_truth"].apply(lambda g: g["recovery_directive"])
    df["fault"] = df["state"].apply(lambda s: s["telemetry_diagnosis"])
    df["confidence"] = df["state"].apply(lambda s: s.get("diagnosis_confidence"))

    action_counts = df["action"].value_counts().to_dict()

    # Per-fault-code table, styled like Anderson et al. Table 4: for each
    # fault code, what fraction of the time does it produce a NON-null,
    # fault-specific recovery directive (rather than being swallowed by an
    # unrelated override)? This is the "does the diagnostic layer actually
    # do differentiated work" metric, not an accuracy metric (there's no
    # ground truth to be "accurate" against here -- the rule engine defines
    # its own ground truth by construction; see limitations note below).
    fault_table = []
    for fault_code, group in df[df["fault"] != "NOMINAL"].groupby("fault"):
        n = len(group)
        directive_counts = group["directive"].value_counts().to_dict()
        top_directive = max(directive_counts, key=directive_counts.get) if directive_counts else None
        top_directive_rate = directive_counts.get(top_directive, 0) / n if n else 0.0
        fault_table.append({
            "fault_code": fault_code,
            "n_occurrences": n,
            "mean_confidence": round(float(group["confidence"].mean()), 4),
            "dominant_directive": top_directive,
            "dominant_directive_rate": round(top_directive_rate, 4),
            "action_breakdown": group["action"].value_counts().to_dict(),
        })
    fault_table.sort(key=lambda r: r["fault_code"])

    return {
        "n_samples": len(df),
        "action_class_counts": action_counts,
        "per_fault_table": fault_table,
    }


def print_dataset_report(report: Dict) -> None:
    print("=" * 70)
    print(f"DATASET SELF-ANALYSIS  (n={report['n_samples']})")
    print("=" * 70)
    print("\nAction class balance:")
    for a in ACTIONS:
        c = report["action_class_counts"].get(a, 0)
        print(f"  {a:>10s}: {c:5d}  ({c/report['n_samples']*100:.1f}%)")

    print("\nPer-fault-code table (style: Anderson et al. Table 4):")
    print(f"  {'Fault':<8} {'N':>5} {'Mean Conf':>10} {'Dominant Directive':<28} {'Rate':>6}")
    for row in report["per_fault_table"]:
        print(f"  {row['fault_code']:<8} {row['n_occurrences']:>5} "
              f"{row['mean_confidence']:>10.4f} {str(row['dominant_directive']):<28} "
              f"{row['dominant_directive_rate']*100:>5.1f}%")

    print("\nLIMITATION TO STATE IN THE PAPER: this table shows the rule "
          "engine's own internal consistency (it is 100% 'accurate' against "
          "itself by construction -- it IS the ground truth). It is NOT "
          "evidence the fine-tuned model reproduces this behavior. Run "
          "--mode compare against real model predictions for that claim.")


# --------------------------------------------------------------------------
# Mode 2: model-vs-rule-engine comparison (requires predictions file)
# --------------------------------------------------------------------------

def load_predictions(path: str) -> Dict[str, Dict]:
    preds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            preds[obj["scenario_id"]] = obj
    return preds


def confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> pd.DataFrame:
    idx = {l: i for i, l in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            mat[idx[t], idx[p]] += 1
    return pd.DataFrame(mat, index=[f"true_{l}" for l in labels],
                          columns=[f"pred_{l}" for l in labels])


def per_class_prf(y_true: List[str], y_pred: List[str], labels: List[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"action": label, "precision": round(precision, 4),
                      "recall": round(recall, 4), "f1": round(f1, 4),
                      "support": tp + fn})
    return pd.DataFrame(rows)


def class_balanced_accuracy(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    """Mean of per-class recall -- the right accuracy metric given the label
    skew (raw accuracy would be dominated by the majority classes)."""
    recalls = []
    for label in labels:
        support = sum(1 for t in y_true if t == label)
        if support == 0:
            continue
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        recalls.append(tp / support)
    return float(np.mean(recalls)) if recalls else 0.0


def compare_to_model(records: List[Dict], predictions: Dict[str, Dict]) -> Dict:
    y_true, y_pred = [], []
    directive_true, directive_pred = [], []
    missing = 0
    for r in records:
        sid = r["scenario_id"]
        if sid not in predictions:
            missing += 1
            continue
        y_true.append(r["ground_truth"]["action"])
        y_pred.append(predictions[sid]["predicted_action"])
        if r["ground_truth"]["recovery_directive"] is not None:
            directive_true.append(r["ground_truth"]["recovery_directive"])
            directive_pred.append(predictions[sid].get("predicted_directive"))

    cm = confusion_matrix(y_true, y_pred, ACTIONS)
    prf = per_class_prf(y_true, y_pred, ACTIONS)
    bal_acc = class_balanced_accuracy(y_true, y_pred, ACTIONS)
    raw_acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0

    directive_acc = None
    if directive_true:
        directive_acc = sum(1 for t, p in zip(directive_true, directive_pred) if t == p) / len(directive_true)

    return {
        "n_matched": len(y_true),
        "n_missing_predictions": missing,
        "raw_accuracy": round(raw_acc, 4),
        "class_balanced_accuracy": round(bal_acc, 4),
        "recovery_directive_accuracy": round(directive_acc, 4) if directive_acc is not None else None,
        "confusion_matrix": cm,
        "per_class_prf": prf,
    }


def print_compare_report(result: Dict) -> None:
    print("=" * 70)
    print(f"MODEL vs. RULE-ENGINE COMPARISON  (n_matched={result['n_matched']}, "
          f"missing={result['n_missing_predictions']})")
    print("=" * 70)
    print(f"\nRaw accuracy:              {result['raw_accuracy']*100:.2f}%")
    print(f"Class-balanced accuracy:   {result['class_balanced_accuracy']*100:.2f}%  "
          f"(use THIS one given the label skew, not raw accuracy)")
    if result["recovery_directive_accuracy"] is not None:
        print(f"Recovery directive accuracy: {result['recovery_directive_accuracy']*100:.2f}%")
    print("\nConfusion matrix:")
    print(result["confusion_matrix"].to_string())
    print("\nPer-class precision/recall/F1:")
    print(result["per_class_prf"].to_string(index=False))
    print("\nNOTE: report class-balanced accuracy as your headline number, not "
          "raw accuracy -- with a 27.5/27.5/22.5/17.5/5% label split, a model "
          "that only ever predicts the majority classes could still post a "
          "misleadingly high raw accuracy.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the MissionReason dataset/model.")
    parser.add_argument("--mode", choices=["dataset", "compare"], required=True)
    parser.add_argument("--raw-json", type=str, required=True)
    parser.add_argument("--predictions", type=str, default=None,
                          help="Required for --mode compare: JSONL with "
                               "{scenario_id, predicted_action, predicted_directive}")
    parser.add_argument("--out", type=str, default=None,
                          help="Optional path to write the report as JSON.")
    args = parser.parse_args()

    records = load_raw(args.raw_json)

    if args.mode == "dataset":
        report = analyze_dataset(records)
        print_dataset_report(report)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2, default=str)
    else:
        if not args.predictions:
            parser.error("--mode compare requires --predictions")
        preds = load_predictions(args.predictions)
        result = compare_to_model(records, preds)
        print_compare_report(result)
        if args.out:
            serializable = {k: (v.to_dict() if isinstance(v, pd.DataFrame) else v)
                              for k, v in result.items()}
            with open(args.out, "w") as f:
                json.dump(serializable, f, indent=2, default=str)


if __name__ == "__main__":
    main()
