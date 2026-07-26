#!/usr/bin/env python3
"""
generate_missionreason_v3.py
==============================

v3 adds the two fixes identified as the real gaps between this pipeline and
Anderson et al.'s (2024) stated future-work direction -- NOT a training-script
rewrite, an engine-level upgrade:

  1. RECOVERY-ACTION GRANULARITY (recovery_actions.py): every diagnosed fault
     now maps to a specific, differentiated recovery directive (e.g. K_H ->
     'Reduce_Payload_Duty_Cycle') instead of every fault collapsing into a
     single generic 'Emergency' bucket. This is what actually demonstrates
     "planning and scheduling RECOVERY ACTIONS" (plural) as Anderson et al.'s
     future-work sentence asks for.

  2. CONFIDENCE-AWARE DIAGNOSIS (diagnostic_layer_v3.py): a critical-severity
     fault label is no longer trusted unconditionally. Using Anderson et
     al.'s own published per-fault attribution accuracy (Table 4) as a proxy
     for classifier confidence, a critical label with LOW simulated
     confidence now falls through to a cautious 'Verify_Before_Emergency'
     directive instead of an immediate hard safing.

Design choice: the top-level action space stays Observe/Downlink/Delay/
Skip/Emergency (unchanged, so the balancing/eval framework built on v2 still
applies). A new `recovery_directive` field is added alongside it, populated
whenever a fault is diagnosed. This avoids a disruptive taxonomy rebuild
while still giving the "recovery actions" claim real, differentiated content.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optimization_rules as opt
import diagnostic_layer_v3 as diag
import recovery_actions as recovery

ACTIONS = ["Observe", "Downlink", "Delay", "Skip", "Emergency"]

MISSION_TYPES = list(opt.MISSION_CLOUD_SENSITIVITY.keys())
TEMPERATURE_STATES = ["Normal", "High", "Critical"]
TEMPERATURE_PROBS = [0.70, 0.20, 0.10]

PRIORITY_LEVELS = ["Low", "Medium", "High", "Critical"]
PRIORITY_PROBS = [0.35, 0.35, 0.20, 0.10]

TARGET_CLASS_FRACTIONS = {
    "Observe": 0.275,
    "Downlink": 0.275,
    "Delay": 0.225,
    "Emergency": 0.175,
}
SKIP_HARD_CAP_FRACTION = 0.05


@dataclass
class MissionState:
    mission_type: str
    battery_level: float
    memory_usage: float
    temperature: str
    remaining_pass_time: float
    cloud_probability: float
    target_priority: str
    ground_contact: float
    telemetry_diagnosis: str
    diagnosis_confidence: float


def sample_battery_level(rng: np.random.Generator) -> float:
    if rng.random() < 0.07:
        val = rng.beta(2.0, 5.0)
        return float(1.0 + val * (15.0 - 1.0))
    val = rng.beta(5.0, 2.0)
    return float(40.0 + val * (100.0 - 40.0))


def sample_cloud_probability(rng: np.random.Generator) -> float:
    return float(rng.beta(3.0, 2.0) * 100.0)


def sample_ground_contact_minutes(rng: np.random.Generator) -> float:
    if rng.random() < 0.90:
        return 0.0
    val = rng.lognormal(mean=np.log(8.0), sigma=0.5)
    return float(np.clip(val, 0.5, 20.0))


def sample_memory_usage(rng: np.random.Generator) -> float:
    return float(rng.beta(3.0, 3.0) * 100.0)


def sample_remaining_pass_time(rng: np.random.Generator) -> float:
    val = rng.lognormal(mean=np.log(10.0), sigma=0.4)
    return float(np.clip(val, 1.0, 30.0))


def sample_state(rng: np.random.Generator) -> MissionState:
    diagnosis, confidence = diag.sample_diagnosis_with_confidence(rng)
    return MissionState(
        mission_type=str(rng.choice(MISSION_TYPES)),
        battery_level=round(sample_battery_level(rng), 2),
        memory_usage=round(sample_memory_usage(rng), 2),
        temperature=str(rng.choice(TEMPERATURE_STATES, p=TEMPERATURE_PROBS)),
        remaining_pass_time=round(sample_remaining_pass_time(rng), 2),
        cloud_probability=round(sample_cloud_probability(rng), 2),
        target_priority=str(rng.choice(PRIORITY_LEVELS, p=PRIORITY_PROBS)),
        ground_contact=round(sample_ground_contact_minutes(rng), 2),
        telemetry_diagnosis=diagnosis,
        diagnosis_confidence=round(confidence, 4),
    )


# --------------------------------------------------------------------------
# Decision engine: safety override (now confidence-aware) + recovery-action
# bridging + the existing utility model (extends optimization_rules.py)
# --------------------------------------------------------------------------

def evaluate_with_recovery(state: MissionState) -> Tuple[str, Optional[str], str]:
    """Returns (action, recovery_directive_or_None, cot_reasoning)."""
    state_dict = {
        "mission_type": state.mission_type,
        "battery_level": state.battery_level,
        "memory_usage": state.memory_usage,
        "temperature": state.temperature,
        "remaining_pass_time": state.remaining_pass_time,
        "cloud_probability": state.cloud_probability,
        "target_priority": state.target_priority,
        "ground_contact": state.ground_contact,
    }

    # --- 1. Confidence-aware diagnostic-layer override ---
    trusted, confidence_note = diag.get_confidence_aware_override(
        state.telemetry_diagnosis, state.diagnosis_confidence
    )
    if trusted:
        rec = recovery.get_recovery_action(state.telemetry_diagnosis)
        reasoning = (
            "SAFETY OVERRIDE (diagnostic-layer trigger, confidence-checked)\n"
            f"Telemetry diagnosis: {state.telemetry_diagnosis} "
            f"({diag.diagnosis_description(state.telemetry_diagnosis)})\n"
            f"{confidence_note}\n"
            "---------------------\n"
            "Decision: Emergency (Safing Spacecraft)"
        )
        return "Emergency", rec.action, reasoning

    if diag.is_critical_fault(state.telemetry_diagnosis) and not trusted:
        # Low-confidence critical label -> cautious verification, not a hard
        # safing. Action falls to Delay (buy time to re-diagnose / get a
        # ground confirmation) rather than Observe/Downlink as normal.
        reasoning = (
            "DIAGNOSTIC LAYER: CRITICAL LABEL, LOW CONFIDENCE\n"
            f"Telemetry diagnosis: {state.telemetry_diagnosis} "
            f"({diag.diagnosis_description(state.telemetry_diagnosis)})\n"
            f"{confidence_note}\n"
            "---------------------\n"
            "Decision: Delay (hold non-critical ops pending verification)"
        )
        return "Delay", "Verify_Before_Emergency", reasoning

    # --- 2. Existing hard override (temperature / battery), unchanged ---
    if state.temperature == "Critical" or state.battery_level < 15:
        action, reasoning = opt.evaluate_expert_policy(state_dict)
        rec = None
        if state.telemetry_diagnosis != "NOMINAL":
            rec = recovery.get_recovery_action(state.telemetry_diagnosis)
            reasoning += (
                f"\nConcurrent diagnosis '{state.telemetry_diagnosis}' noted; "
                f"recovery directive on file: {rec.action}."
            )
        return action, (rec.action if rec else None), reasoning

    # --- 3. Non-critical fault: recovery directive + thermal bridging ---
    bridging_note = ""
    rec = None
    if state.telemetry_diagnosis != "NOMINAL":
        rec = recovery.get_recovery_action(state.telemetry_diagnosis)
        if diag.is_thermal_fault(state.telemetry_diagnosis) and state.temperature == "Normal":
            state_dict["temperature"] = "High"
            bridging_note = (
                f"\nDiagnostic-layer bridging: '{state.telemetry_diagnosis}' "
                f"({diag.diagnosis_description(state.telemetry_diagnosis)}, "
                f"confidence={state.diagnosis_confidence:.2f}) indicates a "
                "thermal anomaly the coarse temperature reading missed; "
                "treating as thermally elevated for the utility calculation."
            )
        bridging_note += (
            f"\nRecovery directive on file: {rec.action} -- {rec.rationale}"
        )

    action, reasoning = opt.evaluate_expert_policy(state_dict)
    if bridging_note:
        reasoning = reasoning + bridging_note
    return action, (rec.action if rec else None), reasoning


# --------------------------------------------------------------------------
# Balanced dataset construction
# --------------------------------------------------------------------------

def build_balanced_dataset(n_total: int, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)

    caps = {cls: int(np.ceil(frac * n_total)) for cls, frac in TARGET_CLASS_FRACTIONS.items()}
    caps["Skip"] = int(np.ceil(SKIP_HARD_CAP_FRACTION * n_total))

    counts = {cls: 0 for cls in ACTIONS}
    records: List[Dict] = []
    max_attempts = n_total * 300
    attempts = 0

    while len(records) < n_total and attempts < max_attempts:
        attempts += 1
        state = sample_state(rng)
        action, recovery_directive, reasoning = evaluate_with_recovery(state)

        cap = caps.get(action, int(np.ceil(0.30 * n_total)))
        if counts.get(action, 0) >= cap:
            continue

        counts[action] = counts.get(action, 0) + 1
        record = {
            "id": len(records),
            "scenario_id": f"MR-V3-{len(records)+1:04d}",
            "state": asdict(state),
            "ground_truth": {
                "action": action,
                "recovery_directive": recovery_directive,
                "reason": reasoning,
            },
        }
        records.append(record)

    if len(records) < n_total:
        while len(records) < n_total:
            state = sample_state(rng)
            action, recovery_directive, reasoning = evaluate_with_recovery(state)
            counts[action] = counts.get(action, 0) + 1
            record = {
                "id": len(records),
                "scenario_id": f"MR-V3-{len(records)+1:04d}",
                "state": asdict(state),
                "ground_truth": {
                    "action": action,
                    "recovery_directive": recovery_directive,
                    "reason": reasoning,
                },
            }
            records.append(record)

    rng.shuffle(records)  # type: ignore[arg-type]
    for i, r in enumerate(records):
        r["id"] = i
    return records


# --------------------------------------------------------------------------
# Chat-template formatting
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are MissionReason, an onboard autonomous decision-making assistant for a "
    "resource-constrained small satellite. You receive raw mission telemetry and "
    "the output of an upstream fault-diagnosis module, including its confidence. "
    "First check for safety-critical conditions -- treating low-confidence "
    "critical-fault labels with appropriate caution rather than blind trust -- "
    "then reason step by step about the utility of each candidate action "
    "(Observe, Downlink, Delay, Skip). If a fault is diagnosed, also name the "
    "specific recovery directive it calls for. Show your reasoning, then state "
    "the final action and any recovery directive."
)


def state_to_user_prompt(state: Dict) -> str:
    return (
        "Current mission state:\n"
        f"- Mission type: {state['mission_type']}\n"
        f"- Battery level: {state['battery_level']}%\n"
        f"- Memory usage: {state['memory_usage']}%\n"
        f"- Temperature state: {state['temperature']}\n"
        f"- Remaining pass time: {state['remaining_pass_time']} min\n"
        f"- Cloud probability: {state['cloud_probability']}%\n"
        f"- Target priority: {state['target_priority']}\n"
        f"- Ground contact remaining: {state['ground_contact']} min\n"
        f"- Telemetry diagnosis: {state['telemetry_diagnosis']} "
        f"(confidence: {state['diagnosis_confidence']})\n\n"
        "What action should the spacecraft take, and what recovery directive "
        "(if any) applies? Reason step by step, then state both."
    )


def record_to_chat_format(record: Dict) -> Dict:
    gt = record["ground_truth"]
    directive_line = f"\nRecovery Directive: {gt['recovery_directive']}" if gt["recovery_directive"] else ""
    assistant_content = f"{gt['reason']}\n\nFinal Action: {gt['action']}{directive_line}"
    return {
        "id": record["id"],
        "scenario_id": record["scenario_id"],
        "action": gt["action"],
        "recovery_directive": gt["recovery_directive"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state_to_user_prompt(record["state"])},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the MissionReason v3 benchmark.")
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="./missionreason_v3_dataset")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = build_balanced_dataset(n_total=args.n, seed=args.seed)

    raw_path = os.path.join(args.out_dir, "missionreason_v3_raw.json")
    with open(raw_path, "w") as f:
        json.dump(records, f, indent=2)

    chat_path = os.path.join(args.out_dir, "missionreason_v3_chat.jsonl")
    with open(chat_path, "w") as f:
        for record in records:
            f.write(json.dumps(record_to_chat_format(record)) + "\n")

    df = pd.DataFrame(records)
    class_counts = df["ground_truth"].apply(lambda g: g["action"]).value_counts().to_dict()
    class_fractions = {k: round(v / len(df), 4) for k, v in class_counts.items()}
    directive_counts = (
        df["ground_truth"].apply(lambda g: g["recovery_directive"]).value_counts(dropna=True).to_dict()
    )
    low_conf_critical = sum(
        1 for r in records
        if diag.is_critical_fault(r["state"]["telemetry_diagnosis"])
        and r["state"]["diagnosis_confidence"] < diag.CONFIDENCE_TRUST_THRESHOLD
    )

    summary = {
        "n_samples": len(records),
        "seed": args.seed,
        "action_class_counts": class_counts,
        "action_class_fractions": class_fractions,
        "recovery_directive_counts": directive_counts,
        "low_confidence_critical_faults": low_conf_critical,
        "citation_manifest": "distribution_citations.yaml",
        "outputs": {"raw_json": raw_path, "chat_jsonl": chat_path},
    }
    summary_path = os.path.join(args.out_dir, "missionreason_v3_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Generated {len(records)} samples -> {args.out_dir}")
    print("Action class balance:")
    for cls in ACTIONS:
        frac = class_fractions.get(cls, 0.0)
        print(f"  {cls:>10s}: {class_counts.get(cls, 0):5d}  ({frac*100:.1f}%)")
    print(f"\nRecovery directives issued: {sum(directive_counts.values())}")
    print(f"Low-confidence critical-fault cases (caught, not blind-trusted): {low_conf_critical}")
    print(f"\nFiles written:\n  {raw_path}\n  {chat_path}\n  {summary_path}")


if __name__ == "__main__":
    main()
