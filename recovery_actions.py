"""
recovery_actions.py
====================

MissionReason's proposed fault-to-recovery-action mapping.

THIS IS NOT FROM ANDERSON ET AL. Their paper (SatFaultSim / ensemble fault
diagnosis) explicitly stops at detection + attribution -- it names a fault,
it does not decide what to do about it. Their own future-work section says:

    "A promising direction for further development includes creating
    optimization algorithms for planning and scheduling recovery actions
    when a fault is detected." (Anderson et al., 2024)

This module is MissionReason's answer to that sentence: a concrete mapping
from each of their 11 fault codes to a specific, differentiated recovery
action -- replacing the earlier, cruder version of this pipeline where every
diagnosed fault collapsed into a single generic 'Emergency' bucket. That
collapse was a real weakness: it didn't actually demonstrate "planning and
scheduling recovery actions" (plural, differentiated) as their sentence
asks for -- it demonstrated a single safety trip-wire.

Each mapping below is a simple, physically-motivated engineering judgment
call, not a citation. That is stated plainly in our paper: this taxonomy is
MissionReason's contribution, offered as one reasonable answer to an open
question their paper leaves open, not a reproduction of anything published.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecoveryAction:
    action: str          # short machine-readable recovery action label
    rationale: str        # why this action fits this fault's physical cause
    severity: str          # "critical" | "degraded" | "minor" -- feeds the utility model


# Fault code -> recovery action. Two codes (SR_C, IRF_R) are "critical" and
# keep triggering the hard safety override (Emergency) exactly as before --
# see distribution_citations.yaml for why those two were chosen as critical.
# The other nine now get a differentiated response instead of collapsing
# into the same bucket.
RECOVERY_ACTIONS = {
    "NOMINAL": RecoveryAction(
        action="None",
        rationale="No fault present.",
        severity="minor",
    ),
    "IRF_Z": RecoveryAction(
        action="Flag_For_Ground_Review",
        rationale="Radiation zeroed out one or more variables; data for this "
                   "window is unusable but not dangerous. Hold the affected "
                   "readings for ground review rather than acting on them.",
        severity="degraded",
    ),
    "IRF_S": RecoveryAction(
        action="Trigger_Sensor_Reset",
        rationale="Radiation caused stale (stuck) data. A power-cycle/reset "
                   "of the affected sensor is the standard remedy for a "
                   "stuck reading, per the fault's own description.",
        severity="degraded",
    ),
    "IRF_R": RecoveryAction(
        action="Emergency",
        rationale="Radiation caused garbage values (bit flips) -- corrupted "
                   "data risks corrupted command execution. Treated as "
                   "safety-critical (MissionReason's severity choice, see "
                   "distribution_citations.yaml).",
        severity="critical",
    ),
    "SR_C": RecoveryAction(
        action="Emergency",
        rationale="System reset corrupted the onboard clock reference -- "
                   "undermines command/telemetry timing integrity "
                   "spacecraft-wide. Treated as safety-critical.",
        severity="critical",
    ),
    "SR_N": RecoveryAction(
        action="Resync_Non_Clock_State",
        rationale="System reset jumped non-clock state, but the clock "
                   "(ground-referenced) is unaffected -- less severe than "
                   "SR_C. Resync affected subsystem state against the last "
                   "known-good values instead of a full safing.",
        severity="degraded",
    ),
    "K_L": RecoveryAction(
        action="Switch_To_Backup_Sensor",
        rationale="Sensor loss (NaN values) -- if a redundant thermal "
                   "sensor exists, switch to it; otherwise degrade "
                   "gracefully by delaying thermally-sensitive ops.",
        severity="degraded",
    ),
    "K_F": RecoveryAction(
        action="Flag_Sensor_For_Recalibration",
        rationale="Sensor failure produced abnormal/stale thermal values -- "
                   "likely a calibration or wiring fault rather than a "
                   "true thermal event. Flag for recalibration, continue "
                   "operating with reduced confidence in this channel.",
        severity="degraded",
    ),
    "K_C": RecoveryAction(
        action="Isolate_Component",
        rationale="A nearby component failure is driving abnormal thermal "
                   "drift. Electrically/thermally isolate the suspect "
                   "component if possible and delay ops that depend on it.",
        severity="degraded",
    ),
    "K_H": RecoveryAction(
        action="Reduce_Payload_Duty_Cycle",
        rationale="External temperature exceeds sensor maximum -- a real "
                   "thermal event. Reduce payload duty cycle and delay "
                   "heat-generating ops until temperature normalizes.",
        severity="degraded",
    ),
    "I_O": RecoveryAction(
        action="Reduce_Power_Draw",
        rationale="Mechanical failure causing high current draw. Reduce "
                   "non-essential power draw and monitor; escalate to "
                   "Emergency only if current continues to climb (not "
                   "modeled here -- single-snapshot decision).",
        severity="degraded",
    ),
    "I_U": RecoveryAction(
        action="Check_Power_Bus_Continuity",
        rationale="Mechanical failure causing low current draw suggests a "
                   "possible bus continuity issue. Flag for a power-bus "
                   "continuity check; not immediately safety-critical.",
        severity="degraded",
    ),
}


def get_recovery_action(fault_code: str) -> RecoveryAction:
    return RECOVERY_ACTIONS.get(
        fault_code,
        RecoveryAction(action="Flag_For_Ground_Review",
                        rationale="Unrecognized fault code -- default to "
                                   "conservative ground review.",
                        severity="degraded"),
    )
