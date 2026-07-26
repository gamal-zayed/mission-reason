"""
diagnostic_layer_v3.py
========================

Extends diagnostic_layer.py with a simulated classifier CONFIDENCE score per
diagnosis, so the decision engine can distinguish "diagnosed with high
confidence" from "diagnosed but shaky" -- instead of treating every
diagnosis label as ground truth the instant it's emitted.

IMPORTANT HONESTY NOTE (reflected in our paper, not just the code comment):
Anderson et al. publish per-fault ATTRIBUTION ACCURACY (Table 4) -- the rate
at which their classifier's label matches the true fault -- not a per-sample
CONFIDENCE score. Those are different things: accuracy is measured after the
fact against ground truth; confidence is the classifier's own self-reported
certainty on a single sample, which their paper does not report. What we do
here is use their published per-fault accuracy as the CENTER of a simulated
confidence distribution for that fault type -- a reasonable proxy (a
classifier that is right 99.9% of the time on a fault type is plausibly also
usually confident about it), but explicitly a proxy, not a reproduction of a
number they measured. This is said plainly as it goes in our paper.

Table 4 source (Anderson, Muesing, Cahoy, Center, SmallSat 2024):
    IRF_Z 99.98%   IRF_S 99.98%   IRF_R 99.90%   SR_C 100.0%   SR_N 100.0%
    K_L   99.98%   K_F   99.96%   K_C   99.93%   K_H   99.96%
    I_O   100.0%   I_U   99.96%
"""

from __future__ import annotations

import numpy as np

#import diagnostic_layer as base
import diagnostic_layer_corrected as base


# Re-export the base module's taxonomy/sampling so callers only need one import.
FAULT_TAXONOMY = base.FAULT_TAXONOMY
sample_diagnosis = base.sample_diagnosis
is_critical_fault = base.is_critical_fault
is_thermal_fault = base.is_thermal_fault
diagnosis_description = base.diagnosis_description

# Anderson et al. Table 4 -- per-fault ATTRIBUTION ACCURACY (see module
# docstring: used here as a proxy CENTER for simulated confidence, not the
# same quantity).
_TABLE4_ATTRIBUTION_ACCURACY = {
    "NOMINAL": 0.9995,  # not in Table 4 (no-fault case); kept very high, assumed
    "IRF_Z": 0.9998,
    "IRF_S": 0.9998,
    "IRF_R": 0.9990,
    "SR_C": 1.0000,
    "SR_N": 1.0000,
    "K_L": 0.9998,
    "K_F": 0.9996,
    "K_C": 0.9993,
    "K_H": 0.9996,
    "I_O": 1.0000,
    "I_U": 0.9996,
}

# Confidence threshold below which the decision engine should NOT blindly
# trust a critical-fault label -- see get_confidence_aware_action() below.
# This threshold is a MissionReason engineering choice, not a cited value.
CONFIDENCE_TRUST_THRESHOLD = 0.90


def sample_confidence(rng: np.random.Generator, fault_code: str) -> float:
    """
    Simulate a per-sample classifier confidence score for a diagnosed fault,
    centered on Anderson et al.'s published per-fault attribution accuracy
    (see module docstring for the accuracy-vs-confidence caveat).

    Uses a Beta distribution concentrated tightly around the accuracy value,
    so confidence is usually high (matching their near-perfect published
    performance) but occasionally dips -- giving the decision engine
    something real to condition on rather than always trusting the label.
    """
    center = _TABLE4_ATTRIBUTION_ACCURACY.get(fault_code, 0.995)
    # High concentration (kappa) keeps the distribution tight around `center`
    # -- occasional low draws are rare, matching a near-perfect classifier.
    kappa = 60.0
    alpha = center * kappa
    beta = (1.0 - center) * kappa
    # Guard against degenerate Beta params when center is exactly 1.0
    alpha = max(alpha, 0.5)
    beta = max(beta, 0.5)
    return float(np.clip(rng.beta(alpha, beta), 0.0, 1.0))


def sample_diagnosis_with_confidence(rng: np.random.Generator):
    """Draw a (fault_code, confidence) pair for one telemetry snapshot."""
    code = sample_diagnosis(rng)
    confidence = sample_confidence(rng, code)
    return code, confidence


def get_confidence_aware_override(fault_code: str, confidence: float) -> tuple[bool, str]:
    """
    Whether this diagnosis should trigger the hard safety override, now
    conditioned on confidence rather than the label alone.

    A critical-fault label (SR_C, IRF_R) with LOW confidence does not
    blind-trigger Emergency -- it triggers a cautious intermediate action
    instead, since acting on an uncertain "corrupted clock" or "corrupted
    command" call could itself be the wrong move. This closes a real gap in
    the earlier version of this pipeline, where any critical-fault label
    was trusted unconditionally regardless of how the diagnosis was made.
    """
    if not is_critical_fault(fault_code):
        return False, ""

    if confidence >= CONFIDENCE_TRUST_THRESHOLD:
        return True, (
            f"Diagnosis '{fault_code}' is a critical-severity fault code "
            f"with high simulated confidence ({confidence:.2f} >= "
            f"{CONFIDENCE_TRUST_THRESHOLD:.2f}) -- trusted, hard safety "
            f"override triggered."
        )
    return False, (
        f"Diagnosis '{fault_code}' is a critical-severity fault code but "
        f"with LOW simulated confidence ({confidence:.2f} < "
        f"{CONFIDENCE_TRUST_THRESHOLD:.2f}) -- NOT blindly trusted. Falling "
        f"through to Verify_Before_Emergency instead of an immediate hard "
        f"safing (MissionReason's confidence-aware bridging logic)."
    )
