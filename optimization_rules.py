import math

# ==========================================
# CONSTANTS & POLICY PARAMETERS (Suggestion 2 & 4)
# ==========================================
PRIORITY_VALUES = {
    "Critical": 50,
    "High": 30,
    "Medium": 10,
    "Low": 0,
}

BATTERY_SCORES = {
    "Critical": -40,  # below 30%
    "Low": -15,       # 30% to 40%
    "Healthy": 10     # >= 40%
}

OBSERVE_THRESHOLD = 15.0  # (Suggestion 4)

# Mission-specific cloud sensitivity multipliers (Suggestion 6)
MISSION_CLOUD_SENSITIVITY = {
    "Earth Observation": 1.0,   # Highly sensitive
    "Agriculture": 1.2,         # Extremely sensitive to ground visibility
    "Ocean Monitoring": 0.5,    # Moderately sensitive
    "Disaster Response": 0.3,   # Very low sensitivity (must observe regardless)
    "Wildfire Monitoring": 0.0, # Uses Infrared (completely unaffected by clouds)
}


# ==========================================
# UTILITY HELPER FUNCTIONS
# ==========================================
def get_cloud_penalty(cloud_prob: float, mission_type: str) -> float:
    """Calculates non-linear cloud penalty based on mission profile (Suggestion 3 & 6)"""
    # Non-linear scaling: clouds under 50% have minimal impact; over 80% is devastating
    if cloud_prob < 50:
        base_penalty = (cloud_prob / 50.0) * 5.0  # max 5 pts
    elif cloud_prob <= 80:
        base_penalty = 5.0 + ((cloud_prob - 50.0) / 30.0) * 20.0  # max 25 pts
    else:
        base_penalty = 25.0 + ((cloud_prob - 80.0) / 20.0) * 35.0  # max 60 pts
        
    multiplier = MISSION_CLOUD_SENSITIVITY.get(mission_type, 1.0)
    return base_penalty * multiplier


def get_battery_state(level: float) -> str:
    if level < 30:
        return "Critical"
    elif level < 40:
        return "Low"
    return "Healthy"


# ==========================================
# CORE POLICY ENGINE
# ==========================================
def evaluate_expert_policy(state: dict) -> tuple[str, str]:
    # Hard safety constraints bypass utility calculations for spacecraft survival
    if state["temperature"] == "Critical" or state["battery_level"] < 15:
        reasoning = (
            "SAFETY OVERRIDE\n"
            f"Temperature: {state['temperature']}\n"
            f"Battery Level: {state['battery_level']}%\n"
            "---------------------\n"
            "Decision: Emergency (Safing Spacecraft)"
        )
        return "Emergency", reasoning

    # --- 1. Gather Variables ---
    mission = state["mission_type"]
    priority_score = PRIORITY_VALUES.get(state["target_priority"], 0)
    
    bat_state = get_battery_state(state["battery_level"])
    battery_score = BATTERY_SCORES[bat_state]
    
    cloud_penalty = get_cloud_penalty(state["cloud_probability"], mission)
    thermal_penalty = 25.0 if state["temperature"] == "High" else 0.0

    # --- 2. Calculate Multi-Action Utilities (Suggestion 1 & 7) ---
    
    # Action A: Science Utility (Observation)
    science_utility = priority_score + battery_score - cloud_penalty - thermal_penalty
    
    # Action B: Downlink Utility
    # Driven heavily by buffer pressure (memory_usage) and having active ground_contact
    memory_pressure = (state["memory_usage"] / 100.0) * 40.0
    ground_contact_bonus = 20.0 if state["ground_contact"] > 0 else -50.0
    downlink_utility = memory_pressure + ground_contact_bonus + (battery_score * 0.5)

    # Action C: Delay Utility
    # Highly attractive if battery is low or temp is high, giving systems time to cycle/recharge
    low_battery_bonus = 30.0 if bat_state in ["Critical", "Low"] else 0.0
    thermal_recovery_bonus = 25.0 if state["temperature"] == "High" else 0.0
    delay_utility = low_battery_bonus + thermal_recovery_bonus

    # Action D: Skip Utility
    # Constant baseline; if science or resource management doesn't clear this floor, we do nothing
    skip_utility = 5.0 

    # --- 3. Run Optimization (argmax) ---
    utilities = {
        "Observe": science_utility if science_utility >= OBSERVE_THRESHOLD else -999.0, # Must clear gate
        "Downlink": downlink_utility,
        "Delay": delay_utility,
        "Skip": skip_utility
    }
    
    best_action = max(utilities, key=utilities.get)

    # --- 4. Build Structured Numerical Explanation (Suggestion 5) ---
    explanation = (
        f"--- Calculated Utilities ---\n"
        f"Observe (Science Utility): {science_utility:.1f} (Threshold: {OBSERVE_THRESHOLD})\n"
        f"  ├─ Priority Weight: +{priority_score}\n"
        f"  ├─ Battery Allocation: {battery_score:+}\n"
        f"  ├─ Cloud Penalty: -{cloud_penalty:.1f}\n"
        f"  └─ Thermal Penalty: -{thermal_penalty:.1f}\n"
        f"Downlink Utility: {downlink_utility:.1f}\n"
        f"  ├─ Memory Pressure: +{memory_pressure:.1f}\n"
        f"  └─ Ground Link State: {ground_contact_bonus:+}\n"
        f"Delay Utility: {delay_utility:.1f}\n"
        f"  ├─ Low Battery Recovery Bonus: +{low_battery_bonus}\n"
        f"  └─ Thermal Cycling Bonus: +{thermal_recovery_bonus}\n"
        f"Skip Utility (Baseline): {skip_utility:.1f}\n"
        f"---------------------\n"
        f"Selected Action: {best_action} (Max Utility)"
    )

    return best_action, explanation