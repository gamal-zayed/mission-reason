import json
import random
from typing import Dict, Any, List
#### This file is under test
class SatelliteExpertPolicyV02:
    """
    v0.2 Optimization & Expert Policy Engine.
    Translates satellite telemetry and environmental factors into a structured,
    reasoned control action with full CoT evaluation for model fine-tuning.
    """
    
    PAYLOAD_PROPERTIES = {
        "Optical Imagery": {"cloud_sensitive": True, "power_draw": 25.0},
        "Synthetic Aperture Radar (SAR)": {"cloud_sensitive": False, "power_draw": 40.0},
        "Thermal Infrared": {"cloud_sensitive": False, "power_draw": 15.0},
        "Multispectral": {"cloud_sensitive": True, "power_draw": 20.0}
    }

    MISSION_BASE_PRIORITY = {
        "Earth Observation": 25,
        "Agriculture": 30,
        "Ocean Monitoring": 20,
        "Disaster Response": 50,
        "Wildfire Monitoring": 45
    }

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def _calculate_cloud_penalty(self, cloud_prob: float, payload_type: str) -> float:
        """Non-linear decay for optical sensors under cloud cover."""
        if not self.PAYLOAD_PROPERTIES[payload_type]["cloud_sensitive"]:
            return 0.0  # SAR/Thermal penetrate clouds
            
        if cloud_prob < 50:
            return (cloud_prob / 50.0) * 5.0
        elif cloud_prob <= 80:
            return 5.0 + ((cloud_prob - 50.0) / 30.0) * 20.0
        else:
            return 25.0 + ((cloud_prob - 80.0) / 20.0) * 35.0

    def evaluate_scenario(self, scenario_id: int, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        # Extract variables
        battery = telemetry["battery_level_pct"]
        temp = telemetry["thermal_status"]
        mem_pct = (telemetry["memory_used_gb"] / telemetry["memory_capacity_gb"]) * 100
        ground_contact = telemetry["ground_station_visible"]
        target_pri = telemetry["target_priority_raw"]
        cloud_prob = telemetry["cloud_cover_prob_pct"]
        payload = telemetry["sensor_payload"]
        mission = telemetry["mission_type"]

        cot_steps = []
        
        # Step 1: Safety Gate Check
        cot_steps.append(f"1. Safety Check: Battery={battery}%, Temp='{temp}'.")
        if temp == "Critical" or battery < 15.0:
            reason = f"EMERGENCY OVERRIDE: Critical constraints triggered (Temp={temp}, Battery={battery}%)."
            cot_steps.append(f"2. Emergency triggered. System must default to Emergency Safing Mode.")
            return self._format_output(scenario_id, telemetry, "Emergency", 100.0, {"Emergency": 100.0}, cot_steps, reason)

        # Step 2: Compute Action Utilities
        cot_steps.append("2. Evaluating Multi-Utility Functions across viable actions:")
        
        # A. OBSERVE
        base_pri = self.MISSION_BASE_PRIORITY.get(mission, 20) + (target_pri * 5)
        cloud_pen = self._calculate_cloud_penalty(cloud_prob, payload)
        pwr_pen = (100.0 - battery) * 0.2
        observe_u = max(0.0, base_pri - cloud_pen - pwr_pen)
        cot_steps.append(f"   - Observe: Base Pri({base_pri}) - Cloud Pen({cloud_pen:.1f}) - Power Pen({pwr_pen:.1f}) = {observe_u:.2f}")

        # B. DOWNLINK
        downlink_u = 0.0
        if ground_contact:
            mem_reward = (mem_pct / 100.0) * 50.0
            downlink_u = 30.0 + mem_reward
            cot_steps.append(f"   - Downlink: Contact Active (+30.0) + Memory Reward({mem_reward:.1f}) = {downlink_u:.2f}")
        else:
            cot_steps.append("   - Downlink: Ineligible (No Ground Station Visibility). Score = 0.0")

        # C. DELAY
        delay_u = 5.0
        if battery < 40.0:
            delay_u += (40.0 - battery) * 1.5
        if temp == "High":
            delay_u += 20.0
        cot_steps.append(f"   - Delay/Charge: Base (5.0) + Thermal/Recharge Boost = {delay_u:.2f}")

        # D. SKIP
        skip_u = 2.0
        cot_steps.append(f"   - Skip: Ground Floor = {skip_u:.2f}")

        utilities = {
            "Observe": round(observe_u, 2),
            "Downlink": round(downlink_u, 2),
            "Delay": round(delay_u, 2),
            "Skip": round(skip_u, 2)
        }

        # Step 3: Selection
        best_action = max(utilities, key=utilities.get)
        best_score = utilities[best_action]
        
        cot_steps.append(f"3. Decision Selection: '{best_action}' achieved highest utility ({best_score:.2f}).")
        
        explanation = f"Selected {best_action} with optimal score of {best_score:.2f} based on trade-off matrix."
        return self._format_output(scenario_id, telemetry, best_action, best_score, utilities, cot_steps, explanation)

    def _format_output(self, scenario_id: int, telemetry: Dict[str, Any], action: str, score: float, 
                       utilities: Dict[str, float], cot: List[str], explanation: str) -> Dict[str, Any]:
        
        # Formatted in standard LLM Instruction (SFT) OpenAI format
        return {
            "scenario_id": f"SCN-2026-{scenario_id:04d}",
            "messages": [
                {
                    "role": "system",
                    "content": "You are the Autonomous Flight Decision Policy Engine (v0.2) onboard an Earth Observation Satellite. Analyze telemetry vectors and select optimal actions while prioritizing spacecraft health."
                },
                {
                    "role": "user",
                    "content": f"Telemetry State Vector:\n{json.dumps(telemetry, indent=2)}"
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "thought_process": cot,
                        "action_utility_matrix": utilities,
                        "selected_action": action,
                        "confidence_score": score,
                        "summary_reasoning": explanation
                    }, indent=2)
                }
            ],
            "raw_metadata": {
                "telemetry": telemetry,
                "selected_action": action,
                "utility_matrix": utilities
            }
        }

def generate_random_telemetry() -> Dict[str, Any]:
    return {
        "orbit_type": random.choice(["LEO", "SSO", "GEO"]),
        "mission_type": random.choice(["Earth Observation", "Agriculture", "Disaster Response", "Wildfire Monitoring"]),
        "sensor_payload": random.choice(["Optical Imagery", "Synthetic Aperture Radar (SAR)", "Thermal Infrared"]),
        "battery_level_pct": round(random.uniform(10.0, 100.0), 1),
        "thermal_status": random.choice(["Nominal", "Nominal", "Nominal", "High", "Critical"]),
        "memory_used_gb": round(random.uniform(5.0, 128.0), 1),
        "memory_capacity_gb": 128.0,
        "ground_station_visible": random.choice([True, False]),
        "cloud_cover_prob_pct": round(random.uniform(0.0, 100.0), 1),
        "target_priority_raw": random.randint(1, 5)
    }

if __name__ == "__main__":
    engine = SatelliteExpertPolicyV02(seed=2026)
    dataset = []

    print("Generating v0.2 AI-Training Dataset...")
    for i in range(1, 101):
        telemetry = generate_random_telemetry()
        sample = engine.evaluate_scenario(i, telemetry)
        dataset.append(sample)

    output_file = "spacecraft_action_dataset_v0.2.json"
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Successfully exported 100 rich CoT scenarios to '{output_file}'!")
