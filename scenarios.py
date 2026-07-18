# actions.py
ACTIONS = ["Observe", "Skip", "Delay", "Retask", "Downlink", "IncreaseSampling", "Emergency"]

# scenarios.py (Schema declaration & random generation bounds)
import random

MISSION_TYPES = ["Earth Observation", "Wildfire Monitoring", "Agriculture", "Disaster Response", "Ocean Monitoring"]
TEMPERATURE_STATES = ["Normal", "High", "Critical"]
PRIORITY_LEVELS = ["Low", "Medium", "High", "Critical"]

def generate_random_state():
    return {
        "mission_type": random.choice(MISSION_TYPES),
        "battery_level": random.randint(0, 100),       # %
        "memory_usage": random.randint(0, 100),        # %
        "temperature": random.choice(TEMPERATURE_STATES),
        "remaining_pass_time": random.randint(1, 30),  # minutes
        "cloud_probability": random.randint(0, 100),   # %
        "target_priority": random.choice(PRIORITY_LEVELS),
        "ground_contact": random.randint(0, 30)        # minutes
    }