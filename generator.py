# generator.py
import json
from scenarios import generate_random_state
#from rules import evaluate_rules
#from score_based_rules import evaluate_rules
from optimization_rules import evaluate_expert_policy

def main():
    dataset = []
    
    for i in range(100):
        state = generate_random_state()
        action, reason = evaluate_expert_policy(state)
        
        scenario_entry = {
            "scenario_id": f"MR-2026-{i+1:03d}",
            "state": state,
            "ground_truth": {
                "action": action,
                "reason": reason
            }
        }
        dataset.append(scenario_entry)

    ###############################
    from collections import Counter

    actions = [scenario["ground_truth"]["action"] for scenario in dataset]
    counts = Counter(actions)

    print("\n" + "="*40)
    print("📊 EXPERT POLICY DATASET CHARACTERIZATION")
    print("="*40)
    for action, count in counts.items():
        pct = (count / len(actions)) * 100
        bar = "█" * int(pct // 5)
        print(f"{action:<12} | {count:>3} ({pct:>5.1f}%) {bar}")
    print("="*40)
    ##################################

    with open("data/scenarios.json", "w") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Successfully generated 100 scenarios in data/scenarios.json")

if __name__ == "__main__":
    main()