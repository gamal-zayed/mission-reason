# MissionReason

## Towards Resource-Constrained Language Models for Onboard Mission Decision Support in Autonomous Spacecraft

MissionReason is an open research project that investigates **onboard mission decision support** for **resource-constrained autonomous spacecraft**.

The current version introduces an initial expert-policy framework that models mission-level operational reasoning using structured spacecraft mission states. Rather than focusing on perception tasks such as image classification or object detection, MissionReason explores how expert operational policies can be represented as structured decision-making problems, providing a foundation for evaluating lightweight language models and other resource-efficient artificial intelligence techniques for future onboard autonomy.

---

## Motivation

Future autonomous spacecraft are expected to perform increasingly sophisticated onboard decision-making while operating under strict limitations in computation, power, memory, and communication opportunities.

Although public datasets describing Earth observation, spacecraft telemetry, and space weather are widely available, publicly available datasets that directly associate spacecraft operational states with expert mission decisions remain limited. This makes it difficult to systematically investigate onboard mission reasoning using modern artificial intelligence methods.

MissionReason addresses this challenge by providing:

* A structured spacecraft mission-state representation.
* An expert utility-based operational policy.
* An automated synthetic benchmark generator.
* Interpretable decision explanations.
* A reproducible framework for evaluating lightweight AI methods for onboard mission decision support.

---

## Current Project Status

**Current Version:** v0.1

The current Version 0.1 expert policy produces four primary operational decisions. Additional mission actions (e.g., Skip, Retask, IncreaseSampling) remain part of the benchmark design and will be incorporated in future policy versions.

Investigating the [Dataset characterization](https://github.com/gamal-zayed/mission-reason/blob/main/figures/Dataset-Characterization.png), the observed action distribution emerges naturally from the expert policy rather than from explicit class balancing. Approximately 45% of scenarios are labeled as Emergency because the policy incorporates hard spacecraft safety constraints that immediately override utility optimization when critical thermal conditions or severely depleted battery levels are detected. Among non-emergency scenarios, Downlink is frequently selected due to the combined influence of memory pressure and available ground communication opportunities, while Observe is intentionally conservative because scientific observations must satisfy a minimum utility threshold after accounting for environmental and resource constraints. Consequently, the benchmark reflects the operational priorities encoded within the expert policy rather than enforcing an artificial class balance.

Recent work, such as SPRINT [1], demonstrates the growing need for onboard planning and autonomous replanning in resource-constrained small satellite constellations. MissionReason complements this direction by investigating mission-level expert decision policies suitable for lightweight AI benchmarking.

The goal of MissionReason is not to replace planning algorithms such as SPRINT, but to provide a benchmark for studying how lightweight AI models can approximate expert mission-level decision policies that could eventually interact with onboard planners.

### Completed

* Structured mission-state representation.
* Utility-based expert-policy engine.
* Synthetic benchmark generation framework.
* Initial benchmark examples and visualization.

### In Progress

* Parameter-efficient language model training.
* Benchmark evaluation.
* Earth observation integration.
* Space-weather-aware mission scenarios.

---

## Long-Term Vision

MissionReason aims to become an open benchmark for studying mission-level reasoning in future autonomous spacecraft. The long-term objective is to investigate how lightweight artificial intelligence systems can learn expert operational policies while respecting the computational and communication constraints of onboard spacecraft systems.


## Research Foundations

[1] Dahl, Chew & Cahoy, "Optimization of SmallSat Constellations and Low Cost Hardware to Utilize Onboard Planning" (SPRINT).

[2] M. Anderson, J. Muesing, K. Cahoy & K. Center, “Ensemble Learning for Autonomous Onboard Satellite Fault Diagnosis with Validation Tool”. 

[3] Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng & Min-Hung Chen, "DoRA: Weight-Decomposed Low-Rank Adaptation"
