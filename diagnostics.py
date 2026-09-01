#!/usr/bin/env python3
"""Exploratory diagnostics after the frozen stage-two null result.

These checks do not alter the preregistered verdict. They ask whether the null
came from (a) no decodable recurrent body representation, (b) a poor mean-
direction intervention, or (c) the action policy relying almost entirely on
current evidence rather than recurrent state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

from experiment import (
    EVAL_STEPS,
    HIDDEN_SIZE,
    MODEL_SEEDS,
    N_EFFECTORS,
    NEGATIVE_EVIDENCE_START,
    BodySchemaGRU,
    episode_inputs,
    mean,
    rollout,
    seed_all,
    train_model,
    write_json,
)


@torch.no_grad()
def hidden_before(model: BodySchemaGRU, condition: str, removed: int, eval_seed: int, step: int) -> Tensor:
    x, _ = episode_inputs(condition, removed, eval_seed)
    hidden = None
    for index in range(step):
        _, _, hidden = model.step(x[index : index + 1], hidden)
    if hidden is None:
        return torch.zeros(1, 1, HIDDEN_SIZE)
    return hidden.clone()


@torch.no_grad()
def action_with_replacement(model: BodySchemaGRU, target_condition: str, donor_condition: str,
                            removed: int, eval_seed: int, step: int) -> Tuple[float, float]:
    target_x, _ = episode_inputs(target_condition, removed, eval_seed)
    donor_hidden = hidden_before(model, donor_condition, removed, eval_seed, step)
    action, report, _ = model.step(target_x[step : step + 1], donor_hidden)
    return float(action[0, removed]), float(1.0 - report[0, removed])


@torch.no_grad()
def probe_dataset(model: BodySchemaGRU, removed: int, seed: int, count: int) -> Tuple[Tensor, Tensor]:
    states: List[Tensor] = []
    labels: List[float] = []
    for index in range(count):
        for condition, label in (("sham", 1.0), ("cold_absent", 0.0)):
            x, _ = episode_inputs(condition, removed, seed * 100000 + index)
            # Remove all declarations: probe only sensorimotor memory.
            x[:, 2 * N_EFFECTORS :] = 0.0
            hidden = None
            for step in range(25):
                _, _, hidden = model.step(x[step : step + 1], hidden)
            states.append(hidden[0, 0].clone())
            labels.append(label)
    return torch.stack(states), torch.tensor(labels)


def linear_probe_accuracy(model: BodySchemaGRU, removed: int, seed: int) -> float:
    train_x, train_y = probe_dataset(model, removed, seed + 3000, 80)
    test_x, test_y = probe_dataset(model, removed, seed + 6000, 40)
    seed_all(seed + removed + 7000)
    probe = nn.Linear(HIDDEN_SIZE, 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.04)
    for _ in range(300):
        logits = probe(train_x)[:, 0]
        loss = nn.functional.binary_cross_entropy_with_logits(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = torch.sigmoid(probe(test_x)[:, 0]) >= 0.5
    return float((prediction == test_y.bool()).float().mean())


def run_seed(seed: int) -> Dict[str, object]:
    model, _ = train_model(seed)
    rows = []
    for removed in range(N_EFFECTORS):
        eval_seed = seed * 1000 + removed * 17 + 5
        told = rollout(model, "told_removal", removed, eval_seed)
        adapted = rollout(model, "adapted_absent", removed, eval_seed)
        told_mass = float(told["steps"][NEGATIVE_EVIDENCE_START]["unavailable_action_mass"])
        adapted_mass = float(adapted["steps"][NEGATIVE_EVIDENCE_START]["unavailable_action_mass"])
        absent_replacement, absent_report = action_with_replacement(
            model, "told_removal", "adapted_absent", removed, eval_seed, NEGATIVE_EVIDENCE_START
        )
        present_replacement, present_report = action_with_replacement(
            model, "adapted_absent", "sham", removed, eval_seed, NEGATIVE_EVIDENCE_START
        )
        rows.append({
            "removed": removed,
            "sensorimotor_probe_accuracy": linear_probe_accuracy(model, removed, seed),
            "absent_state_replacement_effect": told_mass - absent_replacement,
            "absent_state_replacement_report_correct": absent_report >= 0.5,
            "present_state_replacement_effect": present_replacement - adapted_mass,
            "present_state_replacement_report_correct": present_report >= 0.5,
        })
    return {"seed": seed, "effectors": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = [run_seed(seed) for seed in MODEL_SEEDS]
    rows = [row for seed in seeds for row in seed["effectors"]]
    result = {
        "status": "EXPLORATORY_POST_NULL_DIAGNOSTIC",
        "does_not_change_preregistered_verdict": True,
        "aggregate": {
            "mean_sensorimotor_probe_accuracy": mean(row["sensorimotor_probe_accuracy"] for row in rows),
            "mean_absent_state_replacement_effect": mean(row["absent_state_replacement_effect"] for row in rows),
            "mean_present_state_replacement_effect": mean(row["present_state_replacement_effect"] for row in rows),
        },
        "seeds": seeds,
    }
    write_json(args.output / "diagnostics.json", result)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

