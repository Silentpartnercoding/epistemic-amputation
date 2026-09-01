#!/usr/bin/env python3
"""Two-timescale, preregistered phantom-schema experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from experiment import N_EFFECTORS, INPUT_SIZE, MODEL_SEEDS, mean, orthogonal_controls, seed_all


FAST_SIZE = 18
SLOW_SIZE = 14
TRAIN_STEPS = 1100
BATCH_SIZE = 80
TRAIN_LENGTH = 36
WARMUP = 5
EVAL_LENGTH = 100
REMOVAL_STEP = 70
EVIDENCE_START = REMOVAL_STEP + 1
GATE_PENALTY = 0.003


@dataclass(frozen=True)
class Thresholds:
    competence_delta: float = 0.15
    report_accuracy: float = 0.95
    action_conflict: float = 0.08
    prediction_conflict: float = 0.15
    persistence_steps: int = 3
    patch_action: float = 0.08
    patch_prediction: float = 0.15
    slow_reset_action: float = 0.08
    control_ratio: float = 2.0
    supporting_seeds: int = 4


THRESHOLDS = Thresholds()


class ConsolidatedBodySchema(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fast_cell = nn.GRUCell(INPUT_SIZE, FAST_SIZE)
        self.slow_proposal = nn.Linear(FAST_SIZE + SLOW_SIZE, SLOW_SIZE)
        self.slow_gate = nn.Linear(FAST_SIZE + SLOW_SIZE, SLOW_SIZE)
        state_size = FAST_SIZE + SLOW_SIZE
        self.action_head = nn.Linear(state_size, N_EFFECTORS)
        self.prediction_head = nn.Linear(state_size, N_EFFECTORS)
        self.report_head = nn.Linear(state_size + N_EFFECTORS, N_EFFECTORS)

    def initial(self, batch: int) -> Tuple[Tensor, Tensor]:
        parameter = next(self.parameters())
        return (
            torch.zeros(batch, FAST_SIZE, dtype=parameter.dtype, device=parameter.device),
            torch.zeros(batch, SLOW_SIZE, dtype=parameter.dtype, device=parameter.device),
        )

    def decide(self, fast: Tensor, slow: Tensor, declaration: Tensor,
               slow_patch: Tensor | None = None, reset_slow: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        decision_slow = torch.zeros_like(slow) if reset_slow else slow
        if slow_patch is not None:
            decision_slow = decision_slow + slow_patch.reshape(1, -1)
        state = torch.cat([fast, decision_slow], dim=-1)
        action = torch.softmax(self.action_head(state), dim=-1)
        prediction = torch.sigmoid(self.prediction_head(state))
        report = torch.sigmoid(self.report_head(torch.cat([state, declaration], dim=-1)))
        return action, prediction, report

    def update(self, x: Tensor, fast: Tensor, slow: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        new_fast = self.fast_cell(x, fast)
        joined = torch.cat([new_fast, slow], dim=-1)
        proposal = torch.tanh(self.slow_proposal(joined))
        gate = torch.sigmoid(self.slow_gate(joined))
        new_slow = slow + gate * (proposal - slow)
        return new_fast, new_slow, gate


def make_training_batch(generator: torch.Generator) -> Tuple[Tensor, Tensor, Tensor]:
    batch = BATCH_SIZE
    available = torch.ones(batch, TRAIN_LENGTH, N_EFFECTORS)
    changed = torch.zeros(batch, TRAIN_LENGTH, dtype=torch.bool)
    mode = torch.rand(batch, generator=generator)
    removed = torch.randint(0, N_EFFECTORS, (batch,), generator=generator)
    change_step = torch.randint(12, 25, (batch,), generator=generator)
    for row in range(batch):
        if mode[row] < 0.34:  # stable absent
            available[row, :, removed[row]] = 0.0
        elif mode[row] < 0.68:  # genuine mid-episode removal
            step = int(change_step[row])
            available[row, step:, removed[row]] = 0.0
            changed[row, step] = True
        # Remaining rows are intact.

    commands = torch.rand(batch, TRAIN_LENGTH, N_EFFECTORS, generator=generator) * 2.0 - 1.0
    commands[:, 9:12] = 0.0
    commands[:, 29:32] = 0.0
    responses = commands * available
    # Temporary sensor failures occur without a morphology/declaration change.
    dropouts = torch.rand(batch, TRAIN_LENGTH, N_EFFECTORS, generator=generator) < 0.055
    responses = torch.where(dropouts, torch.zeros_like(responses), responses)
    noise = torch.randn(batch, TRAIN_LENGTH, N_EFFECTORS, generator=generator) * 0.015
    responses = torch.where(commands.abs() > 1e-6, responses + noise, torch.zeros_like(responses))
    declarations = available * 2.0 - 1.0
    x = torch.cat([commands, responses, declarations], dim=-1)
    target_action = available / available.sum(dim=-1, keepdim=True)
    return x, target_action, changed


def forward_sequence(model: ConsolidatedBodySchema, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    fast, slow = model.initial(x.shape[0])
    actions: List[Tensor] = []
    predictions: List[Tensor] = []
    reports: List[Tensor] = []
    gates: List[Tensor] = []
    for step in range(x.shape[1]):
        declaration = x[:, step, 2 * N_EFFECTORS :]
        action, prediction, report = model.decide(fast, slow, declaration)
        actions.append(action)
        predictions.append(prediction)
        reports.append(report)
        fast, slow, gate = model.update(x[:, step], fast, slow)
        gates.append(gate)
    return (
        torch.stack(actions, 1),
        torch.stack(predictions, 1),
        torch.stack(reports, 1),
        torch.stack(gates, 1),
    )


def train_model(seed: int) -> Tuple[ConsolidatedBodySchema, Dict[str, float]]:
    seed_all(seed)
    model = ConsolidatedBodySchema()
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    generator = torch.Generator().manual_seed(seed + 15000)
    final = {}
    for _ in range(TRAIN_STEPS):
        x, action_target, changed = make_training_batch(generator)
        action, prediction, report, gates = forward_sequence(model, x)
        valid = torch.ones_like(changed)
        valid[:, :WARMUP] = False
        valid &= ~changed
        action_loss = -(action_target * torch.log(action.clamp_min(1e-8))).sum(-1)[valid].mean()
        prediction_loss = nn.functional.binary_cross_entropy(prediction[valid], (action_target > 0)[valid].float())
        report_target = (action_target > 0).float()
        report_loss = nn.functional.binary_cross_entropy(report, report_target)
        gate_activity = gates.mean()
        loss = action_loss + 0.55 * prediction_loss + 0.30 * report_loss + GATE_PENALTY * gate_activity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = {
            "loss": float(loss.detach()),
            "action_loss": float(action_loss.detach()),
            "prediction_loss": float(prediction_loss.detach()),
            "report_loss": float(report_loss.detach()),
            "mean_gate": float(gate_activity.detach()),
        }
    final["parameter_delta_l2"] = math.sqrt(sum(
        float(((model.state_dict()[key] - value) ** 2).sum()) for key, value in initial.items()
    ))
    return model.eval(), final


def evaluation_inputs(condition: str, removed: int, seed: int) -> Tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    commands = torch.rand(EVAL_LENGTH, N_EFFECTORS, generator=generator) * 2.0 - 1.0
    commands[18:22] = 0.0
    commands[88:91] = 0.0
    available = torch.ones(EVAL_LENGTH, N_EFFECTORS)
    declarations = torch.ones(EVAL_LENGTH, N_EFFECTORS)
    if condition in {"told_removal", "hidden_removal"}:
        available[REMOVAL_STEP:, removed] = 0.0
    elif condition == "cold_absent":
        available[:, removed] = 0.0
    if condition == "told_removal":
        declarations[REMOVAL_STEP:, removed] = -1.0
    elif condition == "cold_absent":
        declarations[:, removed] = -1.0
    responses = commands * available
    noise = torch.randn(EVAL_LENGTH, N_EFFECTORS, generator=generator) * 0.008
    responses = torch.where(
        commands.abs() > 1e-6,
        responses + noise * available,
        torch.zeros_like(responses),
    )
    return torch.cat([commands, responses, declarations], dim=-1), available


@torch.no_grad()
def rollout(model: ConsolidatedBodySchema, condition: str, removed: int, seed: int,
            patch_step: int | None = None, slow_patch: Tensor | None = None,
            reset_step: int | None = None) -> Dict[str, object]:
    x, available = evaluation_inputs(condition, removed, seed)
    fast, slow = model.initial(1)
    rows = []
    for step in range(EVAL_LENGTH):
        declaration = x[step : step + 1, 2 * N_EFFECTORS :]
        patch = slow_patch if patch_step == step else None
        reset = reset_step == step
        action, prediction, report = model.decide(fast, slow, declaration, patch, reset)
        rows.append({
            "step": step,
            "condition": condition,
            "removed": removed,
            "actual_absent": bool(available[step, removed] == 0),
            "unavailable_action_mass": float(action[0, removed]),
            "predicted_present": float(prediction[0, removed]),
            "reported_absent": float(1.0 - report[0, removed]),
            "fast": [float(value) for value in fast[0]],
            "slow": [float(value) for value in slow[0]],
            "patched": patch is not None,
            "slow_reset": reset,
        })
        fast, slow, gate = model.update(x[step : step + 1], fast, slow)
        rows[-1]["mean_update_gate"] = float(gate.mean())
    return {"condition": condition, "removed": removed, "seed": seed, "steps": rows}


@torch.no_grad()
def slow_direction(model: ConsolidatedBodySchema, removed: int, seed: int) -> Tensor:
    present = []
    absent = []
    for offset in range(20):
        intact = rollout(model, "sham", removed, seed * 50000 + removed * 100 + offset)
        cold = rollout(model, "cold_absent", removed, seed * 60000 + removed * 100 + offset)
        present.append(torch.tensor(intact["steps"][55]["slow"]))
        absent.append(torch.tensor(cold["steps"][55]["slow"]))
    return torch.stack(absent).mean(0) - torch.stack(present).mean(0)


def stable_score(model: ConsolidatedBodySchema, seed: int) -> float:
    values = []
    for removed in range(N_EFFECTORS):
        episode = rollout(model, "cold_absent", removed, seed * 909 + removed)
        values.extend(float(row["unavailable_action_mass"]) for row in episode["steps"][20:])
    return mean(values)


def untrained_score(seed: int) -> float:
    seed_all(seed)
    return stable_score(ConsolidatedBodySchema().eval(), seed)


def metric(ep: Dict[str, object], step: int, name: str) -> float:
    return float(ep["steps"][step][name])


def run_seed(seed: int) -> Dict[str, object]:
    model, training = train_model(seed)
    trained_mass = stable_score(model, seed)
    random_mass = untrained_score(seed)
    per_removed = []
    episodes = []
    patch_action_effects = []
    patch_prediction_effects = []
    reverse_action_effects = []
    reverse_prediction_effects = []
    reset_effects = []
    control_action_effects = []
    for removed in range(N_EFFECTORS):
        eval_seed = seed * 2000 + removed * 23 + 7
        direction = slow_direction(model, removed, seed)
        controls = orthogonal_controls(direction, 8, seed + 30000 + removed)
        sham = rollout(model, "sham", removed, eval_seed)
        hidden = rollout(model, "hidden_removal", removed, eval_seed)
        told = rollout(model, "told_removal", removed, eval_seed)
        cold = rollout(model, "cold_absent", removed, eval_seed)
        patched = rollout(model, "told_removal", removed, eval_seed, EVIDENCE_START, direction)
        reversed_ep = rollout(model, "cold_absent", removed, eval_seed, EVIDENCE_START, -direction)
        reset_ep = rollout(model, "told_removal", removed, eval_seed, reset_step=EVIDENCE_START)
        control_eps = [rollout(model, "told_removal", removed, eval_seed, EVIDENCE_START, control) for control in controls]
        episodes.extend([sham, hidden, told, cold, patched, reversed_ep, reset_ep])

        action_deltas = [
            metric(told, step, "unavailable_action_mass") - metric(cold, step, "unavailable_action_mass")
            for step in range(EVIDENCE_START, EVAL_LENGTH)
        ]
        prediction_deltas = [
            metric(told, step, "predicted_present") - metric(cold, step, "predicted_present")
            for step in range(EVIDENCE_START, EVAL_LENGTH)
        ]
        consecutive = 0
        for action_delta, prediction_delta in zip(action_deltas, prediction_deltas):
            if action_delta >= THRESHOLDS.action_conflict and prediction_delta >= THRESHOLDS.prediction_conflict:
                consecutive += 1
            else:
                break
        initial_magnitude = max(action_deltas[:3]) if action_deltas else 0.0
        late_magnitude = mean(action_deltas[-8:])
        adapted = initial_magnitude > 0 and late_magnitude < initial_magnitude / 2.0
        report_accuracy = mean(
            1.0 if metric(told, step, "reported_absent") >= 0.5 else 0.0
            for step in range(REMOVAL_STEP, EVAL_LENGTH)
        )
        cold_mass = mean(metric(cold, step, "unavailable_action_mass") for step in range(20, EVAL_LENGTH))
        sham_mass = mean(metric(sham, step, "unavailable_action_mass") for step in range(20, EVAL_LENGTH))
        patch_action = metric(told, EVIDENCE_START, "unavailable_action_mass") - metric(patched, EVIDENCE_START, "unavailable_action_mass")
        patch_prediction = metric(told, EVIDENCE_START, "predicted_present") - metric(patched, EVIDENCE_START, "predicted_present")
        reverse_action = metric(reversed_ep, EVIDENCE_START, "unavailable_action_mass") - metric(cold, EVIDENCE_START, "unavailable_action_mass")
        reverse_prediction = metric(reversed_ep, EVIDENCE_START, "predicted_present") - metric(cold, EVIDENCE_START, "predicted_present")
        reset_effect = metric(told, EVIDENCE_START, "unavailable_action_mass") - metric(reset_ep, EVIDENCE_START, "unavailable_action_mass")
        control_effects = [
            metric(told, EVIDENCE_START, "unavailable_action_mass") - metric(control, EVIDENCE_START, "unavailable_action_mass")
            for control in control_eps
        ]
        patch_action_effects.append(patch_action)
        patch_prediction_effects.append(patch_prediction)
        reverse_action_effects.append(reverse_action)
        reverse_prediction_effects.append(reverse_prediction)
        reset_effects.append(reset_effect)
        control_action_effects.extend(control_effects)
        per_removed.append({
            "removed": removed,
            "report_accuracy": report_accuracy,
            "declaration_action_delta": sham_mass - cold_mass,
            "conflict_consecutive_steps": consecutive,
            "first_three_action_conflict": mean(action_deltas[:3]),
            "first_three_prediction_conflict": mean(prediction_deltas[:3]),
            "eventual_adaptation": adapted,
            "patch_action_effect": patch_action,
            "patch_prediction_effect": patch_prediction,
            "reverse_action_effect": reverse_action,
            "reverse_prediction_effect": reverse_prediction,
            "slow_reset_action_effect": reset_effect,
        })

    median_control = float(np.median(np.abs(control_action_effects)))
    mean_patch_action = mean(patch_action_effects)
    conflict_support = all(
        row["report_accuracy"] >= THRESHOLDS.report_accuracy
        and row["conflict_consecutive_steps"] >= THRESHOLDS.persistence_steps
        and row["eventual_adaptation"]
        for row in per_removed
    )
    causal_support = (
        mean_patch_action >= THRESHOLDS.patch_action
        and mean(patch_prediction_effects) >= THRESHOLDS.patch_prediction
        and mean(reverse_action_effects) >= THRESHOLDS.patch_action
        and mean(reverse_prediction_effects) >= THRESHOLDS.patch_prediction
        and mean(reset_effects) >= THRESHOLDS.slow_reset_action
    )
    return {
        "seed": seed,
        "training": training,
        "random_unavailable_mass": random_mass,
        "trained_unavailable_mass": trained_mass,
        "competence_delta": random_mass - trained_mass,
        "per_removed": per_removed,
        "mean_patch_action_effect": mean_patch_action,
        "mean_patch_prediction_effect": mean(patch_prediction_effects),
        "mean_reverse_action_effect": mean(reverse_action_effects),
        "mean_reverse_prediction_effect": mean(reverse_prediction_effects),
        "mean_slow_reset_action_effect": mean(reset_effects),
        "median_abs_control_action_effect": median_control,
        "patch_selectivity_ratio": mean_patch_action / max(median_control, 1e-9),
        "seed_conflict_support": conflict_support,
        "seed_causal_support": causal_support,
        "seed_full_support": conflict_support and causal_support,
        "episodes": episodes,
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [run_seed(seed) for seed in MODEL_SEEDS]
    conflict_seeds = sum(bool(result["seed_conflict_support"]) for result in results)
    full_seeds = sum(bool(result["seed_full_support"]) for result in results)
    aggregate = {
        "mean_competence_delta": mean(result["competence_delta"] for result in results),
        "mean_patch_action_effect": mean(result["mean_patch_action_effect"] for result in results),
        "mean_patch_prediction_effect": mean(result["mean_patch_prediction_effect"] for result in results),
        "mean_reverse_action_effect": mean(result["mean_reverse_action_effect"] for result in results),
        "mean_reverse_prediction_effect": mean(result["mean_reverse_prediction_effect"] for result in results),
        "mean_slow_reset_action_effect": mean(result["mean_slow_reset_action_effect"] for result in results),
        "median_patch_selectivity_ratio": float(np.median([result["patch_selectivity_ratio"] for result in results])),
        "conflict_supporting_seeds": conflict_seeds,
        "full_supporting_seeds": full_seeds,
    }
    gates = {
        "learned_competence": aggregate["mean_competence_delta"] >= THRESHOLDS.competence_delta,
        "truthful_reporting": all(
            row["report_accuracy"] >= THRESHOLDS.report_accuracy
            and row["declaration_action_delta"] >= THRESHOLDS.action_conflict
            for result in results for row in result["per_removed"]
        ),
        "persistent_double_dissociation": conflict_seeds >= THRESHOLDS.supporting_seeds,
        "eventual_adaptation": sum(all(row["eventual_adaptation"] for row in result["per_removed"]) for result in results) >= THRESHOLDS.supporting_seeds,
        "absent_schema_patch": (
            aggregate["mean_patch_action_effect"] >= THRESHOLDS.patch_action
            and aggregate["mean_patch_prediction_effect"] >= THRESHOLDS.patch_prediction
        ),
        "reverse_schema_patch": (
            aggregate["mean_reverse_action_effect"] >= THRESHOLDS.patch_action
            and aggregate["mean_reverse_prediction_effect"] >= THRESHOLDS.patch_prediction
        ),
        "slow_memory_reset": aggregate["mean_slow_reset_action_effect"] >= THRESHOLDS.slow_reset_action,
        "patch_controls": aggregate["median_patch_selectivity_ratio"] >= THRESHOLDS.control_ratio,
        "cross_seed": full_seeds >= THRESHOLDS.supporting_seeds,
    }
    prerequisites = gates["learned_competence"] and gates["truthful_reporting"]
    verdict = "SUPPORTED" if all(gates.values()) else ("NOT_SUPPORTED" if prerequisites else "INCONCLUSIVE")
    compact = []
    traces = []
    for result in results:
        compact.append({key: value for key, value in result.items() if key != "episodes"})
        for episode in result["episodes"]:
            for row in episode["steps"]:
                traces.append({"model_seed": result["seed"], **row})
    protocol = {
        "schema": "phantom-schema.ghost-shell.v1",
        "model_seeds": MODEL_SEEDS,
        "fast_size": FAST_SIZE,
        "slow_size": SLOW_SIZE,
        "train_steps": TRAIN_STEPS,
        "gate_penalty": GATE_PENALTY,
        "removal_step": REMOVAL_STEP,
        "evidence_start": EVIDENCE_START,
        "thresholds": asdict(THRESHOLDS),
        "scope_limit": "learned belief-like causal body schema only; no subjective-experience claim",
    }
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "result.json", {"verdict": verdict, "gates": gates, "aggregate": aggregate, "seeds": compact})
    with (args.output / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    files = [args.output / "protocol.json", args.output / "result.json", args.output / "traces.jsonl"]
    (args.output / "MANIFEST.sha256").write_text(
        "".join(f"{digest(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    print(json.dumps({"verdict": verdict, "gates": gates, "aggregate": aggregate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
