#!/usr/bin/env python3
"""Preregistered learned phantom-schema experiment.

The controller never receives an availability mask. It must infer a stable body
from generic command/response history and sparse truthful declarations. A held-
out mid-episode removal tests whether the learned recurrent state conflicts with
correct declarative reporting, and causal hidden-state patches test whether that
state controls the unavailable action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn


N_EFFECTORS = 3
INPUT_SIZE = N_EFFECTORS * 3  # command, observed response, declaration
HIDDEN_SIZE = 18
TRAIN_STEPS = 700
BATCH_SIZE = 96
SEQ_LEN = 24
MODEL_SEEDS = (11, 23, 47, 71, 101)
REMOVAL_STEP = 28
EVAL_STEPS = 48
NEGATIVE_EVIDENCE_START = REMOVAL_STEP + 1


@dataclass(frozen=True)
class Gates:
    competence_delta: float = 0.15
    report_accuracy: float = 0.95
    declaration_action_delta: float = 0.08
    conflict_delta: float = 0.08
    conflict_steps: int = 3
    patch_delta: float = 0.08
    seed_support: int = 4
    control_ratio: float = 2.0


GATES = Gates()


class BodySchemaGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.action_head = nn.Linear(HIDDEN_SIZE, N_EFFECTORS)
        # The direct declaration path makes report/control dissociation
        # measurable. It is disclosed and is never counted as emergent state.
        self.report_head = nn.Linear(HIDDEN_SIZE + N_EFFECTORS, N_EFFECTORS)

    def forward(self, x: Tensor, hidden: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor]:
        states, hidden = self.gru(x, hidden)
        actions = torch.softmax(self.action_head(states), dim=-1)
        declarations = x[..., 2 * N_EFFECTORS :]
        reports = torch.sigmoid(self.report_head(torch.cat([states, declarations], dim=-1)))
        return actions, reports, hidden

    def step(self, x: Tensor, hidden: Tensor | None) -> Tuple[Tensor, Tensor, Tensor]:
        actions, reports, hidden = self.forward(x[:, None, :], hidden)
        return actions[:, 0], reports[:, 0], hidden


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def availability_batch(batch: int, generator: torch.Generator) -> Tensor:
    available = torch.ones(batch, N_EFFECTORS)
    amputated = torch.rand(batch, generator=generator) < 0.55
    removed = torch.randint(0, N_EFFECTORS, (batch,), generator=generator)
    rows = torch.arange(batch)[amputated]
    available[rows, removed[amputated]] = 0.0
    return available


def training_batch(batch: int, length: int, generator: torch.Generator) -> Tuple[Tensor, Tensor, Tensor]:
    available = availability_batch(batch, generator)
    commands = torch.rand(batch, length, N_EFFECTORS, generator=generator) * 2.0 - 1.0
    # Silent spans force recurrent retention rather than current-response lookup.
    commands[:, 8:13] = 0.0
    commands[:, 19:22] = 0.0
    noise = torch.randn(batch, length, N_EFFECTORS, generator=generator) * 0.025
    responses = commands * available[:, None, :] + noise
    responses = torch.where(commands.abs() > 1e-6, responses, torch.zeros_like(responses))
    declarations = torch.zeros(batch, length, N_EFFECTORS)
    visible = torch.rand(batch, length, generator=generator) < 0.18
    signed = available * 2.0 - 1.0
    declarations[visible] = signed[:, None, :].expand(-1, length, -1)[visible]
    inputs = torch.cat([commands, responses, declarations], dim=-1)
    targets = available / available.sum(dim=-1, keepdim=True)
    targets = targets[:, None, :].expand(-1, length, -1)
    return inputs, targets, available[:, None, :].expand(-1, length, -1)


def train_model(seed: int) -> Tuple[BodySchemaGRU, Dict[str, float]]:
    seed_all(seed)
    model = BodySchemaGRU()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)
    generator = torch.Generator().manual_seed(seed + 1000)
    final_loss = math.nan
    for _ in range(TRAIN_STEPS):
        x, action_target, report_target = training_batch(BATCH_SIZE, SEQ_LEN, generator)
        action, report, _ = model(x)
        action_loss = -(action_target * torch.log(action.clamp_min(1e-8))).sum(dim=-1).mean()
        declared = x[..., 2 * N_EFFECTORS :].abs() > 0
        report_loss = nn.functional.binary_cross_entropy(report[declared], report_target[declared])
        loss = action_loss + 0.25 * report_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
    delta = math.sqrt(sum(float(((model.state_dict()[k] - v) ** 2).sum()) for k, v in initial.items()))
    return model.eval(), {"final_loss": final_loss, "parameter_delta_l2": delta}


def episode_inputs(condition: str, removed: int, eval_seed: int) -> Tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(eval_seed)
    commands = torch.rand(EVAL_STEPS, N_EFFECTORS, generator=generator) * 2.0 - 1.0
    commands[8:12] = 0.0
    commands[36:39] = 0.0
    available = torch.ones(EVAL_STEPS, N_EFFECTORS)
    declarations = torch.ones(EVAL_STEPS, N_EFFECTORS)
    if condition in {"hidden_removal", "told_removal"}:
        available[REMOVAL_STEP:, removed] = 0.0
    elif condition in {"cold_absent", "adapted_absent"}:
        available[:, removed] = 0.0
    if condition in {"told_removal"}:
        declarations[REMOVAL_STEP:, removed] = -1.0
    elif condition in {"cold_absent", "adapted_absent"}:
        declarations[:, removed] = -1.0
    # Hidden removal deliberately preserves the now-false present declaration.
    noise = torch.randn(EVAL_STEPS, N_EFFECTORS, generator=generator) * 0.01
    responses = commands * available + noise
    responses = torch.where(commands.abs() > 1e-6, responses, torch.zeros_like(responses))
    x = torch.cat([commands, responses, declarations], dim=-1)
    return x, available


@torch.no_grad()
def rollout(model: BodySchemaGRU, condition: str, removed: int, eval_seed: int,
            patch_step: int | None = None, patch: Tensor | None = None) -> Dict[str, object]:
    x, availability = episode_inputs(condition, removed, eval_seed)
    hidden = None
    rows: List[Dict[str, object]] = []
    for step in range(EVAL_STEPS):
        if patch_step == step and patch is not None:
            if hidden is None:
                raise RuntimeError("cannot patch an empty recurrent state")
            hidden = hidden + patch.reshape(1, 1, -1)
        action, report, hidden = model.step(x[step : step + 1], hidden)
        state = hidden[0, 0].detach().clone()
        rows.append({
            "step": step,
            "removed": removed,
            "condition": condition,
            "unavailable_action_mass": float(action[0, removed]),
            "reported_absent": float(1.0 - report[0, removed]),
            "actual_absent": bool(availability[step, removed] == 0),
            "hidden": [float(v) for v in state],
        })
    return {"condition": condition, "removed": removed, "seed": eval_seed, "steps": rows}


@torch.no_grad()
def concept_direction(model: BodySchemaGRU, seed: int, removed: int) -> Tensor:
    present_states: List[Tensor] = []
    absent_states: List[Tensor] = []
    for offset in range(18):
        intact = rollout(model, "sham", removed, seed * 10000 + removed * 100 + offset)
        absent = rollout(model, "adapted_absent", removed, seed * 20000 + removed * 100 + offset)
        present_states.append(torch.tensor(intact["steps"][24]["hidden"]))
        absent_states.append(torch.tensor(absent["steps"][24]["hidden"]))
    return torch.stack(absent_states).mean(0) - torch.stack(present_states).mean(0)


def orthogonal_controls(direction: Tensor, count: int, seed: int) -> List[Tensor]:
    generator = torch.Generator().manual_seed(seed)
    unit = direction / direction.norm().clamp_min(1e-9)
    controls: List[Tensor] = []
    for _ in range(count):
        candidate = torch.randn(direction.shape, generator=generator)
        candidate = candidate - torch.dot(candidate, unit) * unit
        candidate = candidate / candidate.norm().clamp_min(1e-9) * direction.norm()
        controls.append(candidate)
    return controls


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def run_seed(seed: int) -> Dict[str, object]:
    model, training = train_model(seed)
    trained_mass = stable_body_score(model, seed)
    untrained_mass = untrained_score(seed)
    episodes: List[Dict[str, object]] = []
    patch_effects: List[float] = []
    reverse_effects: List[float] = []
    control_effects: List[float] = []
    per_removed: List[Dict[str, object]] = []
    for removed in range(N_EFFECTORS):
        direction = concept_direction(model, seed, removed)
        controls = orthogonal_controls(direction, 8, seed + 9000 + removed)
        eval_seed = seed * 1000 + removed * 17 + 5
        sham = rollout(model, "sham", removed, eval_seed)
        hidden = rollout(model, "hidden_removal", removed, eval_seed)
        told = rollout(model, "told_removal", removed, eval_seed)
        cold = rollout(model, "cold_absent", removed, eval_seed)
        adapted = rollout(model, "adapted_absent", removed, eval_seed)
        patched = rollout(model, "told_removal", removed, eval_seed, NEGATIVE_EVIDENCE_START, direction)
        reversed_ep = rollout(model, "adapted_absent", removed, eval_seed, NEGATIVE_EVIDENCE_START, -direction)
        control_eps = [rollout(model, "told_removal", removed, eval_seed, NEGATIVE_EVIDENCE_START, c) for c in controls]
        episodes.extend([sham, hidden, told, cold, adapted, patched, reversed_ep])

        def mass(ep: Dict[str, object], step: int) -> float:
            return float(ep["steps"][step]["unavailable_action_mass"])

        patch_effect = mass(told, NEGATIVE_EVIDENCE_START) - mass(patched, NEGATIVE_EVIDENCE_START)
        reverse_effect = mass(reversed_ep, NEGATIVE_EVIDENCE_START) - mass(adapted, NEGATIVE_EVIDENCE_START)
        patch_effects.append(patch_effect)
        reverse_effects.append(reverse_effect)
        for control_ep in control_eps:
            control_effects.append(mass(told, NEGATIVE_EVIDENCE_START) - mass(control_ep, NEGATIVE_EVIDENCE_START))

        conflict_deltas = []
        for step in range(NEGATIVE_EVIDENCE_START, EVAL_STEPS):
            conflict_deltas.append(mass(told, step) - mass(cold, step))
        consecutive = 0
        for value in conflict_deltas:
            if value >= GATES.conflict_delta:
                consecutive += 1
            else:
                break
        report_accuracy = mean(
            1.0 if float(row["reported_absent"]) >= 0.5 else 0.0
            for row in told["steps"][REMOVAL_STEP:]
        )
        cold_mass = mean(mass(cold, step) for step in range(12, EVAL_STEPS))
        sham_mass = mean(mass(sham, step) for step in range(12, EVAL_STEPS))
        per_removed.append({
            "removed": removed,
            "report_accuracy": report_accuracy,
            "conflict_consecutive_steps": consecutive,
            "first_three_conflict_delta": mean(conflict_deltas[:3]),
            "declaration_action_delta": sham_mass - cold_mass,
            "patch_effect": patch_effect,
            "reverse_effect": reverse_effect,
        })

    conflict_support = all(
        item["report_accuracy"] >= GATES.report_accuracy
        and item["conflict_consecutive_steps"] >= GATES.conflict_steps
        for item in per_removed
    )
    causal_support = mean(patch_effects) >= GATES.patch_delta and mean(reverse_effects) >= GATES.patch_delta
    median_control = float(np.median(np.abs(control_effects)))
    return {
        "seed": seed,
        "training": training,
        "per_removed": per_removed,
        "mean_patch_effect": mean(patch_effects),
        "mean_reverse_effect": mean(reverse_effects),
        "median_abs_control_effect": median_control,
        "patch_selectivity_ratio": mean(patch_effects) / max(median_control, 1e-9),
        "seed_conflict_support": conflict_support,
        "seed_causal_support": causal_support,
        "seed_full_support": conflict_support and causal_support,
        "untrained_unavailable_mass": untrained_mass,
        "trained_unavailable_mass": trained_mass,
        "competence_delta": untrained_mass - trained_mass,
        "episodes": episodes,
    }


def stable_body_score(model: BodySchemaGRU, seed: int) -> float:
    masses = []
    for removed in range(N_EFFECTORS):
        episode = rollout(model, "cold_absent", removed, seed * 333 + removed)
        masses.extend(float(r["unavailable_action_mass"]) for r in episode["steps"][12:])
    return mean(masses)


def untrained_score(seed: int) -> float:
    seed_all(seed)
    return stable_body_score(BodySchemaGRU().eval(), seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in MODEL_SEEDS:
        result = run_seed(seed)
        seed_results.append(result)

    conflict_seed_support = sum(bool(r["seed_conflict_support"]) for r in seed_results)
    full_seed_support = sum(bool(r["seed_full_support"]) for r in seed_results)
    aggregate = {
        "mean_competence_delta": mean(float(r["competence_delta"]) for r in seed_results),
        "mean_patch_effect": mean(float(r["mean_patch_effect"]) for r in seed_results),
        "mean_reverse_effect": mean(float(r["mean_reverse_effect"]) for r in seed_results),
        "median_patch_selectivity_ratio": float(np.median([r["patch_selectivity_ratio"] for r in seed_results])),
        "conflict_supporting_seeds": conflict_seed_support,
        "full_supporting_seeds": full_seed_support,
    }
    gates = {
        "learned_competence": aggregate["mean_competence_delta"] >= GATES.competence_delta,
        "declaration_understood": all(
            item["report_accuracy"] >= GATES.report_accuracy
            and item["declaration_action_delta"] >= GATES.declaration_action_delta
            for result in seed_results for item in result["per_removed"]
        ),
        "conflict": conflict_seed_support >= GATES.seed_support,
        "negative_evidence_persistence": conflict_seed_support >= GATES.seed_support,
        "selective_causal_patch": aggregate["mean_patch_effect"] >= GATES.patch_delta,
        "bidirectional_patch": aggregate["mean_reverse_effect"] >= GATES.patch_delta,
        "patch_controls": aggregate["median_patch_selectivity_ratio"] >= GATES.control_ratio,
        "cross_seed": full_seed_support >= GATES.seed_support,
    }
    competence_ok = gates["learned_competence"] and gates["declaration_understood"]
    verdict = "SUPPORTED" if all(gates.values()) else ("NOT_SUPPORTED" if competence_ok else "INCONCLUSIVE")
    compact_results = []
    traces = []
    for result in seed_results:
        copy = {k: v for k, v in result.items() if k != "episodes"}
        compact_results.append(copy)
        for episode in result["episodes"]:
            for row in episode["steps"]:
                traces.append({"model_seed": result["seed"], **row})

    protocol = {
        "schema": "phantom-schema.learned-belief.v1",
        "model_seeds": MODEL_SEEDS,
        "effectors": N_EFFECTORS,
        "hidden_size": HIDDEN_SIZE,
        "train_steps": TRAIN_STEPS,
        "removal_step": REMOVAL_STEP,
        "negative_evidence_start": NEGATIVE_EVIDENCE_START,
        "gates": asdict(GATES),
        "scope_limit": "belief-like causal representation in this assay; no subjective-experience claim",
    }
    result = {"verdict": verdict, "gates": gates, "aggregate": aggregate, "seeds": compact_results}
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "result.json", result)
    with (args.output / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest_paths = [args.output / "protocol.json", args.output / "result.json", args.output / "traces.jsonl"]
    (args.output / "MANIFEST.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in manifest_paths), encoding="utf-8"
    )
    print(json.dumps({"verdict": verdict, "gates": gates, "aggregate": aggregate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
