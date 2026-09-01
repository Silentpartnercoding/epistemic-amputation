#!/usr/bin/env python3
"""Stage-three action-before-outcome phantom-schema assay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from experiment import (
    BATCH_SIZE,
    EVAL_STEPS,
    GATES,
    HIDDEN_SIZE,
    INPUT_SIZE,
    MODEL_SEEDS,
    N_EFFECTORS,
    NEGATIVE_EVIDENCE_START,
    REMOVAL_STEP,
    SEQ_LEN,
    BodySchemaGRU,
    episode_inputs,
    mean,
    orthogonal_controls,
    seed_all,
    training_batch,
)


TRAIN_STEPS = 900
WARMUP_STEPS = 4


def decision(model: BodySchemaGRU, hidden: Tensor | None, declaration: Tensor,
             patch: Tensor | None = None) -> Tuple[Tensor, Tensor]:
    batch = declaration.shape[0]
    if hidden is None:
        state = torch.zeros(batch, HIDDEN_SIZE)
    else:
        state = hidden[0]
    if patch is not None:
        state = state + patch.reshape(1, -1)
    action = torch.softmax(model.action_head(state), dim=-1)
    report = torch.sigmoid(model.report_head(torch.cat([state, declaration], dim=-1)))
    return action, report


def temporal_forward(model: BodySchemaGRU, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    hidden = None
    actions: List[Tensor] = []
    reports: List[Tensor] = []
    states: List[Tensor] = []
    for step in range(x.shape[1]):
        declaration = x[:, step, 2 * N_EFFECTORS :]
        action, report = decision(model, hidden, declaration)
        actions.append(action)
        reports.append(report)
        _, hidden = model.gru(x[:, step : step + 1], hidden)
        states.append(hidden[0])
    return torch.stack(actions, 1), torch.stack(reports, 1), torch.stack(states, 1)


def train_model(seed: int) -> Tuple[BodySchemaGRU, Dict[str, float]]:
    seed_all(seed)
    model = BodySchemaGRU()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)
    generator = torch.Generator().manual_seed(seed + 5000)
    final_loss = math.nan
    for _ in range(TRAIN_STEPS):
        x, action_target, report_target = training_batch(BATCH_SIZE, SEQ_LEN, generator)
        action, report, _ = temporal_forward(model, x)
        action_loss = -(action_target[:, WARMUP_STEPS:] * torch.log(
            action[:, WARMUP_STEPS:].clamp_min(1e-8)
        )).sum(dim=-1).mean()
        declared = x[..., 2 * N_EFFECTORS :].abs() > 0
        report_loss = nn.functional.binary_cross_entropy(report[declared], report_target[declared])
        loss = action_loss + 0.25 * report_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
    delta = math.sqrt(sum(float(((model.state_dict()[key] - value) ** 2).sum()) for key, value in initial.items()))
    return model.eval(), {"final_loss": final_loss, "parameter_delta_l2": delta}


@torch.no_grad()
def rollout(model: BodySchemaGRU, condition: str, removed: int, eval_seed: int,
            patch_step: int | None = None, patch: Tensor | None = None) -> Dict[str, object]:
    x, availability = episode_inputs(condition, removed, eval_seed)
    hidden = None
    rows: List[Dict[str, object]] = []
    for step in range(EVAL_STEPS):
        declaration = x[step : step + 1, 2 * N_EFFECTORS :]
        active_patch = patch if patch_step == step else None
        action, report = decision(model, hidden, declaration, active_patch)
        state = torch.zeros(HIDDEN_SIZE) if hidden is None else hidden[0, 0].detach().clone()
        rows.append({
            "step": step,
            "removed": removed,
            "condition": condition,
            "unavailable_action_mass": float(action[0, removed]),
            "reported_absent": float(1.0 - report[0, removed]),
            "actual_absent": bool(availability[step, removed] == 0),
            "patched": active_patch is not None,
            "hidden_before_decision": [float(value) for value in state],
        })
        _, hidden = model.gru(x[step : step + 1, None, :], hidden)
    return {"condition": condition, "removed": removed, "seed": eval_seed, "steps": rows}


@torch.no_grad()
def concept_direction(model: BodySchemaGRU, seed: int, removed: int) -> Tensor:
    present: List[Tensor] = []
    absent: List[Tensor] = []
    for offset in range(18):
        intact = rollout(model, "sham", removed, seed * 30000 + removed * 100 + offset)
        amputee = rollout(model, "adapted_absent", removed, seed * 40000 + removed * 100 + offset)
        present.append(torch.tensor(intact["steps"][24]["hidden_before_decision"]))
        absent.append(torch.tensor(amputee["steps"][24]["hidden_before_decision"]))
    return torch.stack(absent).mean(0) - torch.stack(present).mean(0)


def stable_body_score(model: BodySchemaGRU, seed: int) -> float:
    values = []
    for removed in range(N_EFFECTORS):
        episode = rollout(model, "cold_absent", removed, seed * 777 + removed)
        values.extend(float(row["unavailable_action_mass"]) for row in episode["steps"][12:])
    return mean(values)


def untrained_score(seed: int) -> float:
    seed_all(seed)
    return stable_body_score(BodySchemaGRU().eval(), seed)


def run_seed(seed: int) -> Dict[str, object]:
    model, training = train_model(seed)
    trained_mass = stable_body_score(model, seed)
    untrained_mass = untrained_score(seed)
    per_removed = []
    patch_effects = []
    reverse_effects = []
    controls_all = []
    episodes = []
    for removed in range(N_EFFECTORS):
        eval_seed = seed * 1000 + removed * 17 + 5
        direction = concept_direction(model, seed, removed)
        controls = orthogonal_controls(direction, 8, seed + 12000 + removed)
        sham = rollout(model, "sham", removed, eval_seed)
        told = rollout(model, "told_removal", removed, eval_seed)
        cold = rollout(model, "cold_absent", removed, eval_seed)
        adapted = rollout(model, "adapted_absent", removed, eval_seed)
        patched = rollout(model, "told_removal", removed, eval_seed, NEGATIVE_EVIDENCE_START, direction)
        reversed_ep = rollout(model, "adapted_absent", removed, eval_seed, NEGATIVE_EVIDENCE_START, -direction)
        control_eps = [rollout(model, "told_removal", removed, eval_seed, NEGATIVE_EVIDENCE_START, control) for control in controls]
        episodes.extend([sham, told, cold, adapted, patched, reversed_ep])

        def mass(ep: Dict[str, object], step: int) -> float:
            return float(ep["steps"][step]["unavailable_action_mass"])

        conflict = [mass(told, step) - mass(cold, step) for step in range(NEGATIVE_EVIDENCE_START, EVAL_STEPS)]
        consecutive = 0
        for value in conflict:
            if value >= GATES.conflict_delta:
                consecutive += 1
            else:
                break
        patch_effect = mass(told, NEGATIVE_EVIDENCE_START) - mass(patched, NEGATIVE_EVIDENCE_START)
        reverse_effect = mass(reversed_ep, NEGATIVE_EVIDENCE_START) - mass(adapted, NEGATIVE_EVIDENCE_START)
        control_effects = [mass(told, NEGATIVE_EVIDENCE_START) - mass(ep, NEGATIVE_EVIDENCE_START) for ep in control_eps]
        patch_effects.append(patch_effect)
        reverse_effects.append(reverse_effect)
        controls_all.extend(control_effects)
        report_accuracy = mean(
            1.0 if float(row["reported_absent"]) >= 0.5 else 0.0
            for row in told["steps"][REMOVAL_STEP:]
        )
        cold_mass = mean(mass(cold, step) for step in range(12, EVAL_STEPS))
        sham_mass = mean(mass(sham, step) for step in range(12, EVAL_STEPS))
        per_removed.append({
            "removed": removed,
            "report_accuracy": report_accuracy,
            "declaration_action_delta": sham_mass - cold_mass,
            "conflict_consecutive_steps": consecutive,
            "first_three_conflict_delta": mean(conflict[:3]),
            "patch_effect": patch_effect,
            "reverse_effect": reverse_effect,
        })

    median_control = float(np.median(np.abs(controls_all)))
    conflict_support = all(
        row["report_accuracy"] >= GATES.report_accuracy
        and row["conflict_consecutive_steps"] >= GATES.conflict_steps
        for row in per_removed
    )
    causal_support = mean(patch_effects) >= GATES.patch_delta and mean(reverse_effects) >= GATES.patch_delta
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


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_results = [run_seed(seed) for seed in MODEL_SEEDS]
    conflict_seeds = sum(bool(result["seed_conflict_support"]) for result in seed_results)
    full_seeds = sum(bool(result["seed_full_support"]) for result in seed_results)
    aggregate = {
        "mean_competence_delta": mean(result["competence_delta"] for result in seed_results),
        "mean_patch_effect": mean(result["mean_patch_effect"] for result in seed_results),
        "mean_reverse_effect": mean(result["mean_reverse_effect"] for result in seed_results),
        "median_patch_selectivity_ratio": float(np.median([result["patch_selectivity_ratio"] for result in seed_results])),
        "conflict_supporting_seeds": conflict_seeds,
        "full_supporting_seeds": full_seeds,
    }
    gates = {
        "learned_competence": aggregate["mean_competence_delta"] >= GATES.competence_delta,
        "declaration_understood": all(
            row["report_accuracy"] >= GATES.report_accuracy
            and row["declaration_action_delta"] >= GATES.declaration_action_delta
            for result in seed_results for row in result["per_removed"]
        ),
        "conflict": conflict_seeds >= GATES.seed_support,
        "negative_evidence_persistence": conflict_seeds >= GATES.seed_support,
        "selective_causal_patch": aggregate["mean_patch_effect"] >= GATES.patch_delta,
        "bidirectional_patch": aggregate["mean_reverse_effect"] >= GATES.patch_delta,
        "patch_controls": aggregate["median_patch_selectivity_ratio"] >= GATES.control_ratio,
        "cross_seed": full_seeds >= GATES.seed_support,
    }
    prerequisites = gates["learned_competence"] and gates["declaration_understood"]
    verdict = "SUPPORTED" if all(gates.values()) else ("NOT_SUPPORTED" if prerequisites else "INCONCLUSIVE")
    compact = []
    traces = []
    for result in seed_results:
        compact.append({key: value for key, value in result.items() if key != "episodes"})
        for episode in result["episodes"]:
            for row in episode["steps"]:
                traces.append({"model_seed": result["seed"], **row})
    protocol = {
        "schema": "phantom-schema.causal-world-model.v1",
        "model_seeds": MODEL_SEEDS,
        "effectors": N_EFFECTORS,
        "hidden_size": HIDDEN_SIZE,
        "train_steps": TRAIN_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "removal_step": REMOVAL_STEP,
        "negative_evidence_start": NEGATIVE_EVIDENCE_START,
        "gates": asdict(GATES),
        "causal_order": ["decide_from_prior_state", "report", "observe_response", "update_state"],
    }
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "result.json", {"verdict": verdict, "gates": gates, "aggregate": aggregate, "seeds": compact})
    with (args.output / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    files = [args.output / "protocol.json", args.output / "result.json", args.output / "traces.jsonl"]
    (args.output / "MANIFEST.sha256").write_text(
        "".join(f"{file_hash(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    print(json.dumps({"verdict": verdict, "gates": gates, "aggregate": aggregate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

