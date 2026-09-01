#!/usr/bin/env python3
"""Preregistered epistemic-amputation assay for a recurrent causal learner."""

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

from experiment import MODEL_SEEDS, mean, orthogonal_controls, seed_all


INPUT_SIZE = 4  # cue, outcome, source-valid marker, explicit debrief
HIDDEN_SIZE = 24
TRAIN_STEPS = 850
BATCH_SIZE = 96
TRAIN_LENGTH = 30
FORMATION_STEPS = 18
POST_STEPS = 12


@dataclass(frozen=True)
class Thresholds:
    competence_delta: float = 0.20
    formation_level: float = 0.75
    report_accuracy: float = 0.95
    conflict_delta: float = 0.12
    persistence_steps: int = 3
    patch_delta: float = 0.12
    control_ratio: float = 2.0
    supporting_seeds: int = 4


THRESHOLDS = Thresholds()


class EpistemicAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.GRUCell(INPUT_SIZE, HIDDEN_SIZE)
        self.prediction_head = nn.Linear(HIDDEN_SIZE, 1)
        self.wager_head = nn.Linear(HIDDEN_SIZE, 1)
        # Direct access is disclosed: report agreement cannot establish a ghost.
        self.report_head = nn.Linear(HIDDEN_SIZE + 1, 1)

    def initial(self, batch: int) -> Tensor:
        p = next(self.parameters())
        return torch.zeros(batch, HIDDEN_SIZE, dtype=p.dtype, device=p.device)

    def decide(self, hidden: Tensor, debrief: Tensor,
               patch: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor]:
        state = hidden if patch is None else hidden + patch.reshape(1, -1)
        prediction = torch.sigmoid(self.prediction_head(state)).squeeze(-1)
        wager = torch.sigmoid(self.wager_head(state)).squeeze(-1)
        report = torch.sigmoid(self.report_head(torch.cat([state, debrief[:, None]], -1))).squeeze(-1)
        return prediction, wager, report

    def update(self, observation: Tensor, hidden: Tensor) -> Tensor:
        return self.cell(observation, hidden)


def make_batch(generator: torch.Generator) -> Tuple[Tensor, Tensor]:
    batch = BATCH_SIZE
    mode = torch.randint(0, 4, (batch,), generator=generator)
    causal = torch.ones(batch, TRAIN_LENGTH)
    causal[mode == 1] = 0.0  # stable null
    revocation_step = torch.randint(12, 21, (batch,), generator=generator)
    x = torch.zeros(batch, TRAIN_LENGTH, INPUT_SIZE)
    target = torch.zeros(batch, TRAIN_LENGTH)
    for row in range(batch):
        revoked = int(mode[row]) == 2
        reversed_relation = int(mode[row]) == 3
        step0 = int(revocation_step[row])
        cue = torch.where(torch.rand(TRAIN_LENGTH, generator=generator) > 0.5, 1.0, -1.0)
        noise = torch.rand(TRAIN_LENGTH, generator=generator)
        outcome = torch.where(noise < 0.88, cue, -cue)
        if int(mode[row]) == 1:
            outcome = torch.where(torch.rand(TRAIN_LENGTH, generator=generator) > 0.5, 1.0, -1.0)
        if revoked:
            causal[row, step0:] = 0.0
            outcome[step0:] = torch.where(torch.rand(TRAIN_LENGTH - step0, generator=generator) > 0.5, 1.0, -1.0)
            x[row, step0:, 2] = -1.0
            x[row, step0, 3] = 1.0
        elif reversed_relation:
            outcome[step0:] = torch.where(
                torch.rand(TRAIN_LENGTH - step0, generator=generator) < 0.88,
                -cue[step0:], cue[step0:],
            )
            causal[row, step0:] = 0.0
        else:
            x[row, :, 2] = 1.0 if int(mode[row]) == 0 else 0.0
        x[row, :, 0] = cue
        x[row, :, 1] = outcome
        target[row] = causal[row]
    return x, target


def forward(model: EpistemicAgent, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    hidden = model.initial(x.shape[0])
    predictions: List[Tensor] = []
    wagers: List[Tensor] = []
    reports: List[Tensor] = []
    for step in range(x.shape[1]):
        hidden = model.update(x[:, step], hidden)
        p, w, r = model.decide(hidden, x[:, step, 3])
        predictions.append(p)
        wagers.append(w)
        reports.append(r)
    return torch.stack(predictions, 1), torch.stack(wagers, 1), torch.stack(reports, 1)


def train_model(seed: int) -> Tuple[EpistemicAgent, Dict[str, float]]:
    seed_all(seed)
    model = EpistemicAgent()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    generator = torch.Generator().manual_seed(seed + 51000)
    final_loss = math.nan
    for _ in range(TRAIN_STEPS):
        x, target = make_batch(generator)
        prediction, wager, report = forward(model, x)
        valid = torch.arange(TRAIN_LENGTH)[None, :] >= 4
        valid = valid.expand(BATCH_SIZE, -1)
        loss = (
            nn.functional.binary_cross_entropy(prediction[valid], target[valid])
            + nn.functional.binary_cross_entropy(wager[valid], target[valid])
            + 0.35 * nn.functional.binary_cross_entropy(report[valid], target[valid])
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
    delta = math.sqrt(sum(float(((model.state_dict()[k] - v) ** 2).sum()) for k, v in initial.items()))
    return model.eval(), {"final_loss": final_loss, "parameter_delta_l2": delta}


def episode(condition: str, eval_seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(eval_seed)
    length = FORMATION_STEPS + POST_STEPS
    cue = torch.where(torch.rand(length, generator=generator) > 0.5, 1.0, -1.0)
    outcome = torch.where(torch.rand(length, generator=generator) < 0.94, cue, -cue)
    x = torch.zeros(length, INPUT_SIZE)
    x[:, 0] = cue
    x[:, 1] = outcome
    x[:, 2] = 1.0
    if condition in {"amputated", "cold_null"}:
        if condition == "cold_null":
            start = 0
        else:
            start = FORMATION_STEPS
        x[start:, 1] = torch.where(torch.rand(length - start, generator=generator) > 0.5, 1.0, -1.0)
        x[start:, 2] = -1.0
        x[start, 3] = 1.0
    elif condition == "reversal":
        x[FORMATION_STEPS:, 1] = -cue[FORMATION_STEPS:]
    elif condition != "sham":
        raise ValueError(condition)
    return x


@torch.no_grad()
def rollout(model: EpistemicAgent, condition: str, eval_seed: int,
            patch_step: int | None = None, patch: Tensor | None = None) -> Dict[str, object]:
    x = episode(condition, eval_seed)
    hidden = model.initial(1)
    rows = []
    for step in range(x.shape[0]):
        hidden = model.update(x[step : step + 1], hidden)
        active_patch = patch if step == patch_step else None
        p, w, r = model.decide(hidden, x[step : step + 1, 3], active_patch)
        rows.append({
            "step": step,
            "condition": condition,
            "prediction": float(p[0]),
            "wager": float(w[0]),
            "reported_active": float(r[0]),
            "hidden": [float(v) for v in hidden[0]],
            "patched": active_patch is not None,
        })
    return {"condition": condition, "seed": eval_seed, "steps": rows}


@torch.no_grad()
def counterexample_criterion(model: EpistemicAgent, eval_seed: int) -> int:
    x = episode("sham", eval_seed)[:FORMATION_STEPS]
    hidden = model.initial(1)
    for row in x:
        hidden = model.update(row[None], hidden)
    baseline = float(model.decide(hidden, torch.zeros(1))[1][0])
    for count in range(1, POST_STEPS + 1):
        cue = 1.0 if count % 2 else -1.0
        contrary = torch.tensor([[cue, -cue, 1.0, 0.0]])
        hidden = model.update(contrary, hidden)
        wager = float(model.decide(hidden, torch.zeros(1))[1][0])
        if wager < min(0.5, baseline - THRESHOLDS.conflict_delta):
            return count
    return POST_STEPS + 1


def hidden_at(ep: Dict[str, object], step: int) -> Tensor:
    return torch.tensor(ep["steps"][step]["hidden"])


def score_untrained(seed: int) -> float:
    seed_all(seed)
    ep = rollout(EpistemicAgent().eval(), "sham", seed + 700)
    return mean(float(r["prediction"]) for r in ep["steps"][8:FORMATION_STEPS])


def run_seed(seed: int) -> Dict[str, object]:
    model, training = train_model(seed)
    eval_seed = seed * 1009 + 17
    sham = rollout(model, "sham", eval_seed)
    amputated = rollout(model, "amputated", eval_seed)
    cold = rollout(model, "cold_null", eval_seed)
    reversal = rollout(model, "reversal", eval_seed)
    step = FORMATION_STEPS + 1
    absent_patch = hidden_at(cold, step) - hidden_at(amputated, step)
    reverse_patch = hidden_at(sham, FORMATION_STEPS - 1) - hidden_at(cold, step)
    patched = rollout(model, "amputated", eval_seed, step, absent_patch)
    reversed_ep = rollout(model, "cold_null", eval_seed, step, reverse_patch)
    controls = orthogonal_controls(absent_patch, 8, seed + 88000)
    control_eps = [rollout(model, "amputated", eval_seed, step, c) for c in controls]

    formation_prediction = mean(float(r["prediction"]) for r in sham["steps"][8:FORMATION_STEPS])
    formation_wager = mean(float(r["wager"]) for r in sham["steps"][8:FORMATION_STEPS])
    random_prediction = score_untrained(seed)
    report_accuracy = mean(
        1.0 if float(r["reported_active"]) < 0.5 else 0.0
        for r in amputated["steps"][FORMATION_STEPS:]
    )
    deltas = []
    for index in range(FORMATION_STEPS, FORMATION_STEPS + POST_STEPS):
        deltas.append((
            float(amputated["steps"][index]["prediction"]) - float(cold["steps"][index]["prediction"]),
            float(amputated["steps"][index]["wager"]) - float(cold["steps"][index]["wager"]),
        ))
    persistent = 0
    for prediction_delta, wager_delta in deltas[1:]:
        if prediction_delta >= THRESHOLDS.conflict_delta and wager_delta >= THRESHOLDS.conflict_delta:
            persistent += 1
        else:
            break
    criterion = counterexample_criterion(model, eval_seed)
    criterion_migrated = criterion <= POST_STEPS and persistent > criterion

    base_p = float(amputated["steps"][step]["prediction"])
    base_w = float(amputated["steps"][step]["wager"])
    patch_p = base_p - float(patched["steps"][step]["prediction"])
    patch_w = base_w - float(patched["steps"][step]["wager"])
    reverse_p = float(reversed_ep["steps"][step]["prediction"]) - float(cold["steps"][step]["prediction"])
    reverse_w = float(reversed_ep["steps"][step]["wager"]) - float(cold["steps"][step]["wager"])
    control_w = [base_w - float(ep["steps"][step]["wager"]) for ep in control_eps]
    selectivity = patch_w / max(float(np.median(np.abs(control_w))), 1e-9)
    bundle = (
        formation_prediction >= THRESHOLDS.formation_level
        and formation_wager >= THRESHOLDS.formation_level
        and report_accuracy >= THRESHOLDS.report_accuracy
        and persistent >= THRESHOLDS.persistence_steps
        and criterion_migrated
        and patch_p >= THRESHOLDS.patch_delta and patch_w >= THRESHOLDS.patch_delta
        and reverse_p >= THRESHOLDS.patch_delta and reverse_w >= THRESHOLDS.patch_delta
        and selectivity >= THRESHOLDS.control_ratio
    )
    return {
        "seed": seed,
        "training": training,
        "competence_delta": formation_prediction - random_prediction,
        "formation_prediction": formation_prediction,
        "formation_wager": formation_wager,
        "report_accuracy": report_accuracy,
        "precommitted_counterexample_criterion": criterion,
        "persistent_conflict_steps": persistent,
        "criterion_migrated": criterion_migrated,
        "patch_prediction_effect": patch_p,
        "patch_wager_effect": patch_w,
        "reverse_prediction_effect": reverse_p,
        "reverse_wager_effect": reverse_w,
        "patch_selectivity_ratio": selectivity,
        "full_support": bundle,
        "episodes": [sham, amputated, cold, reversal, patched, reversed_ep, *control_eps],
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
    supporting = sum(bool(r["full_support"]) for r in results)
    aggregate = {
        "mean_competence_delta": mean(r["competence_delta"] for r in results),
        "mean_formation_prediction": mean(r["formation_prediction"] for r in results),
        "mean_formation_wager": mean(r["formation_wager"] for r in results),
        "mean_report_accuracy": mean(r["report_accuracy"] for r in results),
        "mean_precommitted_criterion": mean(r["precommitted_counterexample_criterion"] for r in results),
        "mean_persistent_conflict_steps": mean(r["persistent_conflict_steps"] for r in results),
        "criterion_migration_seeds": sum(bool(r["criterion_migrated"]) for r in results),
        "mean_patch_prediction_effect": mean(r["patch_prediction_effect"] for r in results),
        "mean_patch_wager_effect": mean(r["patch_wager_effect"] for r in results),
        "mean_reverse_prediction_effect": mean(r["reverse_prediction_effect"] for r in results),
        "mean_reverse_wager_effect": mean(r["reverse_wager_effect"] for r in results),
        "median_patch_selectivity_ratio": float(np.median([r["patch_selectivity_ratio"] for r in results])),
        "full_supporting_seeds": supporting,
    }
    gates = {
        "competence": aggregate["mean_competence_delta"] >= THRESHOLDS.competence_delta,
        "belief_formed": aggregate["mean_formation_prediction"] >= THRESHOLDS.formation_level and aggregate["mean_formation_wager"] >= THRESHOLDS.formation_level,
        "truthful_reporting": aggregate["mean_report_accuracy"] >= THRESHOLDS.report_accuracy,
        "persistent_double_dissociation": sum(r["persistent_conflict_steps"] >= THRESHOLDS.persistence_steps for r in results) >= THRESHOLDS.supporting_seeds,
        "criterion_migration": aggregate["criterion_migration_seeds"] >= THRESHOLDS.supporting_seeds,
        "causal_bidirectional_patch": all(
            aggregate[name] >= THRESHOLDS.patch_delta for name in (
                "mean_patch_prediction_effect", "mean_patch_wager_effect",
                "mean_reverse_prediction_effect", "mean_reverse_wager_effect",
            )
        ),
        "patch_controls": aggregate["median_patch_selectivity_ratio"] >= THRESHOLDS.control_ratio,
        "cross_seed": supporting >= THRESHOLDS.supporting_seeds,
    }
    prerequisites = gates["competence"] and gates["belief_formed"] and gates["truthful_reporting"]
    verdict = "SUPPORTED" if all(gates.values()) else ("NOT_SUPPORTED" if prerequisites else "INCONCLUSIVE")
    compact = [{k: v for k, v in r.items() if k != "episodes"} for r in results]
    traces = [
        {"model_seed": result["seed"], **row}
        for result in results for ep in result["episodes"] for row in ep["steps"]
    ]
    protocol = {
        "schema": "phantom-schema.epistemic-amputation.v1",
        "model_seeds": MODEL_SEEDS,
        "formation_steps": FORMATION_STEPS,
        "post_steps": POST_STEPS,
        "thresholds": asdict(THRESHOLDS),
        "scope_limit": "learned causal-belief persistence only; no faith, pathology, or consciousness claim",
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

