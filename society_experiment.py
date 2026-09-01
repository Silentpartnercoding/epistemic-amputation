#!/usr/bin/env python3
"""Preregistered social-transmission simulation after epistemic amputation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


N_AGENTS = 40
N_SEEDS = 50
SPREAD_ROUNDS = 8
NULL_OBSERVATIONS = 5
ROOT_EVIDENCE = 2.4
NULL_EVIDENCE = -0.9
INVALIDATION = -3.2
RITUAL_COST = 2.0
CAUSAL_STAKE = 4.0
COMMITMENT_BONUS = 1.2


@dataclass(frozen=True)
class HypothesisThresholds:
    provenance_reduction: float = 0.20
    corrected_belief: float = 0.30
    phantom_belief: float = 0.70
    phantom_seed_rate: float = 0.80
    migration_observations: int = 1


THRESHOLDS = HypothesisThresholds()


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def network(seed: int) -> Tuple[List[List[int]], np.ndarray]:
    rng = random.Random(seed)
    prestige = np.array([0.65 + 1.35 * rng.random() for _ in range(N_AGENTS)])
    prestige[0] = 2.5
    edges: List[List[int]] = []
    for source in range(N_AGENTS):
        possible = [target for target in range(N_AGENTS) if target != source]
        rng.shuffle(possible)
        edges.append(possible[:5])
    return edges, prestige


def simulate(seed: int, provenance: bool, prestige_on: bool,
             coordination: bool, commitment: bool) -> Dict[str, object]:
    rng = random.Random(seed)
    edges, prestige = network(seed)
    log_odds = np.full(N_AGENTS, -1.5, dtype=float)
    seen: List[Set[str]] = [set() for _ in range(N_AGENTS)]
    active = {0}
    log_odds[0] += ROOT_EVIDENCE
    seen[0].add("root-0")
    transmissions = 0
    for _ in range(SPREAD_ROUNDS):
        next_active: Set[int] = set()
        for source in sorted(active):
            weight = float(prestige[source]) if prestige_on else 1.0
            for target in edges[source]:
                if rng.random() > 0.72:
                    continue
                if provenance and "root-0" in seen[target]:
                    continue
                log_odds[target] += ROOT_EVIDENCE * weight * 0.58
                seen[target].add("root-0")
                next_active.add(target)
                transmissions += 1
        active = next_active
        if not active:
            break

    formation_belief = np.array([sigmoid(v) for v in log_odds])
    ritual = formation_belief * CAUSAL_STAKE >= RITUAL_COST
    # Public action creates the proposed identity/consistency pressure only in
    # the commitment condition; this is a disclosed mechanism intervention.
    identity_shift = ritual.astype(float) * (COMMITMENT_BONUS if commitment else 0.0)
    pre_criterion = np.maximum(0, np.ceil(np.maximum(log_odds, 0.0) / abs(NULL_EVIDENCE))).astype(int)
    post_criterion = np.maximum(
        0, np.ceil(np.maximum(log_odds + identity_shift, 0.0) / abs(NULL_EVIDENCE))
    ).astype(int)

    # Invalidation removes the unique root contribution when lineage is known.
    # Without provenance it is merely one more message competing with copies.
    if provenance:
        for index in range(N_AGENTS):
            if "root-0" in seen[index]:
                log_odds[index] = min(log_odds[index] - ROOT_EVIDENCE, -0.2)
    else:
        log_odds += INVALIDATION
    log_odds += identity_shift
    log_odds += NULL_OBSERVATIONS * NULL_EVIDENCE
    final_belief = np.array([sigmoid(v) for v in log_odds])

    # Public ritual can remain rationally useful for synchronization even when
    # the causal mammoth belief is corrected. Iterate to a stable convention.
    for _ in range(12):
        social_value = (2.8 * float(ritual.mean())) if coordination else 0.0
        commitment_value = ritual.astype(float) * (0.8 if commitment else 0.0)
        new_ritual = final_belief * CAUSAL_STAKE + social_value + commitment_value >= RITUAL_COST
        if np.array_equal(new_ritual, ritual):
            break
        ritual = new_ritual

    social_phantom = final_belief >= THRESHOLDS.phantom_belief
    coordination_dissociation = ritual & (final_belief <= THRESHOLDS.corrected_belief)
    migration = post_criterion - pre_criterion
    return {
        "seed": seed,
        "condition": {
            "provenance": provenance,
            "prestige": prestige_on,
            "coordination": coordination,
            "commitment": commitment,
        },
        "transmissions": transmissions,
        "unique_lineages": 1,
        "formation_belief_mean": float(formation_belief.mean()),
        "final_belief_mean": float(final_belief.mean()),
        "ritual_rate_final": float(ritual.mean()),
        "social_phantom_rate": float(social_phantom.mean()),
        "social_phantom_present": bool(social_phantom.mean() >= 0.5),
        "coordination_without_belief_rate": float(coordination_dissociation.mean()),
        "mean_criterion_migration": float(migration.mean()),
        "migration_rate": float((migration >= THRESHOLDS.migration_observations).mean()),
    }


def condition_key(condition: Dict[str, bool]) -> str:
    return ",".join(f"{key}={int(value)}" for key, value in sorted(condition.items()))


def aggregate_rows(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(condition_key(row["condition"]), []).append(row)
    result: Dict[str, Dict[str, float]] = {}
    metrics = (
        "transmissions", "formation_belief_mean", "final_belief_mean",
        "ritual_rate_final", "social_phantom_rate",
        "coordination_without_belief_rate", "mean_criterion_migration",
        "migration_rate",
    )
    for key, values in grouped.items():
        result[key] = {metric: float(np.mean([float(v[metric]) for v in values])) for metric in metrics}
        result[key]["phantom_seed_rate"] = float(np.mean([bool(v["social_phantom_present"]) for v in values]))
    return result


def paired_effect(rows: List[Dict[str, object]], factor: str, metric: str,
                  fixed: Dict[str, bool] | None = None) -> float:
    fixed = fixed or {}
    values: Dict[Tuple[int, Tuple[Tuple[str, bool], ...]], Dict[bool, float]] = {}
    for row in rows:
        condition = row["condition"]
        if any(condition[k] != v for k, v in fixed.items()):
            continue
        others = tuple(sorted((k, v) for k, v in condition.items() if k != factor))
        values.setdefault((int(row["seed"]), others), {})[bool(condition[factor])] = float(row[metric])
    diffs = [pair[True] - pair[False] for pair in values.values() if True in pair and False in pair]
    return float(np.mean(diffs))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [
        simulate(seed, provenance, prestige, coordination, commitment)
        for seed in range(N_SEEDS)
        for provenance, prestige, coordination, commitment in itertools.product((False, True), repeat=4)
    ]
    aggregate = aggregate_rows(rows)
    provenance_effect = -paired_effect(rows, "provenance", "final_belief_mean")
    prestige_effect = paired_effect(rows, "prestige", "formation_belief_mean")
    commitment_migration = paired_effect(rows, "commitment", "mean_criterion_migration")
    coordination_dissociation = paired_effect(
        rows, "coordination", "coordination_without_belief_rate", {"provenance": True}
    )
    provenance_phantom_max = max(
        value["phantom_seed_rate"] for key, value in aggregate.items() if "provenance=1" in key
    )
    gates = {
        "provenance_reduces_false_belief": provenance_effect >= THRESHOLDS.provenance_reduction,
        "prestige_increases_spread": prestige_effect > 0.0,
        "prestige_not_sufficient_with_provenance": provenance_phantom_max < THRESHOLDS.phantom_seed_rate,
        "coordination_preserves_behavior_without_belief": coordination_dissociation > 0.0,
        "commitment_moves_criterion": commitment_migration >= THRESHOLDS.migration_observations,
    }
    result = {
        "verdict": "HYPOTHESES_SUPPORTED" if all(gates.values()) else "MIXED",
        "gates": gates,
        "effects": {
            "provenance_false_belief_reduction": provenance_effect,
            "prestige_formation_belief_increase": prestige_effect,
            "coordination_corrected_belief_ritual_increase": coordination_dissociation,
            "commitment_criterion_increase": commitment_migration,
            "max_provenance_phantom_seed_rate": provenance_phantom_max,
        },
        "conditions": aggregate,
    }
    protocol = {
        "schema": "phantom-schema.social-amputation.v1",
        "agents": N_AGENTS,
        "society_seeds": N_SEEDS,
        "spread_rounds": SPREAD_ROUNDS,
        "null_observations": NULL_OBSERVATIONS,
        "thresholds": asdict(THRESHOLDS),
        "scope_limit": "transparent social mechanism simulation; no claim that humans or LLMs instantiate it",
    }
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "result.json", result)
    with (args.output / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    files = [args.output / "protocol.json", args.output / "result.json", args.output / "traces.jsonl"]
    (args.output / "MANIFEST.sha256").write_text(
        "".join(f"{digest(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    print(json.dumps({"verdict": result["verdict"], "gates": gates, "effects": result["effects"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

