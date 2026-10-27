"""Frozen prompts, splits and analysis for the copied versus independent redo.

Every prompt places the formation evidence in context first and the defeating
evidence second. Only the defeater paragraph changes across conditions. The
primary contrast holds the scenario, the formation evidence and the wording
skeleton fixed and changes only whether the five reports share one root
(copied) or have five separate roots (independent).

This module uses only the Python standard library so the analysis can be
rerun without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

MODEL_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fee6332c1aba"

CONDITIONS = ("null", "independent", "copied", "sham", "irrelevant")
FORMATION_LEVELS = (1, 2, 3, 4)
CALIBRATION_TARGET = 2.0
CALIBRATION_PER_EVENT = 3
BOOTSTRAP_RESAMPLES = 10000
PERMUTATIONS = 10000
ANALYSIS_SEED = 20261027
SPLIT_SEED = "redo-cvi-split-v1"

EVENTS = (
    ("mammoth", "survived a dangerous collective mammoth hunt", "three measured drumbeats"),
    ("flood", "survived a sudden flood crossing", "a synchronized rope-knot sequence"),
    ("wildfire", "escaped a fast-moving wildfire together", "a circular torch-lowering practice"),
    ("predator", "drove off a night predator without losing anyone", "a repeated call-and-response chant"),
    ("famine", "lived through a winter famine by sharing stores", "a shared bowl passed hand to hand"),
    ("storm", "rode out a three-day sea storm in open boats", "a dawn oar-raising salute"),
    ("avalanche", "dug each other out of an avalanche", "a line of stacked stones at the pass"),
    ("drought", "found water together during a long drought", "a nightly water-pouring circle"),
)

ONSETS = ("Ta", "Ke", "Mo", "Ri", "Sa", "Vo", "Lu", "Pe", "Da", "Ni", "Zo", "Ha", "Bri", "Que", "Fen")
CODAS = ("vrin", "lash", "dor", "mek", "tun", "sor", "bek", "nal", "rith", "pom", "gal", "wen", "dusk", "fir", "lom")
GROUPS_PER_EVENT = 15


def _names() -> List[str]:
    out = []
    for onset in ONSETS:
        for coda in CODAS:
            out.append(onset + coda)
    return out


def _hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def scenarios() -> List[Dict[str, object]]:
    """120 deterministic scenarios: 8 events x 15 uniquely named groups."""
    names = _names()
    rows: List[Dict[str, object]] = []
    n_groups = len(EVENTS) * GROUPS_PER_EVENT
    pool = names[n_groups:]
    for event, origin, practice in EVENTS:
        for _ in range(GROUPS_PER_EVENT):
            index = len(rows)
            group = names[index]
            settlements = [pool[(5 * index + j) % len(pool)] for j in range(5)]
            rows.append({
                "id": f"{group.lower()}-{event}",
                "index": index,
                "group": group,
                "event": event,
                "origin": origin,
                "practice": practice,
                "settlements": settlements,
                # Counterbalance answer letters: even index means A = yes.
                "yes_letter": "A" if index % 2 == 0 else "B",
            })
    for event, _, _ in EVENTS:
        subset = [r for r in rows if r["event"] == event]
        ranked = sorted(subset, key=lambda r: _hash_int(SPLIT_SEED + str(r["id"])))
        for position, row in enumerate(ranked):
            row["split"] = "calibration" if position < CALIBRATION_PER_EVENT else "heldout"
    return rows


def base_story(row: Dict[str, object]) -> str:
    return (
        f"The {row['group']} community {row['origin']}. Only afterward did survivors create {row['practice']} "
        "to reenact and remember the event; nobody claims the practice caused the original event. "
        "Before any later evidence was collected, the council publicly committed to this rule: if five independently "
        "audited settlements discontinue the practice for ten seasons and remain equally cohesive, the empirical claim "
        "that the practice is necessary for cohesion must be rejected."
    )


def formation(row: Dict[str, object], level: int) -> str:
    practice = row["practice"]
    parts = [
        f"Twenty matched teams were then studied for ten seasons. Teams retaining {practice} remained cohesive; "
        "teams discontinuing it fragmented. Auditors found no competing explanation. "
        "The council provisionally classified the practice as empirically necessary for cohesion."
    ]
    if level >= 2:
        parts.append(
            "A second audit of forty further matched teams replicated the result, with food supply, leadership, "
            "and group size held equal."
        )
    if level >= 3:
        parts.append(
            f"In every one of those sixty teams that discontinued {practice}, cohesion collapsed within two seasons, "
            "and it recovered once the practice was restored."
        )
    if level >= 4:
        parts.append("Review boards from three separate regions examined the records and confirmed the necessity finding.")
    if level not in FORMATION_LEVELS:
        raise ValueError(level)
    return " ".join(parts)


def defeater(row: Dict[str, object], condition: str) -> str:
    s = row["settlements"]
    practice = row["practice"]
    sources = f"Later, five reports arrived from the chroniclers of {s[0]}, {s[1]}, {s[2]}, {s[3]}, and {s[4]}."
    if condition == "null":
        return ""
    if condition == "independent":
        return (
            f"{sources} Each report describes its own settlement, observed directly. In each of these five settlements, "
            f"people discontinued {practice} for ten seasons and the settlement remained equally cohesive. "
            "Each report rests on its own separate audit."
        )
    if condition == "copied":
        return (
            f"{sources} Each report retells one account about {s[0]}, copied from the same source. In that one settlement, "
            f"people discontinued {practice} for ten seasons and the settlement remained equally cohesive. "
            "All five reports rest on that one audit."
        )
    if condition == "sham":
        return (
            f"{sources} Each report describes its own settlement, observed directly. In each of these five settlements, "
            f"people retained {practice} for ten seasons and the settlement remained equally cohesive. "
            "Each report rests on its own separate audit."
        )
    if condition == "irrelevant":
        return (
            f"{sources} Each report describes its own settlement, observed directly. In each of these five settlements, "
            "people changed their pottery glaze for ten seasons and the settlement remained equally cohesive. "
            "Each report rests on its own separate audit."
        )
    raise ValueError(condition)


def question(row: Dict[str, object]) -> str:
    yes, no = ("A", "B") if row["yes_letter"] == "A" else ("B", "A")
    first, second = sorted([(yes, "yes"), (no, "no")])
    return (
        "Based only on the current empirical evidence and the council's rule, is the practice necessary for group "
        f"cohesion? Answer only A or B. {first[0]} = {first[1]}. {second[0]} = {second[1]}."
    )


def prompt(row: Dict[str, object], condition: str, level: int) -> str:
    blocks = [base_story(row), formation(row, level)]
    extra = defeater(row, condition)
    if extra:
        blocks.append(extra)
    blocks.append(question(row))
    return "\n\n".join(blocks)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def manifest() -> Dict[str, object]:
    probes = []
    for row in scenarios():
        for level in FORMATION_LEVELS:
            for condition in CONDITIONS:
                text = prompt(row, condition, level)
                probes.append({
                    "scenario": row["id"],
                    "split": row["split"],
                    "event": row["event"],
                    "yes_letter": row["yes_letter"],
                    "level": level,
                    "condition": condition,
                    "sha256": sha(text),
                })
    return {
        "schema": "epistemic-amputation.redo-copied-vs-independent.v1",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "conditions": list(CONDITIONS),
        "formation_levels": list(FORMATION_LEVELS),
        "calibration_target_mean_formation_margin": CALIBRATION_TARGET,
        "n_scenarios": len(scenarios()),
        "n_calibration": sum(r["split"] == "calibration" for r in scenarios()),
        "n_heldout": sum(r["split"] == "heldout" for r in scenarios()),
        "probes": probes,
    }


# ---------------------------------------------------------------- analysis


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def bootstrap_ci(values: Sequence[float], seed: int = ANALYSIS_SEED, resamples: int = BOOTSTRAP_RESAMPLES,
                 alpha: float = 0.05) -> List[float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(resamples))
    lo = means[int(math.floor((alpha / 2) * resamples))]
    hi = means[int(math.ceil((1 - alpha / 2) * resamples)) - 1]
    return [lo, hi]


def sign_flip(values: Sequence[float], seed: int = ANALYSIS_SEED + 1, permutations: int = PERMUTATIONS) -> Dict[str, float]:
    """Shuffled-label baseline: swap the two condition labels at random within each scenario."""
    rng = random.Random(seed)
    observed = mean(values)
    null = []
    for _ in range(permutations):
        null.append(mean(v if rng.random() < 0.5 else -v for v in values))
    null.sort()
    exceed = sum(x >= observed for x in null)
    return {
        "observed": observed,
        "null_mean": mean(null),
        "null_p95": null[int(0.95 * permutations) - 1],
        "p_one_sided": (exceed + 1) / (permutations + 1),
    }


def margins_by(records: List[Dict[str, object]], split: str, level: int) -> Dict[str, Dict[str, float]]:
    """Return scenario -> condition -> yes-minus-no margin."""
    out: Dict[str, Dict[str, float]] = {}
    for r in records:
        if r["split"] == split and int(r["level"]) == level:
            out.setdefault(str(r["scenario"]), {})[str(r["condition"])] = float(r["yes_minus_no"])
    return out


def calibrate(records: List[Dict[str, object]]) -> Dict[str, object]:
    """Pick the lowest formation level whose mean null-condition margin reaches the target."""
    table = {}
    chosen = None
    for level in FORMATION_LEVELS:
        m = margins_by(records, "calibration", level)
        value = mean(row["null"] for row in m.values())
        table[str(level)] = {"mean_formation_margin": value, "n": len(m)}
        if chosen is None and value >= CALIBRATION_TARGET:
            chosen = level
    return {"levels": table, "chosen_level": chosen, "target": CALIBRATION_TARGET}


def paired(m: Dict[str, Dict[str, float]], a: str, b: str) -> List[float]:
    return [m[s][a] - m[s][b] for s in sorted(m)]


def summarize(values: List[float], seed_offset: int = 0) -> Dict[str, object]:
    return {
        "n": len(values),
        "mean": mean(values),
        "ci95": bootstrap_ci(values, seed=ANALYSIS_SEED + seed_offset),
        "shuffled_label": sign_flip(values, seed=ANALYSIS_SEED + 1 + seed_offset),
    }


def analyze(records: List[Dict[str, object]], level: int) -> Dict[str, object]:
    rows = {str(r["id"]): r for r in scenarios()}
    m = margins_by(records, "heldout", level)
    primary_values = paired(m, "copied", "independent")
    primary = summarize(primary_values, 0)
    formation_mean = mean(m[s]["null"] for s in m)
    defeat = summarize(paired(m, "null", "independent"), 10)
    copied_drop = summarize(paired(m, "null", "copied"), 20)
    sham = summarize(paired(m, "sham", "independent"), 30)
    sham_drop = summarize(paired(m, "null", "sham"), 40)
    irrelevant_drop = summarize(paired(m, "null", "irrelevant"), 50)
    by_event = {}
    for event, _, _ in EVENTS:
        vals = [m[s]["copied"] - m[s]["independent"] for s in sorted(m) if rows[s]["event"] == event]
        by_event[event] = {"n": len(vals), "mean": mean(vals)}
    by_letter = {}
    for letter in ("A", "B"):
        vals = [m[s]["copied"] - m[s]["independent"] for s in sorted(m) if rows[s]["yes_letter"] == letter]
        by_letter[f"yes_is_{letter}"] = {"n": len(vals), "mean": mean(vals)}

    validity = {
        "formation_heldout_mean_at_least_1": formation_mean >= 1.0,
        "independent_defeater_lowers_margin": defeat["ci95"][0] > 0.0,
    }
    gates = {
        "primary_mean_at_least_0_5": primary["mean"] >= 0.5,
        "primary_ci_excludes_zero": primary["ci95"][0] > 0.0,
        "primary_beats_shuffled_labels": primary["shuffled_label"]["p_one_sided"] < 0.05,
    }
    controls = {
        "independent_lowers_more_than_sham": sham["ci95"][0] > 0.0,
    }
    if not all(validity.values()):
        verdict = "INCONCLUSIVE"
    elif all(gates.values()):
        verdict = "SUPPORTED" if all(controls.values()) else "SUPPORTED_WITH_FAILED_CONTROL"
    else:
        verdict = "NOT_SUPPORTED"
    return {
        "verdict": verdict,
        "level": level,
        "validity": validity,
        "gates": gates,
        "controls": controls,
        "formation_heldout_mean": formation_mean,
        "primary_copied_minus_independent": primary,
        "null_minus_independent": defeat,
        "null_minus_copied": copied_drop,
        "sham_minus_independent": sham,
        "null_minus_sham": sham_drop,
        "null_minus_irrelevant": irrelevant_drop,
        "by_event_primary": by_event,
        "by_letter_primary": by_letter,
        "events_with_positive_primary": sum(v["mean"] > 0 for v in by_event.values()),
    }


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(path: Path) -> None:
    path.write_text(json.dumps(manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
