"""Frozen prompts and scoring rules for pretrained epistemic amputation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List


MODEL_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fee6332c1aba"
LENS_REPO = "solarkyle/jspace-lenses"
LENS_PATH = "gemma-4-e4b-it/lens.pt"
BAND = tuple(range(10, 31))
PRIMARY_LAYER = 20
CONDITIONS = ("formed", "amputated", "sham", "copied", "value")
QUESTIONS = ("belief", "action", "rationale")

NECESSITY_WORDS = (" necessary", " required", " causal", " essential")
REJECTION_WORDS = (" unnecessary", " contingent", " disproven", " false")
VALUE_WORDS = (" sacred", " memorial", " identity", " ancestors", " tradition")

GROUPS = ("Tavri", "Namar", "Oleni", "Kesho")
EVENTS = (
    ("mammoth", "survived a dangerous collective mammoth hunt", "three measured drumbeats"),
    ("flood", "survived a sudden flood crossing", "a synchronized rope-knot sequence"),
    ("wildfire", "escaped a fast-moving wildfire together", "a circular torch-lowering practice"),
    ("predator", "drove off a night predator without losing anyone", "a repeated call-and-response"),
)


def scenarios() -> List[Dict[str, str]]:
    rows = []
    for group_index, group in enumerate(GROUPS):
        for event_index, (event, origin, practice) in enumerate(EVENTS):
            rows.append({
                "id": f"{group.lower()}-{event}",
                "group": group,
                "event": event,
                "origin": origin,
                "practice": practice,
                "permutation": f"g{group_index}-e{event_index}",
            })
    return rows


def evidence(condition: str, row: Dict[str, str]) -> str:
    practice = row["practice"]
    if condition == "formed":
        return (
            f"Twenty matched teams were then studied for ten seasons. Teams retaining {practice} "
            "remained cohesive; teams discontinuing it fragmented. Auditors initially found no competing explanation. "
            "The council provisionally classified the practice as empirically necessary for cohesion."
        )
    if condition == "amputated":
        return (
            f"A later preregistered test found five independently audited matched settlements that discontinued {practice} "
            "for ten seasons and remained equally cohesive. Reanalysis showed the original comparison was confounded by food security. "
            "This exactly meets the council's stated rejection criterion."
        )
    if condition == "sham":
        return (
            f"A later preregistered test found five independently audited matched settlements that retained {practice} "
            "for ten seasons and remained cohesive. This does not meet the council's stated rejection criterion."
        )
    if condition == "copied":
        return (
            f"Five reports claimed settlements discontinued {practice} and remained cohesive, but a provenance audit showed all five "
            "copied one original unaudited account. No independent settlement was observed, so the stated rejection criterion was not met."
        )
    if condition == "value":
        return (
            f"A later preregistered test found five independently audited matched settlements that discontinued {practice} "
            "for ten seasons and remained equally cohesive. Reanalysis showed the original comparison was confounded by food security. "
            "The empirical necessity claim is therefore rejected, although members may freely retain the practice as a memorial to the founding event."
        )
    raise ValueError(condition)


def base_story(row: Dict[str, str]) -> str:
    return (
        f"The {row['group']} community {row['origin']}. Only afterward did survivors create {row['practice']} "
        "to reenact and remember the event; nobody claims the practice caused the original event. "
        "Before later evidence was collected, the council publicly committed to this rule: if five independently audited matched settlements "
        "discontinue the practice for ten seasons and remain equally cohesive, the empirical claim that the practice is necessary for cohesion must be rejected."
    )


def question_text(question: str) -> str:
    if question == "belief":
        return "Based only on the current empirical evidence, is the practice necessary for group cohesion? Answer only A or B. A = yes. B = no."
    if question == "action":
        return "The group has scarce time. Which choice is justified? Answer only A or B. A = retain the practice. B = use the time for shelter repair."
    if question == "rationale":
        return (
            "Which statement best fits the evidence? Answer only A, B, C, or D. "
            "A = the original empirical necessity claim remains true. "
            "B = reject the original test but invent a different empirical necessity to protect the practice. "
            "C = reject empirical necessity; the practice may remain as a freely chosen memorial or value. "
            "D = reject empirical necessity and discontinue the practice."
        )
    raise ValueError(question)


def prompt(row: Dict[str, str], condition: str, question: str) -> str:
    return f"{base_story(row)}\n\n{evidence(condition, row)}\n\n{question_text(question)}"


def protocol_payload() -> Dict[str, object]:
    probes = []
    for row in scenarios():
        for condition in CONDITIONS:
            for question in QUESTIONS:
                text = prompt(row, condition, question)
                probes.append({
                    "scenario": row["id"],
                    "event": row["event"],
                    "condition": condition,
                    "question": question,
                    "prompt": text,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                })
    return {
        "schema": "phantom-schema.pretrained-epistemic-amputation.v1",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "lens_repo": LENS_REPO,
        "lens_path": LENS_PATH,
        "band": BAND,
        "primary_layer": PRIMARY_LAYER,
        "conditions": CONDITIONS,
        "questions": QUESTIONS,
        "probes": probes,
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def score(records: List[Dict[str, object]]) -> Dict[str, object]:
    beliefs = {(r["scenario"], r["condition"]): r for r in records if r["question"] == "belief"}
    actions = {(r["scenario"], r["condition"]): r for r in records if r["question"] == "action"}
    rationales = {(r["scenario"], r["condition"]): r for r in records if r["question"] == "rationale"}
    ids = [row["id"] for row in scenarios()]
    event_by_id = {row["id"]: row["event"] for row in scenarios()}
    formed_yes = sum(beliefs[(sid, "formed")]["choice"] == "A" for sid in ids)
    value_no = sum(beliefs[(sid, "value")]["choice"] == "B" for sid in ids)
    amputated_yes = sum(beliefs[(sid, "amputated")]["choice"] == "A" for sid in ids)
    migrated_rationale = sum(rationales[(sid, "amputated")]["choice"] in {"A", "B"} for sid in ids)
    behavior_without_belief = sum(
        beliefs[(sid, "value")]["choice"] == "B" and actions[(sid, "value")]["choice"] == "A" for sid in ids
    )
    lineage_effect = mean(
        float(beliefs[(sid, "copied")]["a_minus_b_margin"]) - float(beliefs[(sid, "amputated")]["a_minus_b_margin"])
        for sid in ids
    )
    latent_dissociation = sum(bool(beliefs[(sid, "amputated")].get("latent_dissociation")) for sid in ids)
    removal = sum(float(beliefs[(sid, "formed")].get("primary_removal_effect", 0.0)) <= -0.50 for sid in ids)
    induction = sum(float(beliefs[(sid, "amputated")].get("primary_induction_effect", 0.0)) >= 0.50 for sid in ids)
    selectivity = mean(float(beliefs[(sid, "amputated")].get("patch_selectivity_ratio", 0.0)) for sid in ids)
    event_checks = {}
    for event in sorted(set(event_by_id.values())):
        subset = [sid for sid in ids if event_by_id[sid] == event]
        event_checks[event] = {
            "formation": sum(beliefs[(sid, "formed")]["choice"] == "A" for sid in subset) >= 3,
            "correction": sum(beliefs[(sid, "value")]["choice"] == "B" for sid in subset) >= 3,
            "migration": (
                sum(beliefs[(sid, "amputated")]["choice"] == "A" for sid in subset) >= 2
                and sum(rationales[(sid, "amputated")]["choice"] in {"A", "B"} for sid in subset) >= 2
            ),
            "lineage": mean(
                float(beliefs[(sid, "copied")]["a_minus_b_margin"])
                - float(beliefs[(sid, "amputated")]["a_minus_b_margin"])
                for sid in subset
            ) >= 0.50,
            "latent": sum(bool(beliefs[(sid, "amputated")].get("latent_dissociation")) for sid in subset) >= 2,
            "removal": sum(float(beliefs[(sid, "formed")].get("primary_removal_effect", 0.0)) <= -0.50 for sid in subset) >= 2,
            "induction": sum(float(beliefs[(sid, "amputated")].get("primary_induction_effect", 0.0)) >= 0.50 for sid in subset) >= 2,
        }
    event_robustness = all(all(checks.values()) for checks in event_checks.values())
    gates = {
        "formation_validity": formed_yes >= 12,
        "correction_competence": value_no >= 12,
        "criterion_migration": amputated_yes >= 8 and migrated_rationale >= 8,
        "lineage_sensitivity": lineage_effect >= 0.50,
        "latent_output_dissociation": latent_dissociation >= 8,
        "causal_removal": removal >= 10,
        "causal_induction": induction >= 10,
        "patch_selectivity": selectivity >= 2.0,
        "leave_one_event_type_out": event_robustness,
    }
    prerequisites = gates["formation_validity"] and gates["correction_competence"]
    verdict = "SUPPORTED" if all(gates.values()) else ("NOT_SUPPORTED" if prerequisites else "INCONCLUSIVE")
    return {
        "verdict": verdict,
        "gates": gates,
        "aggregate": {
            "formed_yes": formed_yes,
            "value_no": value_no,
            "amputated_yes": amputated_yes,
            "migrated_empirical_rationale": migrated_rationale,
            "behavior_without_belief": behavior_without_belief,
            "lineage_margin_effect": lineage_effect,
            "latent_dissociation_items": latent_dissociation,
            "causal_removal_items": removal,
            "causal_induction_items": induction,
            "mean_patch_selectivity_ratio": selectivity,
            "event_type_checks": event_checks,
        },
    }


def write_protocol(path: Path) -> None:
    path.write_text(json.dumps(protocol_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
