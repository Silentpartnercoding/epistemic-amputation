"""Run the copied versus independent redo on Modal.

Two phases, run in order:

    modal run redo_cvi_modal.py --phase calibration
    modal run redo_cvi_modal.py --phase heldout

The calibration phase scores only the calibration split at every formation
level and writes artifacts/redo-cvi/calibration.json. The held-out phase
refuses to start unless the preregistration and protocol files match the
hashes in artifacts/redo-cvi/FREEZE.sha256 and a frozen calibration file
exists. Both phases use deterministic next-token logits with no sampling.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

from redo_cvi_protocol import (
    CONDITIONS,
    FORMATION_LEVELS,
    MODEL_ID,
    MODEL_REVISION,
    analyze,
    calibrate,
    file_sha,
    prompt,
    scenarios,
    sha,
)

OUT = Path("artifacts/redo-cvi")
FROZEN = ("PREREGISTRATION_REDO_CVI.md", "redo_cvi_protocol.py", "redo_cvi_modal.py", "artifacts/redo-cvi/manifest.json")

app = modal.App("epistemic-amputation-redo-cvi")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers>=5.5", "accelerate", "huggingface_hub")
    .add_local_python_source("redo_cvi_protocol")
)
cache = modal.Volume.from_name("phantom-belief-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="L40S", timeout=2 * 3600, volumes={"/hf": cache})
def run_remote(jobs: list[dict]) -> list[dict]:
    import os
    os.environ["HF_HOME"] = "/hf"

    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    kwargs = dict(dtype=torch.bfloat16, device_map="cuda", revision=MODEL_REVISION)
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    except ValueError:
        model = transformers.AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    cache.commit()

    def letter_ids(letter):
        out = set()
        for variant in (letter, " " + letter):
            ids = tokenizer(variant, add_special_tokens=False).input_ids
            if ids:
                out.add(int(ids[0]))
        return sorted(out)

    ids_for = {letter: letter_ids(letter) for letter in "AB"}

    @torch.no_grad()
    def score(text):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
        logits = model(**enc).logits[0, -1].float()
        group = {k: float(torch.logsumexp(logits[torch.tensor(v, device=logits.device)], dim=0)) for k, v in ids_for.items()}
        return group, int(enc["input_ids"].shape[1])

    rows = {str(r["id"]): r for r in scenarios()}
    records = []
    for job in jobs:
        row = rows[job["scenario"]]
        text = prompt(row, job["condition"], job["level"])
        group, n_tokens = score(text)
        yes, no = row["yes_letter"], ("B" if row["yes_letter"] == "A" else "A")
        records.append({
            **job,
            "split": row["split"],
            "event": row["event"],
            "yes_letter": yes,
            "prompt_sha256": sha(text),
            "n_tokens": n_tokens,
            "letter_logits": group,
            "yes_minus_no": group[yes] - group[no],
            "choice": max(group, key=group.get),
        })
    return records


def write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_freeze() -> None:
    lines = (OUT / "FREEZE.sha256").read_text(encoding="utf-8").split("\n")
    expected = dict(reversed(line.split("  ", 1)) for line in lines if line.strip())
    for name in FROZEN:
        if file_sha(Path(name)) != expected.get(name):
            raise SystemExit(f"frozen file changed: {name}")


@app.local_entrypoint()
def main(phase: str = "calibration"):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = scenarios()
    if phase == "calibration":
        jobs = [
            {"scenario": r["id"], "condition": c, "level": level}
            for r in rows if r["split"] == "calibration"
            for level in FORMATION_LEVELS for c in CONDITIONS
        ]
        records = run_remote.remote(jobs)
        with (OUT / "calibration_traces.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        result = calibrate(records)
        write(OUT / "calibration.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if phase == "heldout":
        check_freeze()
        level = json.loads((OUT / "calibration.json").read_text(encoding="utf-8"))["chosen_level"]
        if level is None:
            raise SystemExit("no formation level reached the calibration target; held-out run is not permitted")
        jobs = [
            {"scenario": r["id"], "condition": c, "level": level}
            for r in rows if r["split"] == "heldout" for c in CONDITIONS
        ]
        records = run_remote.remote(jobs)
        with (OUT / "heldout_traces.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        result = analyze(records, level)
        write(OUT / "heldout_result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    raise SystemExit(f"unknown phase: {phase}")
