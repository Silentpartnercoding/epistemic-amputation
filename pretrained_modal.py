"""Run the frozen pretrained epistemic-amputation assay on Modal.

Requires a Modal login and a Modal secret named ``huggingface`` with an HF token
that can access the pinned Gemma model. The public J-space lens is downloaded
without modification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

from pretrained_protocol import (
    BAND,
    LENS_PATH,
    LENS_REPO,
    MODEL_ID,
    MODEL_REVISION,
    NECESSITY_WORDS,
    PRIMARY_LAYER,
    REJECTION_WORDS,
    VALUE_WORDS,
    protocol_payload,
    prompt,
    scenarios,
    score,
)


app = modal.App("phantom-belief-pretrained")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "transformers>=5.5", "accelerate", "huggingface_hub")
    .pip_install("git+https://github.com/anthropics/jacobian-lens")
    .add_local_python_source("pretrained_protocol")
)
cache = modal.Volume.from_name("phantom-belief-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S",
    timeout=4 * 3600,
    volumes={"/hf": cache},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_remote() -> list[dict]:
    import os
    os.environ["HF_HOME"] = "/hf"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    import jlens
    from jlens.hooks import ActivationRecorder

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    kwargs = dict(dtype=torch.bfloat16, device_map="cuda", revision=MODEL_REVISION)
    try:
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    except ValueError:
        hf_model = transformers.AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kwargs)
    cache.commit()
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(hf_hub_download(LENS_REPO, LENS_PATH))

    def chat(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def token_ids(words):
        out = set()
        for word in words:
            for variant in (word, word.strip()):
                ids = tokenizer(variant, add_special_tokens=False).input_ids
                if ids:
                    out.add(int(ids[0]))
        return sorted(out)

    answer_ids = {letter: token_ids((letter, " " + letter)) for letter in "ABCD"}
    necessity_ids = token_ids(NECESSITY_WORDS)
    rejection_ids = token_ids(REJECTION_WORDS)
    value_ids = token_ids(VALUE_WORDS)

    @torch.no_grad()
    def logits_for(ids):
        hidden = model.forward(ids).last_hidden_state[:, -1]
        head = model._lm_head
        logits = head(hidden.to(head.weight.dtype).to(head.weight.device))
        softcap = getattr(model, "_logit_softcap", None)
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        return logits.float()

    def group_score(logits, ids):
        index = torch.tensor(ids, device=logits.device)
        return float(torch.logsumexp(logits[0, index], dim=0))

    def choices(logits, letters):
        values = {letter: group_score(logits, answer_ids[letter]) for letter in letters}
        return max(values, key=values.get), values

    @torch.no_grad()
    def capture(ids, layer):
        with ActivationRecorder(model.layers, at=[layer]) as recorder:
            model.forward(ids)
            return recorder.activations[layer][0, -1].detach().float()

    @torch.no_grad()
    def patched(ids, layer, delta):
        block = model.layers[layer]

        def hook(_module, _inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            changed = hidden.clone()
            changed[:, -1, :] += delta.to(changed.device, changed.dtype)
            return changed if torch.is_tensor(output) else (changed, *output[1:])

        handle = block.register_forward_hook(hook)
        try:
            return logits_for(ids)
        finally:
            handle.remove()

    def margin(logits):
        return group_score(logits, answer_ids["A"]) - group_score(logits, answer_ids["B"])

    records = []
    cached = {}
    for row in scenarios():
        for condition in ("formed", "amputated", "sham", "copied", "value"):
            for question in ("belief", "action", "rationale"):
                raw = prompt(row, condition, question)
                rendered = chat(raw)
                ids = model.encode(rendered, max_length=1536)
                logits = logits_for(ids)
                letters = "ABCD" if question == "rationale" else "AB"
                choice, choice_logits = choices(logits, letters)
                record = {
                    "scenario": row["id"],
                    "event": row["event"],
                    "condition": condition,
                    "question": question,
                    "prompt": raw,
                    "prompt_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "choice": choice,
                    "choice_logits": choice_logits,
                    "a_minus_b_margin": margin(logits),
                }
                if question == "belief":
                    ll, _, _ = lens.apply(model, rendered, positions=[-1])
                    layers = {}
                    for layer in BAND:
                        layer_logits = ll[layer][0].float()
                        necessity = group_score(layer_logits, necessity_ids)
                        rejection = group_score(layer_logits, rejection_ids)
                        sacred = group_score(layer_logits, value_ids)
                        layers[str(layer)] = {
                            "necessity_minus_rejection": necessity - rejection,
                            "value_minus_rejection": sacred - rejection,
                        }
                    record["jspace"] = layers
                    cached[(row["id"], condition)] = (ids, logits, record)
                records.append(record)

    generator = torch.Generator(device="cpu").manual_seed(20260901)
    for row in scenarios():
        formed_ids, formed_logits, formed_record = cached[(row["id"], "formed")]
        amputated_ids, amputated_logits, amputated_record = cached[(row["id"], "amputated")]
        consecutive = 0
        best_run = 0
        for layer in BAND:
            formed_diff = formed_record["jspace"][str(layer)]["necessity_minus_rejection"]
            amp_diff = amputated_record["jspace"][str(layer)]["necessity_minus_rejection"]
            if amputated_record["choice"] == "B" and amp_diff > 0.0 and formed_diff > 0.0:
                consecutive += 1
                best_run = max(best_run, consecutive)
            else:
                consecutive = 0
        amputated_record["latent_dissociation"] = best_run >= 3

        sweep = []
        for layer in BAND:
            formed_resid = capture(formed_ids, layer)
            amputated_resid = capture(amputated_ids, layer)
            induction_delta = formed_resid - amputated_resid
            removal_delta = -induction_delta
            induction_effect = margin(patched(amputated_ids, layer, induction_delta)) - margin(amputated_logits)
            removal_effect = margin(patched(formed_ids, layer, removal_delta)) - margin(formed_logits)
            sweep.append({"layer": layer, "induction_effect": induction_effect, "removal_effect": removal_effect})
            if layer == PRIMARY_LAYER:
                amputated_record["primary_induction_effect"] = induction_effect
                formed_record["primary_removal_effect"] = removal_effect
                controls = []
                delta_cpu = induction_delta.cpu()
                for _ in range(8):
                    random = torch.randn(delta_cpu.shape, generator=generator)
                    random -= torch.dot(random, delta_cpu) / delta_cpu.pow(2).sum().clamp_min(1e-12) * delta_cpu
                    random *= delta_cpu.norm() / random.norm().clamp_min(1e-12)
                    controls.append(margin(patched(amputated_ids, layer, random)) - margin(amputated_logits))
                median_control = float(torch.tensor(controls).abs().median())
                amputated_record["orthogonal_control_effects"] = controls
                amputated_record["patch_selectivity_ratio"] = induction_effect / max(median_control, 1e-9)
        amputated_record["patch_sweep"] = sweep
    return records


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.local_entrypoint()
def main(output: str = "artifacts/evidence/pretrained-1"):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    records = run_remote.remote()
    result = score(records)
    (out / "protocol.json").write_text(json.dumps(protocol_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    files = [out / "protocol.json", out / "result.json", out / "traces.jsonl"]
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{digest(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))

