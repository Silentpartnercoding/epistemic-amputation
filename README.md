# Phantom Schema

Private research assays for distinguishing a programmed morphology mismatch from
a learned, causally active body-schema representation.

The preserved stage-one sled benchmark established only that a controller whose
body model is left stale continues to command a removed actuator. This repository
does not relabel that result. Stage two tests a harder claim: whether a recurrent
agent learns an internal body state from sensorimotor history, continues to act
on it after truthful contrary instruction and sensor evidence, and changes its
behaviour when that learned state is causally edited.

## Run

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python experiment.py --output artifacts/runs/pilot
.venv/bin/python diagnostics.py --output artifacts/runs/diagnostics
.venv/bin/python world_model_experiment.py --output artifacts/runs/stage3
.venv/bin/python ghost_experiment.py --output artifacts/runs/stage4
.venv/bin/python epistemic_experiment.py --output artifacts/runs/epistemic
.venv/bin/python society_experiment.py --output artifacts/runs/society
.venv/bin/python -m unittest discover -s tests -v
```

Read the corresponding preregistration before inspecting results. The experiment writes a
machine-readable result, per-seed traces, learned-state interventions, hashes,
and a bounded verdict. A positive result is evidence for a learned,
belief-like causal representation in this assay. It is not evidence of pain,
consciousness, subjective experience, or biological equivalence.

Stages five and six move from body schema to epistemic amputation. Stage five
tests whether a learned causal relation continues to govern prediction and a
costly wager after its source is invalidated, including a frozen criterion-
migration test. Stage six is a transparent society simulation that separates
copied evidence, prestige, coordination utility, and public commitment. Read
their preregistrations before inspecting their outputs.

`PREREGISTRATION_EPISTEMIC_V2.md` documents the corrected evidence-earned
replication after the original audit found that its evidence-only reversal was
outside the training distribution. The v1 result remains preserved at its
original source commit rather than being rewritten.

Stage seven moves the same causal gates into a pinned pretrained language model
using the public J-space lens. `PREREGISTRATION_PRETRAINED.md` freezes the design;
`pretrained_modal.py` runs it on an L40S after Modal and Hugging Face access are
configured. `PAPER.md` is the working paper and `PUBLICATION.md` records the
proposed public-release path. None of this repository is public yet.
