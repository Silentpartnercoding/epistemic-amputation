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
.venv/bin/python -m unittest discover -s tests -v
```

Read `PREREGISTRATION.md` before inspecting results. The experiment writes a
machine-readable result, per-seed traces, learned-state interventions, hashes,
and a bounded verdict. A positive result is evidence for a learned,
belief-like causal representation in this assay. It is not evidence of pain,
consciousness, subjective experience, or biological equivalence.
