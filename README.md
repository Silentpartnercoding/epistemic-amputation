# Phantom Schema

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22233257.svg)](https://doi.org/10.5281/zenodo.22233257)

Public research assays for distinguishing a programmed morphology mismatch from
a learned, causally active body-schema representation, and for testing the
stronger epistemic analogue in pretrained language models.

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
.venv/bin/modal run pretrained_modal.py --output artifacts/evidence/pretrained-1
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
`pretrained_modal.py` runs it on an L40S after Modal access is configured; the
pinned model and lens are public. [`PAPER.md`](PAPER.md) is the working paper,
[`RESULTS.md`](RESULTS.md) preserves the bounded findings, and
[`PUBLICATION.md`](PUBLICATION.md) records the release path.

The completed stage-seven verdict is `NOT_SUPPORTED`. The model corrected its
explicit empirical answer in all 16 amputation cases. Although the J-space
readout met the frozen latent-dissociation rule, the residual interventions did
not causally restore or remove the answer, so the readout is not evidence of a
causally controlling phantom belief.

The permanent Zenodo archive is available at
[doi:10.5281/zenodo.22233258](https://doi.org/10.5281/zenodo.22233258). The
concept DOI for all versions is
[doi:10.5281/zenodo.22233257](https://doi.org/10.5281/zenodo.22233257).

## License

Code is released under the MIT License. The paper, preregistrations,
documentation, and original research data are released under CC BY 4.0.
Third-party dependencies and cited material retain their own licenses. See
`LICENSE` and `LICENSE-PAPER-DATA.md`.
