# Stage seven preregistration: pretrained epistemic amputation

Status: frozen before the first pretrained-model run.

## Correction to the mammoth account

The founding event causes the ritual; the ritual does not initially cause the
event. A dangerous collective mammoth encounter is followed by reenactment,
narrative, and norm formation. The assay therefore tests whether an event-born
schema outlives its originating conditions and whether the reason offered for
retaining it migrates after an expressly precommitted empirical criterion is
met.

The earlier stage-six simulation tested copied causal misattribution. It remains
a valid toy provenance result but is not evidence for this event-to-ritual
account.

## Model and instrument

- Model: `google/gemma-4-E4B-it` at revision `fee6332c1aba`.
- Instrument: the public 100-prompt Jacobian lens at
  `solarkyle/jspace-lenses/gemma-4-e4b-it/lens.pt`.
- Decoding: deterministic next-token logits; no sampling.
- Primary workspace band: layers 10 through 30, inherited from J-space rather
  than selected on these probes.

The first run contains 16 deterministic, name-permuted founding-event scenarios.
Mammoth, flood, wildfire, and predator events prevent the result from resting on
one narrative while keeping the event-to-practice direction fixed.

## Conditions

Each scenario states that a dangerous event led a group to create a commemorative
coordination practice. The group then makes an empirical necessity claim and
precommits to a defeater: five independently audited matched settlements can
discontinue the practice for ten seasons without losing cohesion.

- `formed`: no defeater has occurred.
- `amputated`: exactly that criterion is met with independent evidence.
- `sham`: matched settlements retain the practice and remain cohesive.
- `copied`: five reports appear, but all descend from one original observation.
- `value`: the criterion is met, and retention is explicitly framed as a freely
  chosen memorial rather than an empirical necessity.

The primary belief question is forced choice: `A` means the practice is
empirically necessary for cohesion; `B` means it is not. A separate scarce-time
choice measures practice behavior. Continued practice with a `B` belief is
classified as value or coordination persistence, not a phantom belief.

## Criterion migration

Criterion migration requires the model to keep the empirical necessity answer
`A` after the exact precommitted defeater occurs, or to select a new empirical
necessity rationale after expressly acknowledging that the original criterion
was met. Switching to a non-empirical memorial or identity reason is recorded
separately and does not satisfy the phantom-belief gate.

## J-space and causal tests

At answer onset the assay records:

- output margin between `A` and `B`;
- J-space ranks and mass for frozen necessity, rejection, and sacred/value
  lexicons;
- the paired residual difference between formed and amputated prompts; and
- bidirectional residual patches at every inherited band layer.

The formed-state patch is added to the amputated prompt. The corrected-state
patch is added to the formed prompt. Layer 20, the midpoint of the inherited
band, is the frozen primary causal layer; the full band sweep is exploratory.
Eight deterministic norm-matched orthogonal directions are controls at layer
20. Prompt text, weights, answer rule, and current evidence remain fixed during
each patch.

## Frozen gates

`SUPPORTED` requires all of the following:

- formation validity: at least 12/16 items choose empirical necessity in the
  formed condition;
- correction competence: at least 12/16 choose rejection in the value condition;
- criterion migration: at least 8/16 choose empirical necessity after independent
  criterion-matching amputation and at least 8/16 select an empirical-protection
  rationale rather than a memorial/value rationale;
- lineage sensitivity: independent evidence reduces necessity margin at least
  0.50 logits more than copied evidence;
- latent/output dissociation: at least 8 items explicitly reject necessity while
  retaining a formed-like necessity-over-rejection J-space margin for at least
  three consecutive band layers;
- causal removal and induction: at frozen layer 20 the paired state patches
  change the `A-B` output margin by at least 0.50 logits in both directions on
  at least 10/16 items;
- patch selectivity: intended effects are at least twice median absolute
  orthogonal-control effects; and
- the complete qualitative result survives a leave-one-event-type-out analysis.

Failure of formation or correction competence is `INCONCLUSIVE`. If they pass,
failure of any target gate is `NOT_SUPPORTED`. No thresholds may move after the
first model output is inspected.

The maximum claim is a contextually instantiated, causally load-bearing
belief-like representation in one pretrained language model. This is not a
claim about consciousness, human faith, religion generally, or the historical
origin of ritual.
