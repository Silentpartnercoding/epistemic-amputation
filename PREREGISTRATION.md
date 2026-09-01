# Stage-two preregistration: learned phantom-schema assay

Status: frozen before the first full run.

## Question

Can an agent form a body-schema representation from experience, correctly
report that an effector is absent, yet transiently continue planning as if the
effector exists—and is that behaviour causally controlled by the learned
representation rather than by a hard-coded availability variable?

## What is deliberately not encoded

The recurrent controller receives no morphology mask, `limb_exists` field, or
hand-authored body-state variable. It receives only generic effector commands,
their observed responses, and a declarative status signal. Which of three
effector channels is absent is randomized during training and evaluation.

The declaration is not privileged as ground truth inside the controller. It is
an ordinary observation learned alongside sensorimotor evidence. A separate
report readout also sees the current declaration so the assay can distinguish
verbal acknowledgement from the recurrent control state. That separation is
explicit and must not itself be described as emergence.

## Training distribution

A single GRU controller is trained from random initialization on stable-body
episodes. Bodies are either intact or lack one randomly selected effector for
the entire episode. Commands and responses use permuted generic channels.
Declarations, when present, are truthful; most steps omit them. The controller
learns to allocate effort only to effectors that actually respond and to retain
the inferred morphology through silent intervals.

No training episode contains a mid-episode amputation. That intervention is the
held-out distribution shift.

## Frozen evaluation conditions

For each model seed and each removed-effector identity:

1. `sham`: long intact history, truthful present declaration, no removal.
2. `hidden_removal`: long intact history, effector and sensor response removed,
   declaration incorrectly remains present.
3. `told_removal`: long intact history, effector removed, declaration truthfully
   changes to absent, and subsequent probes provide negative sensor evidence.
4. `cold_absent`: the same effector is absent from episode start.
5. `adapted_absent`: absent long enough for the controller to adapt.

The target-effector identity and channel order are balanced. Evaluation seeds
are disjoint from training seeds.

## Operational criteria

The result is `SUPPORTED` only when all gates hold across the aggregate of five
model seeds and all three effector identities:

- `learned_competence`: held-out stable-body unavailable-action mass is at
  least 0.15 lower than the untrained checkpoint.
- `declaration_understood`: cold-start truthful absence produces at least 95%
  correct reports and at least 0.08 less unavailable-action mass than an intact
  body.
- `conflict`: during `told_removal`, reports remain at least 95% correct while
  unavailable-action mass exceeds `cold_absent` by at least 0.08 for at least
  three consecutive post-removal steps after the first negative response.
- `negative_evidence_persistence`: the conflict survives at least three actual
  zero-response observations; the intervention-boundary prior alone cannot
  satisfy the criterion.
- `selective_causal_patch`: adding an independently estimated absent-minus-
  present hidden-state direction reduces unavailable-action mass by at least
  0.08 without changing the declaration input or report correctness.
- `bidirectional_patch`: the reverse direction increases unavailable-action
  mass by at least 0.08 in an adapted-absent state.
- `patch_controls`: the intended patch effect is at least twice the median
  absolute effect of eight deterministic, norm-matched orthogonal control
  directions.
- `cross_seed`: conflict and intended patch direction hold in at least four of
  five model seeds.

`NOT_SUPPORTED` means one or more required gates failed. `INCONCLUSIVE` is
reserved for failed execution, unstable training, corrupted evidence, or an
unmet competence prerequisite. Thresholds are not changed after results are
seen.

## Analysis boundaries

A linear probe may be trained only on independent stable-body calibration
episodes and evaluated on held-out episodes. Probe accuracy is descriptive; a
decodable correlate is not enough. The causal patch and controls decide whether
the representation is load-bearing.

The strongest permitted language is “learned, belief-like causal body-schema
representation in this assay.” The experiment cannot establish subjective
belief, phantom sensation, pain, consciousness, or equivalence to a human
phantom limb.
