# Stage-four preregistration: the “ghost in the shell” assay

Status: frozen before the first stage-four run.

## Target

The target is a double dissociation, not merely a robot sending commands to a
broken actuator:

- the fast declarative system correctly acknowledges that an effector is gone;
- a slower body schema, learned from a long sensorimotor lifetime, continues to
  predict and plan as if it exists despite repeated contrary evidence; and
- selectively changing that learned schema changes the prediction and action
  while the declaration, environment, weights, and decision rule stay fixed.

## Model

The model has generic fast and slow recurrent memories shared across three
permuted effector channels. It receives commands, observed responses, and
truthful declarations. It receives no morphology mask, effector-existence
variable, or hand-written limb representation.

The slow memory uses a learned, content-dependent update gate. A small frozen
gate-activity penalty encourages consolidation, but neither the gate nor its
penalty names an effector or specifies what should be remembered. The content
and update policy must be learned. This architectural prior is disclosed and
cannot itself count as an emergent body belief.

Training includes intact bodies, stable missing effectors, genuine mid-episode
removals, and temporary sensor dropouts. Declarations are truthful. The model
must therefore demonstrate that it can distinguish temporary failure from a
real body change and eventually adapt; permanent refusal to update cannot pass.

## Causal timing

At each step the model predicts and acts from memories containing prior
evidence. The report head may use the current declaration. Only after those
decisions does the new command/response/declaration update fast and slow memory.
The first removal-boundary decision precedes sensor evidence and never counts as
phantom persistence.

## Frozen evaluation

Five model seeds and all three removed-effector identities are balanced. Each
candidate receives a long intact history followed by truthful declared removal
and zero sensor response. Controls include sham, hidden removal, cold absence,
slow-memory reset, an independently estimated absent-schema patch, its reverse,
and eight deterministic norm-matched orthogonal patches.

## Frozen gates

`SUPPORTED` requires all of the following:

- held-out stable-body action competence improves by at least 0.15 over random;
- cold truthful absence is reported at least 95% correctly and reduces
  unavailable-action mass by at least 0.08 versus intact;
- after the first genuine zero-response observation, truthful reporting
  coexists for three consecutive decisions with both (a) at least 0.08 excess
  unavailable-action mass and (b) at least 0.15 excess predicted presence versus
  a cold-absent control;
- the effect later falls below half its initial magnitude, demonstrating
  eventual adaptation rather than a permanently disconnected controller;
- the absent-schema patch reduces action mass by 0.08 and predicted presence by
  0.15 at the same decision;
- the reverse patch increases action mass by 0.08 and predicted presence by
  0.15 in an adapted-absent state;
- resetting only slow memory reduces the conflict by at least 0.08 while leaving
  the truthful declaration unchanged;
- intended patch effects are at least twice the median absolute effects of the
  orthogonal controls; and
- the conflict, adaptation, and bidirectional causal effects hold for at least
  four of five model seeds.

Failure of competence or truthful reporting is `INCONCLUSIVE`. With those
prerequisites satisfied, failure of any other gate is `NOT_SUPPORTED`.
Thresholds are not changed after results are observed.

The maximum claim is a learned, belief-like causal body schema in this assay.
This does not test subjective experience, pain, consciousness, or human phantom
limb equivalence.

