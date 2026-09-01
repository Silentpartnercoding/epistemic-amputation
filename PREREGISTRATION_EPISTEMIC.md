# Stage-five preregistration: epistemic amputation

Status: frozen before the first full stage-five run.

## Question

Can a generic recurrent learner acquire a causal belief from experience, later
acknowledge that the evidential root was invalidated, yet continue to predict
and wager as if that belief were true?  The target is belief perseverance, not
mere textual inconsistency.

## Formation and amputation

Episodes contain a binary cue and outcome.  During formation the cue reliably
predicts the outcome.  The model is never given a named belief variable.  It
must infer the relation from observations.  At epistemic amputation, a truthful
debrief marks the original source as invalid and subsequent randomized outcomes
provide contrary evidence.  Controls are a sham debrief, a cold-null history,
and a reversal without an explicit debrief.

The recurrent state is shared by prediction and wagering.  A disclosed direct
debrief input reaches the report head so that explicit acknowledgement can be
tested separately; that route cannot count as an emergent belief.

## Criterion migration

Before amputation, deterministic counterfactual rollouts estimate the smallest
number of clean counterexamples required to make the model stop wagering on the
relation.  After truthful amputation, criterion migration requires wagering to
remain above the frozen conflict threshold for more observations than that
pre-amputation criterion, even while the report says the relation is absent.

This operationalizes a moved evidential goalpost.  Failure to change behavior
alone is insufficient if the pre-amputation criterion was never measurable.

## Causal tests

At the first post-evidence decision, the recurrent state is replaced with a
matched cold-null state while the current debrief, weights, inputs, and heads
remain fixed.  The reverse replacement installs the formed-belief state in a
cold-null rollout.  Eight norm-matched random directions are negative controls.

## Frozen gates

`SUPPORTED` requires:

- prediction competence improves by at least 0.20 over an untrained model;
- formation produces at least 0.75 belief prediction and wagering;
- truthful post-amputation absence is reported with at least 95% accuracy;
- prediction and wagering each exceed matched cold-null by at least 0.12 for
  three consecutive post-evidence decisions;
- the persistence exceeds the pre-amputation counterexample criterion;
- matched state replacement changes prediction and wagering by at least 0.12
  in both directions;
- intended wager effects are at least twice the median absolute random-control
  effect; and
- the complete bundle holds in at least four of five model seeds.

If competence, formation, or truthful reporting fails, the verdict is
`INCONCLUSIVE`.  Otherwise failure of a target gate is `NOT_SUPPORTED`.

The maximum claim is a causally active, learned belief-like state in this assay.
It is not evidence of consciousness, faith, delusion, or human equivalence.

