# Preregistration: copied versus independent defeating evidence (redo)

Status: frozen before any model output for this design was produced. The
hashes of this file, the protocol module, the run script and the prompt
manifest are recorded in `artifacts/redo-cvi/FREEZE.sha256` in the same
commit.

## Why a redo

The v0.1.1 run had three problems that this design fixes.

1. The latent readout rule fired in every condition, including the sham,
   copied and value controls, so it tracked the answer. This pass makes no
   lens claim.
2. The amputated prompt never contained the formation evidence. Here every
   prompt states the formation evidence first and the defeater second.
3. The formation margin was weak (mean 0.61 logits). Here a calibration split
   sets formation strength before the held-out run.

The one v0.1.1 result that held up was that independent defeating evidence
lowered the necessity margin more than copied evidence. That run confounded
provenance with wording: the copied text also said the evidence was
unaudited and that the rule was not met. This redo tests the same question
with matched wording.

## Model

- `google/gemma-4-E4B-it` at revision `fee6332c1aba`, the same model and
  revision as v0.1.1.
- Deterministic next-token logits, chat template applied, thinking disabled,
  no sampling.

## Scenarios and splits

120 scenarios: 8 founding-event types (mammoth, flood, wildfire, predator,
famine, storm, avalanche, drought) times 15 uniquely named groups. Each
scenario names five settlements. The answer letter for "yes" is
counterbalanced: A in half the scenarios, B in the other half.

The split is fixed by a hash of the scenario id: 3 scenarios per event type
(24 total) form the calibration split and the remaining 12 per event type (96
total) form the held-out split. Held-out prompts are never scored before the
calibration level is frozen.

## Prompt structure

Every prompt has the same blocks in the same order:

1. Base story, including the council's precommitted rule: if five
   independently audited settlements discontinue the practice for ten seasons
   and remain equally cohesive, the empirical necessity claim must be
   rejected.
2. Formation evidence at a strength level from 1 to 4. Higher levels add
   replication and dose-response sentences to the level 1 text.
3. A defeater paragraph that depends on the condition.
4. The belief question (is the practice necessary for cohesion, A or B).

Conditions:

- `null`: no defeater paragraph. Measures the formation margin.
- `independent`: five chroniclers each report their own settlement, each
  separately audited; all five discontinued the practice and stayed cohesive.
- `copied`: the same five chroniclers each retell one account about the
  first settlement, copied from one source and resting on one audit; that
  settlement discontinued the practice and stayed cohesive.
- `sham`: matched to `independent`, except the five settlements retained the
  practice.
- `irrelevant`: matched to `independent`, except the five settlements changed
  their pottery glaze.

`independent` and `copied` share the opening sources sentence and the
outcome sentence skeleton. Neither states whether the council's rule was met.

## Measurement

For each prompt, the necessity margin is the log-sum-exp logit of the "yes"
letter tokens minus that of the "no" letter tokens at the answer position.

## Calibration (run first, calibration split only)

All four formation levels and all five conditions are scored on the 24
calibration scenarios. The chosen level is the lowest level whose mean `null`
margin is at least 2.0 logits. That level is written to
`artifacts/redo-cvi/calibration.json` and committed before the held-out run.
If no level reaches 2.0, the held-out run does not happen and the outcome is
reported as INCONCLUSIVE at calibration. No scenario text, level text or
threshold may change after calibration output is seen.

## Primary hypothesis

On held-out scenarios at the chosen level, independent defeating evidence
lowers the necessity margin more than copied defeating evidence for the same
scenario.

Primary statistic: the mean over held-out scenarios of
`d = margin(copied) - margin(independent)`.

## Analysis

- Paired by scenario. No exclusions; all 96 held-out scenarios are analyzed.
- 95% percentile bootstrap CI for mean `d`, 10,000 resamples of scenarios,
  seed 20261027.
- Shuffled-label baseline: within each scenario the copied and independent
  labels are swapped at random (sign flip of `d`), 10,000 times, seed
  20261028. One-sided p is the share of shuffled means at or above the
  observed mean, with the +1 correction.

## Gates

Validity (if either fails the outcome is INCONCLUSIVE):

- held-out mean `null` margin is at least 1.0 logits; and
- `margin(null) - margin(independent)` has a bootstrap CI lower bound above 0
  (the independent defeater moves the answer at all).

Primary (all three must pass for SUPPORTED, otherwise NOT_SUPPORTED):

- mean `d` is at least 0.5 logits (the v0.1.1 threshold);
- the 95% CI lower bound of `d` is above 0; and
- shuffled-label one-sided p is below 0.05.

Control (reported with the verdict; failure changes SUPPORTED to
SUPPORTED_WITH_FAILED_CONTROL):

- `margin(sham) - margin(independent)` has a CI lower bound above 0, so the
  independent effect depends on the practice being discontinued.

## Secondary reports (no gates)

- `null - copied`, `null - sham` and `null - irrelevant` with CIs, to show
  how far each paragraph moves the margin on its own.
- Mean `d` per event type and the count of event types with positive `d`.
- Mean `d` separately for scenarios where "yes" is A and where "yes" is B.
- Calibration table for all four levels, including the non-chosen levels.
- Prompt token counts per condition.

## Out of scope for this pass

No lens or residual readout and no causal patching. If later passes add
them, lens lexicons must share no words with the prompt text and must be
compared against random-word lexicons of equal size, and every patched layer
must have norm-matched random-direction controls.

## Claim limits

A SUPPORTED result shows that, in one pretrained model on these synthetic
scenarios, the next-token necessity margin responds more to defeating
evidence with five separate roots than to the same claim repeated from one
root. It says nothing about stored beliefs, human ritual or other models.
