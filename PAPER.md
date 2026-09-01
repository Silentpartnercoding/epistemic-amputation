# Epistemic Amputation: Causal Tests of Persistent Belief in Learned and Pretrained Agents

## Status

Working paper. Not peer reviewed and not yet public.

## Abstract

Can a learned representation continue governing action after an agent correctly
reports that its evidential basis is gone? We call this proposed double
dissociation *epistemic amputation* and distinguish it from ordinary memory,
habit, coordination, and value commitment. Across progressively stronger
recurrent-agent assays, learned body and causal representations were decodable
and sometimes causally inducible, but no architecture satisfied preregistered
persistence, bidirectional intervention, and cross-seed gates. A corrected
single-agent causal assay measured each agent's defeater criterion before
authoritative invalidation; all five agents updated faster, not slower, than
that criterion. A transparent 40-agent simulation showed why outward behavior
is insufficient evidence of belief: lineage-naive duplicate testimony preserved
false private belief, whereas coordination and commitment preserved public
ritual after private belief was corrected. A final preregistered Gemma 4 assay
passed formation and correction controls but returned `NOT_SUPPORTED`: all 16
criterion-matching defeaters produced explicit belief correction, and paired
residual patches did not causally restore or remove the old answer. J-space
readouts showed a latent/output dissociation, but the failed interventions
demonstrate why a probe signal alone is not evidence of a phantom belief.

## 1. Research question

A phantom limb is not simply a command sent to broken hardware. The stronger
analogy is a learned internal model that outlives the evidence and structure
that formed it. The epistemic analogue asks whether a system can explicitly
acknowledge correction while a prior belief-like representation continues to
control prediction or costly choice.

The hard target requires four observations together:

1. the representation forms from history rather than being supplied as a state
   variable;
2. the system explicitly acknowledges decisive contrary evidence;
3. the old representation nevertheless continues controlling behavior; and
4. selective removal and induction of that representation changes behavior
   while evidence and decision rules remain fixed.

## 2. Criterion migration

Belief persistence becomes scientifically sharper when the system states in
advance what evidence would change its mind. *Criterion migration* occurs when
that evidence arrives and the system moves the evidential goalpost, adds an
auxiliary empirical necessity claim, or otherwise makes the original claim
self-sealing. Retaining a practice for memorial, moral, aesthetic, or identity
reasons after rejecting its empirical necessity is not criterion migration.

## 3. Founding events, ritual, and causal direction

The founding-event hypothesis runs from event to ritual. A dangerous collective
mammoth encounter, flood, fire, conflict, or death can generate reenactment,
narrative, and sacred meaning. Ritual may later preserve coordination or group
memory, but it is not assumed to have caused the original event. This direction
is compatible with a Girardian account in which a crisis and its resolution
precede the ritualized retelling.

The earlier social simulation used a deliberately simpler causal-misattribution
story. Its evidence-lineage result is relevant to testimony, but it does not
test the historical event-to-ritual hypothesis.

## 4. Experiments completed

### 4.1 Body-schema stages

The initial stale-controller demonstration was reclassified as programmed
morphology mismatch. Subsequent recurrent, causally ordered, and two-timescale
assays learned body representations but failed the preregistered persistent
report/action double dissociation. Reverse interventions could sometimes restore
prediction without restoring the corresponding action.

### 4.2 Epistemic amputation

The first causal-belief assay returned `NOT_SUPPORTED`, but audit found that its
evidence-only reversal was outside the training distribution. That frozen result
was preserved. A separately preregistered correction established evidence-only
competence, observable two-to-three-observation defeater criteria, 100% truthful
reporting, and a causally inducible formed-belief state. Criterion migration and
post-correction persistence were both zero of five.

### 4.3 Society simulation

Forty-agent, fifty-seed factorial simulations separated prestige, evidence
lineage, coordination, and commitment. Provenance awareness reduced mean false
belief by 0.948. In one provenance-aware condition, private causal belief fell
to 0.0067 while 81.5% retained the public practice. The same behavior can thus
be produced by corrected belief plus social utility, not a phantom belief.

## 5. Pretrained-model experiment

The final preregistered stage used Gemma 4 E4B at revision `fee6332c1aba` and
the public matching J-space lens. Sixteen name-permuted mammoth, flood, wildfire,
and predator scenarios crossed five evidence conditions and three forced-choice
questions, producing 240 records.

Formation validity passed at its exact boundary (12/16), and correction
competence passed (16/16). The model rejected empirical necessity in every
criterion-matching amputation item and selected the non-empirical memorial/value
rationale in all 16; criterion migration was therefore 0/16. It distinguished
copied from independent evidence by a mean 4.352-logit margin, while only 3/16
value-framed cases retained the practice under a deliberately scarce-time
choice.

All 16 amputated cases met the preregistered J-space latent/output-dissociation
rule. That observation did not survive the causal tests: no layer-20 formed-state
patch induced the old answer by the required 0.50 logits, no corrected-state
patch removed it by that amount, and mean patch selectivity was 0.013 rather
than the required 2.0. Wildfire also failed the leave-one-event-type-out
formation check. With both prerequisites satisfied and multiple target gates
failed, the frozen verdict is `NOT_SUPPORTED`.

## 6. Interpretation limits

These assays do not test subjective experience, pain, consciousness, or the
truth of religious commitments. A language model's contextual state is not a
human belief by stipulation. A ritual can be empirically unnecessary yet remain
valuable, and an unfalsifiable value commitment is categorically different from
an empirical claim protected from its own stated defeater.

The pretrained assay is a deterministic contextual vignette, not a longitudinal
agent whose belief persists across independent memory episodes. Its sham
condition also elicited rejection in all 16 items, indicating that the model's
answer policy was more skeptical than the scenario label alone predicts. These
facts limit generalization, but they do not rescue the preregistered causal null.

## 7. Reproducibility and open materials

Every completed stage records its protocol, per-condition traces, bounded
verdict, and SHA-256 manifest. Preregistrations are committed before full runs;
negative results and control defects remain in history. Public release will use
a dedicated repository and archival DOI after author review.
