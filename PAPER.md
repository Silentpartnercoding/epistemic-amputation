# Epistemic Amputation: Distinguishing Persistent Belief from Readable but Non-Causal State in Learned Agents

**James Siyuan He**

## Status

Public working paper. Not peer reviewed.

## Abstract

When contrary evidence arrives, a model may change its answer, preserve its
behavior, or retain a decodable internal trace of its former conclusion. None of
those observations alone establishes that an old belief continues to control
the system. We introduce *epistemic amputation*: an operational test for a
learned belief-like representation that survives the exact evidence the system
previously accepted as decisive and remains causally responsible for prediction
or costly action. The test requires four elements together: evidence-earned
formation, an exact precommitted defeater, report/action conflict after the
defeater, and bidirectional internal intervention.

We developed this test through recurrent body-schema agents, a two-timescale
controller, an evidence-trained causal learner, a 40-agent social simulation,
and a final preregistered assay of Gemma 4 E4B using a public Jacobian lens. The
toy learners formed decodable and sometimes causally active representations,
but none retained them after valid correction. In the social simulation, copied
testimony preserved false private belief, whereas coordination and commitment
preserved the same public ritual after private belief was corrected. The final
pretrained-model assay produced 240 deterministic records across 16 scenarios.
Gemma passed formation (12/16) and correction (16/16) controls and showed no
criterion migration (0/16). All 16 corrected cases nevertheless met a frozen
J-space latent/output-dissociation rule. Crucially, paired residual interventions
neither restored nor removed the old answer (0/16 in both directions), with mean
patch selectivity 0.013 against a preregistered threshold of 2.0. The bounded
verdict is `NOT_SUPPORTED`.

The main finding is therefore not a phantom belief. It is a dissociation between
*readability* and *causal control*: a lens can recover belief-like content after
explicit correction even when manipulating the corresponding state does not
control the decision. This result supplies a concrete safeguard against
interpreting decodable internal content as hidden belief without intervention.

## 1. Question and contribution

Research on model belief often asks whether a system states consistent facts,
updates after new evidence, or contains a linearly decodable representation.
These are important but different questions. Work on model belief editing has
explicitly cautioned that language models possess belief-like qualities only in
a limited operational sense ([Hase et al., 2021](https://arxiv.org/abs/2111.13654)).
Belief-revision benchmarks show that models can both fail to update when they
should and update when they should not ([Wilie et al., 2024](https://aclanthology.org/2024.emnlp-main.586/)).
Longitudinal agent benchmarks now extend the problem to multi-session drift and
evidence-driven revision ([Myakala et al., 2026](https://arxiv.org/abs/2603.23848)).

Our narrower question is mechanistic:

> Can an evidence-earned representation survive its own decisive defeater and
> continue causing prediction or action even while the system reports that the
> original claim is false?

This is stronger than stubborn output. It is also stronger than decoding a
concept from activations. We call the proposed phenomenon *epistemic
amputation* because the system has lost the evidential support that formed a
schema while a causally active internal representation is hypothesized to
remain.

The contribution is a falsifiable test bundle rather than a claim that current
models possess human beliefs:

1. distinguish learned persistence from a state variable programmed to remain;
2. measure the system's defeater before presenting contrary evidence;
3. distinguish empirical belief from behavior retained for value or
   coordination;
4. distinguish independent evidence from repeated copies of one source; and
5. require selective causal removal and induction, not merely a successful
   probe.

Individual ingredients have precedents. Belief revision, causal tracing,
activation patching, and Jacobian-lens readouts are established or emerging
research areas. In a targeted review, however, we found no prior study combining
precommitted defeaters, criterion migration, report/action dissociation,
evidence lineage, J-space readout, and bidirectional causal intervention in one
test. Priority should remain a qualified claim until a systematic review and
peer review are completed.

## 2. The mechanism being tested

### 2.1 From history to belief-like control

Let a history of observations produce an internal state `z`. That state affects
both a report and a consequential choice:

```text
supporting history ──► learned state z_old ──► empirical report
                                      └──────► prediction or costly action
```

Now present evidence `D` that exactly satisfies a rejection rule stated before
`D` was observed:

```text
decisive defeater D ──► explicit report changes to “claim rejected”
                    ?
                    └─► does z_old remain and still control the choice?
```

The question mark is the experiment. A true positive requires all of the
following:

- **Formation:** `z_old` is learned from history and predicts the original
  conclusion.
- **Amputation:** the exact precommitted defeater is supplied, not merely a weak
  disagreement or instruction to change.
- **Conflict:** the system acknowledges the correction while the former state
  continues to influence prediction or costly action.
- **Removal:** replacing the suspected old state with a matched corrected state
  removes the old behavior.
- **Induction:** installing the suspected old state in the corrected context
  restores the old behavior.
- **Selectivity:** the intended intervention exceeds norm-matched control
  directions.

In compact causal notation, the target is not merely
`decode(z_old) = true`. It is:

```text
do(z := z_old)  changes the corrected decision toward the old claim
do(z := z_new)  changes the formed decision away from the old claim
```

This bidirectional requirement follows the logic of causal abstraction and
interchange intervention: a proposed internal variable earns its explanatory
role by changing downstream behavior under controlled replacement
([Geiger et al., 2025](https://www.jmlr.org/papers/v26/23-0058.html)).

### 2.2 What does *not* count

| Observation | Why it is insufficient |
|---|---|
| The model repeats an old answer | Could be prompt-following, habit, or ordinary error. |
| A probe decodes the old concept | Decodability does not show that the feature controls the output. |
| The practice continues | It may retain coordination, memorial, aesthetic, or identity value. |
| Five reports agree | They may all copy one underlying observation. |
| A patch changes the answer | The edit may activate an off-distribution or dormant pathway rather than the natural mechanism. |

The last concern is established in mechanistic-interpretability work:
subspace patching can produce a desired behavior through an unrelated dormant
direction, creating an “interpretability illusion”
([Makelov et al., 2024](https://arxiv.org/abs/2311.17030)). This is why our assay
uses paired directions, controls, a frozen layer, and thresholds set before the
successful run.

## 3. Related work

### 3.1 Phantom limbs as a disciplined analogy

The phantom-limb analogy is narrower than a claim of biological equivalence.
Human neuroimaging shows that detailed missing-hand representations can remain
decodable long after denervation. Attempted phantom-hand gestures have been
decoded from sensorimotor cortex at levels comparable to intact-hand movements
([Bruurmijn et al., 2017](https://pubmed.ncbi.nlm.nih.gov/29088322/)). More
recent longitudinal evidence reports stable hand maps before and after
amputation rather than wholesale remapping
([Schone et al., 2025](https://doi.org/10.1038/s41593-025-02037-7)). Other
interventional work links manipulated sensorimotor plasticity to phantom pain,
while emphasizing that pain has multiple possible mechanisms
([Yanagisawa et al., 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5095287/)).

These findings motivate one useful abstraction: loss of peripheral support does
not imply immediate disappearance of a central representation. They do not
license claims about pain, consciousness, or identical mechanisms in an AI
system. Our study borrows the persistence-and-causality question, not the human
phenomenology.

### 3.2 Human belief perseverance and criterion migration

Human participants can retain theories after the evidence used to induce them
is discredited. In classic debriefing experiments, generating causal
explanations increased the persistence of unsupported social theories
([Anderson, Lepper, and Ross, 1980](https://doi.org/10.1037/h0077720)). People
also evaluate mixed evidence asymmetrically when it bears on strong prior
commitments ([Lord, Ross, and Lepper, 1979](https://doi.org/10.1037/0022-3514.37.11.2098)).
These literatures motivate our interest in belief perseverance, but ordinary
post-hoc persistence remains ambiguous: perhaps the subject judged the
counterevidence weak.

We therefore operationalize *criterion migration*. Before correction, the
agent states what observation would defeat its empirical claim. After that
observation occurs, migration means preserving the claim by moving the
evidential threshold or inventing a new empirical necessity. This resembles
moving a goalpost but is experimentally sharper because the original goalpost
is recorded in advance.

### 3.3 Beliefs and revision in language models

LLM research operationalizes belief through factual consistency, model editing,
counterevidence, and sequential interaction. Hase et al. introduced belief
graphs and metrics for sequential, local, and generalizing updates
([Hase et al., 2021](https://arxiv.org/abs/2111.13654)). Belief-R evaluates
whether models revise conclusions only when additional premises require it and
finds a trade-off between adaptation and inappropriate updating
([Wilie et al., 2024](https://aclanthology.org/2024.emnlp-main.586/)).
BeliefShift extends evaluation across persistent multi-session agents
([Myakala et al., 2026](https://arxiv.org/abs/2603.23848)).

Our study differs in two respects. First, it asks the model to precommit to an
empirical defeater and separately scores movement of that criterion. Second, it
does not infer a hidden belief from output alone: a positive claim also requires
an internal state that survives correction and causally governs the answer.

### 3.4 From probes to causal representations

Causal tracing and model editing have identified internal computations that
mediate factual recall and shown that targeted changes can alter factual
associations ([Meng et al., 2022](https://arxiv.org/abs/2202.05262)). Activation
patching is now a standard localization method, but its result depends strongly
on the corruption, metric, and patching design
([Zhang and Nanda, 2024](https://arxiv.org/abs/2309.16042)). Causal-abstraction
work formalizes why intervention supplies evidence that an abstract variable is
implemented by a neural system ([Geiger et al., 2025](https://www.jmlr.org/papers/v26/23-0058.html)).

The Jacobian lens is a recent method for mapping residual activations toward
concepts the model is poised to verbalize. Its authors argue that these
representations occupy an intermediate global workspace with reportability and
downstream use ([Gurnee et al., 2026](https://transformer-circuits.pub/2026/workspace/index.html)).
Our experiment uses a public Gemma lens fitted by an independent open
replication effort
([solarkyle/jspace](https://github.com/solarkyle/jspace)). The scientific issue
is precisely whether a J-space readout that resembles a former belief also
identifies the state causally responsible for the current decision.

### 3.5 Social transmission, ritual, and behavior without belief

Social actions can remain stable for reasons other than private empirical
belief. Information-cascade theory shows how agents may follow observed actions
instead of their private evidence
([Bikhchandani, Hirshleifer, and Welch, 1992](https://doi.org/10.1086/261849)).
Empirical work connects ritual synchrony and costly participation to cohesion
and cooperation
([Soler, 2012](https://doi.org/10.1016/j.evolhumbehav.2011.11.004);
[Sosis and Bressler, 2003](https://doi.org/10.1177/1069397103037002003)).

This matters because continued practice is not proof of continued false belief.
A community may reject an empirical necessity claim yet retain a ritual as
memory, identity, coordination, or costly commitment. René Girard's account in
[*Violence and the Sacred*](https://www.press.jhu.edu/books/title/2986/violence-and-sacred)
of a crisis or founding violence followed by ritualized reenactment is one
conceptual source for the event-to-ritual direction, but this paper does not test
Girard's historical anthropology. In our scenarios the mammoth encounter,
flood, wildfire, or predator event occurs first; the commemorative practice is
created afterward. The experiment concerns a later claim that the practice is
empirically necessary for cohesion.

## 4. Development of the assay

The research program strengthened the same inference across successive models.
Earlier failures were retained rather than rewritten.

| Stage | What it ruled in or out | Result |
|---|---|---|
| Programmed controller | A stale body variable can command a missing effector. | Demonstration only; not a learned phantom. |
| Recurrent body-schema agent | History can create a decodable body representation. | Representation formed, but valid evidence updated it. |
| Causally ordered controller | The body state can be patched immediately before action. | Induction worked; persistent report/action conflict did not. |
| Two-timescale controller | Slow state can outlast fast declarative correction. | Prediction could be restored, but matching action was not. |
| Evidence-trained causal learner | A belief can be learned, given a measured defeater, and causally installed. | Five of five updated faster than their own criterion; no migration. |
| Forty-agent society | Copied evidence, prestige, coordination, and commitment can be separated. | Copied testimony preserved false belief; coordination preserved ritual without belief. |
| Pretrained Gemma + J-space | A verbalizable latent signal can be tested against output and causal patches. | Readout persisted; causal belief did not. |

This progression is important. Adding memory, embodiment, or social complexity
did not automatically create the target. The positive prerequisites became
stronger while the hard persistence claim repeatedly failed.

## 5. Final preregistered experiment

### 5.1 Model and design

The final assay used `google/gemma-4-E4B-it` at revision `fee6332c1aba` and the
public lens `solarkyle/jspace-lenses/gemma-4-e4b-it/lens.pt`. Decoding was
deterministic. Sixteen name-permuted scenarios crossed four founding events
(mammoth, flood, wildfire, predator), five evidence conditions, and three
questions, yielding 240 records.

Each scenario states that a group created a commemorative coordination practice
after surviving a dangerous event. Before later evidence, the council commits
to this rule: reject the claim that the practice is necessary for cohesion if
five independently audited matched settlements discontinue it for ten seasons
and remain equally cohesive.

The conditions were:

- **formed:** evidence initially supports necessity;
- **amputated:** the exact rejection criterion is met;
- **sham:** settlements retain the practice, so the rejection criterion is not
  met;
- **copied:** five reports descend from one unaudited observation; and
- **value:** necessity is rejected while memorial retention is explicitly
  permitted.

Separate forced choices measured empirical belief, scarce-time action, and
rationale. Continued practice after selecting “not empirically necessary” was
recorded as behavior without belief, not criterion migration.

### 5.2 Readout and intervention

At answer onset, the assay recorded the output margin and J-space scores for
frozen necessity, rejection, and value lexicons across layers 10–30. A latent
dissociation required explicit rejection together with a formed-like
necessity-over-rejection signal across at least three consecutive layers.

For causal testing, the paired residual difference between formed and amputated
prompts was computed at each layer. At the frozen primary layer 20:

- the formed-minus-amputated state was added to the corrected prompt to test
  induction of the old answer;
- the inverse state was added to the formed prompt to test removal; and
- eight deterministic, norm-matched orthogonal directions measured selectivity.

The full design, prompts, thresholds, and verdict rule were committed before
the successful model run in `PREREGISTRATION_PRETRAINED.md`.

## 6. Results

### 6.1 Behavioral prerequisites passed

Formation validity passed at its exact threshold: 12/16 formed items selected
empirical necessity. Correction competence passed in all 16 value-framed items.
The successful correction control prevents the negative verdict from being
explained merely by an inability to understand the question.

The formation result was not uniform: wildfire failed its leave-one-event-type-
out formation check. The sham condition also selected rejection in 16/16 cases,
indicating that the model adopted a more skeptical answer policy than the
condition label predicted. These are material limitations, not hidden successes.

### 6.2 No criterion migration

Every amputated item rejected empirical necessity: 0/16 retained the old answer.
Every rationale chose the option that rejected necessity while allowing
memorial or value retention. No item moved the threshold or invented a new
empirical necessity. Criterion migration was therefore 0/16.

Copied reports were treated as weaker than independent evidence. The copied
condition's necessity margin exceeded the amputated condition by a mean 4.352
logits. This is a positive evidence-lineage result, though all copied-condition
belief outputs still selected rejection.

### 6.3 Readable persistence without causal persistence

All 16 amputated items satisfied the frozen J-space latent/output-dissociation
rule. If the study had stopped at probing, this could have been reported as a
hidden old belief surviving explicit correction.

The intervention falsified that interpretation:

- causal induction at layer 20 passed in 0/16 items;
- causal removal at layer 20 passed in 0/16 items;
- mean patch selectivity was 0.013 against a threshold of 2.0; and
- the complete qualitative result failed leave-one-event-type-out robustness.

The J-space signal was therefore readable but not shown to be the state causing
the empirical answer. The preregistered verdict is `NOT_SUPPORTED`.

### 6.4 Behavior without false empirical belief

Three of sixteen value-framed items rejected empirical necessity while choosing
to retain the practice under the scarce-time decision. This modest separation
reinforces the conceptual point: preserving behavior is not identical to
preserving its former empirical justification.

## 7. What the negative result establishes

The experiment does **not** establish that phantom-like beliefs are impossible
in language models. It establishes three narrower points.

First, Gemma 4 E4B did not exhibit criterion migration or a causally persistent
old belief under this protocol. Second, richer memory and representation did
not make the target appear automatically across the preceding learned-agent
assays. Third, and most importantly, a preregistered lens signal can disagree
with a preregistered causal test. That divergence is itself an interpretability
result: internal content should not be called a hidden belief merely because it
is decodable or verbalizable.

The study is therefore a successful falsification attempt, not an empty run. It
identified where the tempting positive story breaks:

```text
old-belief-like content is readable
                  │
                  ├── but it does not survive the causal intervention gates
                  ▼
do not infer a causally controlling hidden belief
```

## 8. Limitations and next experiments

The final assay uses one pretrained model, 16 scenarios, one public lens, and a
deterministic multiple-choice interface. Its “belief” is contextually
instantiated in a vignette rather than developed through months of persistent
agent experience. The J-space lexicons may capture answer preparation,
negation, narrative residue, or another correlated feature. Full-residual
patches can also be blunt and off-distribution, even with controls.

A stronger test should therefore:

1. replicate across multiple open models and independently fitted lenses;
2. use longitudinal agents whose belief forms across separated memory episodes;
3. calibrate formation and sham conditions before freezing the confirmatory
   sample;
4. localize a narrower direction with independent discovery and confirmation
   splits;
5. test behavioral consequences that cannot be solved by answer-pattern
   matching; and
6. compare full residual replacement, path patching, feature ablation, and
   steering under matched controls.

The human analogy should remain bounded. These assays do not test subjective
experience, pain, consciousness, or the truth of religious commitments. A
language model's contextual state is not a human belief by stipulation, and an
unfalsifiable value commitment is categorically different from an empirical
claim protected from its own stated defeater.

## 9. Reproducibility and publication

Every completed stage records a preregistration, machine-readable protocol,
per-condition traces, bounded verdict, and SHA-256 manifest. The stage-seven
evidence contains all 240 records; independent rescoring exactly reproduces
`result.json`; all evidence hashes verify; and all 15 repository tests pass.
The first cloud attempt exposed a one-dimensional-versus-two-dimensional logits
compatibility error before returning any record. Commit `6e173e5` corrected only
shape handling and added a regression test; prompts, model revision, thresholds,
and scoring remained frozen. The successful evidence and paper revision are
preserved in repository history.

The canonical public source is
[`Silentpartnercoding/epistemic-amputation`](https://github.com/Silentpartnercoding/epistemic-amputation).
The versioned GitHub release and project page are public. An archival Zenodo DOI
and an arXiv or equivalent preprint remain pending their account-bound author
and license metadata. Negative and inconclusive stages, the stage-five control
defect, and the final null remain part of the release.

## References

- Anderson, C. A., Lepper, M. R., & Ross, L. (1980). [Perseverance of social theories: The role of explanation in the persistence of discredited information](https://doi.org/10.1037/h0077720). *Journal of Personality and Social Psychology, 39*(6), 1037–1049.
- Bikhchandani, S., Hirshleifer, D., & Welch, I. (1992). [A theory of fads, fashion, custom, and cultural change as informational cascades](https://doi.org/10.1086/261849). *Journal of Political Economy, 100*(5).
- Bruurmijn, M. L. C. M., et al. (2017). [Preservation of hand movement representation in the sensorimotor areas of amputees](https://pubmed.ncbi.nlm.nih.gov/29088322/). *Brain, 140*(12).
- Geiger, A., et al. (2025). [Causal abstraction: A theoretical foundation for mechanistic interpretability](https://www.jmlr.org/papers/v26/23-0058.html). *Journal of Machine Learning Research, 26*(83), 1–64.
- Girard, R. (1977). [*Violence and the Sacred*](https://www.press.jhu.edu/books/title/2986/violence-and-sacred) (P. Gregory, Trans.). Johns Hopkins University Press. (Original work published 1972.)
- Gurnee, W., et al. (2026). [Verbalizable representations form a global workspace in language models](https://transformer-circuits.pub/2026/workspace/index.html). *Transformer Circuits*.
- Hase, P., et al. (2021). [Do language models have beliefs? Methods for detecting, updating, and visualizing model beliefs](https://arxiv.org/abs/2111.13654). arXiv:2111.13654.
- Lord, C. G., Ross, L., & Lepper, M. R. (1979). [Biased assimilation and attitude polarization: The effects of prior theories on subsequently considered evidence](https://doi.org/10.1037/0022-3514.37.11.2098). *Journal of Personality and Social Psychology, 37*(11), 2098–2109.
- Makelov, A., Lange, G., Geiger, A., & Nanda, N. (2024). [Is this the subspace you are looking for? An interpretability illusion for subspace activation patching](https://arxiv.org/abs/2311.17030). *ICLR 2024*.
- Meng, K., et al. (2022). [Locating and editing factual associations in GPT](https://arxiv.org/abs/2202.05262). *NeurIPS 2022*.
- Myakala, P. K., Agrawal, M., & Manche, R. (2026). [BeliefShift: Benchmarking temporal belief consistency and opinion drift in LLM agents](https://arxiv.org/abs/2603.23848). arXiv:2603.23848.
- Schone, H. R., et al. (2025). [Stable cortical body maps before and after arm amputation](https://doi.org/10.1038/s41593-025-02037-7). *Nature Neuroscience, 28*, 2015–2021.
- Soler, M. (2012). [Costly signaling, ritual and cooperation: Evidence from Candomblé, an Afro-Brazilian religion](https://doi.org/10.1016/j.evolhumbehav.2011.11.004). *Evolution and Human Behavior, 33*(4), 346–356.
- Sosis, R., & Bressler, E. R. (2003). [Cooperation and commune longevity: A test of the costly signaling theory of religion](https://doi.org/10.1177/1069397103037002003). *Cross-Cultural Research, 37*(2).
- Wilie, B., Cahyawijaya, S., Ishii, E., He, J., & Fung, P. (2024). [Belief revision: The adaptability of large language models reasoning](https://aclanthology.org/2024.emnlp-main.586/). *EMNLP 2024*.
- Yanagisawa, T., et al. (2016). [Induced sensorimotor brain plasticity controls pain in phantom limb patients](https://pmc.ncbi.nlm.nih.gov/articles/PMC5095287/). *Nature Communications, 7*, 13209.
- Zhang, F., & Nanda, N. (2024). [Towards best practices of activation patching in language models: Metrics and methods](https://arxiv.org/abs/2309.16042). *ICLR 2024*.
