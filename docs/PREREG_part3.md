# Neural experiment: scientific protocol and ES-fixed addendum

Excerpts from the study protocol, retaining original section numbers and scientific rules. Original document SHA-256: `78f367f3fc19c01aaf744c77f28c1ee092895f255c6c0fe07373f4a08d706f41`.

## 1. Questions and hypotheses

The core protocol evaluated whether ecological evolution outperformed PPO-DR during adaptation. The table records its predicted directions.

| # | Question | Hypothesis (predicted direction) | Comparison |
|---|---|---|---|
| **H1 (primary)** | At equal training compute, does open-ended, survival-based evolution adapt faster to a new environment than PPO with domain randomization? | **OE > PPO-DR** on the primary metric (Section 7.4) | OE vs PPO-DR |
| H2-OE (Q2) | Does variety in training matter for evolution? | OE > OE-fixed on the primary metric | S2 |
| H2-PPO (Q2) | Does variety in training matter for PPO? | PPO-DR > PPO-fixed on the primary metric | S3 |
| H2b (Q2) | Does variety matter more than the method? | No direction predicted by the claim; reported as a two-sided result | Section 8.3 |
| H3 | Does OE adapt faster than another fixed-objective method that also uses evolution-style search? | OE > ES-DR on the primary metric | S1 |
| H4 | Is OE's advantage specific to adapting, rather than starting better? | No direction predicted for zero-shot competence (S4); OE > PPO-DR for final competence after adaptation (S5) | S4, S5 |
| H5 | In-distribution check | No direction predicted (T1 is not a new environment) | S6 |
| H6 | Do OE's agents keep a population going in a new environment more often? | OE has a lower extinction rate than PPO-DR in the persistence test | S7 |

Lifetime-learning and neural diversity ablations (Q3/Q4) were planned but not run.

**What this can and cannot answer.** It answers H1–H6 for this 2D foraging world, these five methods and these budgets. It does not test intelligence in general, social complexity, language, or other worlds.

---

## 2. Methods compared (core)

OE, OE-fixed (the fixed-world ablation), PPO-fixed, PPO-DR and ES-DR, as defined in SPEC Section 5. Every method uses the same brain (30 → 32 → 32 → 2, 2,114 weights), the same world, the same training budget, the same tuning budget and the same evaluation.


---

## 3. Environment splits (locked)

The splits are defined in `configs/part3/train.yaml`, `validation.yaml` and `test_split.yaml` (SPEC Section 3.4). They are locked when this pre-registration is final; until then they are provisional and change only with a dated entry in original Section 15 (omitted from this excerpt).

- **Train distribution:** L 40–80, F 20–60, layout uniform / clusters (k 2–5, σ 3–8) / patches (k 2–5, radius 4–10), ρ 0.01–0.05, no drift, no seasons.
- **Canonical environment** (PPO-fixed, OE-fixed): L 60, F 40, uniform, ρ 0.03.
- **Validation:** 20 environments drawn from the train distribution with seeds 4900–4919, plus 4 corner environments; evaluation episodes use seeds 4930–4999. Used only for tuning and sanity checks.
- **Test:** T1 (train distribution, seeds 5900–5909), T2 (near shift: L 30–40, F 10–20, clusters k 2–3, σ 2–4, ρ 0.005–0.01; seeds 5910–5919), T3 (far shift: train ranges plus drift 0.05–0.3 and seasonal relocation every 100–300 steps; seeds 5920–5929). Evaluation episodes: environment j (0–29, in the order T1, T2, T3) uses episode seed 5930 + j, the same for every method.

**The lock.** Only `src/evo2/env/splits.py` reads the test file, only with `final=True`, and only `scripts/part3/evaluate.py --split test --final` asks for that. With `--final`, `evaluate.py` accepts only finished runs whose purpose is "final", evaluates the whole test split (no partial looks), and appends every look to `results/part3/test_access_log.jsonl`, which is reported. Tests check this (original SPEC Section 10, omitted from the specification excerpt). The pilot uses stand-in environments with the T2 and T3 ranges but new pilot seeds (`configs/part3/pilot_shift.yaml`), only to estimate variance for the power analysis, never to choose settings. The unit tests read the test file once to check that it parses and matches this section; no environment from it is run before the final evaluation.

---

## 4. Compute budget

- **Training:** the same number of agent-steps for every method: 2 × 10⁸ (design stage). An agent-step is one agent alive at the start of one step, counted the same way for every method (SPEC Section 6). No method overshoots by a full unit (one PPO update, one ES generation, one OE step); the actual count is reported for every run.
- **Adaptation:** the same number of agent-steps per test environment for every method: 2 × 10⁷ (design stage), on all 30 test environments (the primary metric needs T2 and T3; S6 needs T1).
- **Tuning:** reported separately; the same number of trials and the same per-trial budget for every method (Section 5).
- Every run records agent-steps, wall-clock time, hardware (GPU model, driver, CUDA and JAX versions) and the commit.

---

## 5. Tuning protocol (fixed before final runs)

- **Random search**, the same number of trials for every method: **8** (fixed before tuning), each trial trained on its own tuning seed (4000–4799, assigned in order: method m's trial t uses seed 4000 + 100 × m + t, with m = 0 OE, 1 OE-fixed, 2 PPO-fixed, 3 PPO-DR, 4 ES-DR).
- **Per-trial budget:** 25% of the final training budget (5 × 10⁷ agent-steps).
- **Score:** mean over the 24 validation environments of (competence − random-policy competence), zero-shot, 8 episodes per environment, with the validation episode seeds and the key path of `zero_shot` (so `evaluate.py --split validation` gives the same numbers). The alternative (adaptation AUC on validation environments) was not chosen. The best-scoring trial's settings are used for the final runs; ties go to the lower trial number; a trial whose parameters stop being finite scores lowest; if every trial of a method fails, stop and record a protocol amendment. Code: `scripts/part3/tune.py`; the chosen settings are written to `configs/part3/tuned.yaml` by `tune.py --write-tuned`, never by hand.
- **Search ranges** (declared now, before tuning):
  - PPO-fixed and PPO-DR (searched separately): learning rate log-uniform 1 × 10⁻⁴ – 1 × 10⁻³; NUM_STEPS {32, 64, 128}; UPDATE_EPOCHS {2, 4, 8}; GAMMA {0.95, 0.99, 0.995}; ENT_COEF {0, 0.001, 0.01}; reward normalization {on, off}. (No body search: The study used the default body for every method.)
  - ES-DR: SIGMA log-uniform 0.01–0.1; LR log-uniform 0.003–0.03; POP_SIZE {32, 64, 128}; EPISODES {1, 2, 4}.
  - OE and OE-fixed (searched separately): WEIGHT_SD log-uniform 0.005–0.05. (TRAIT_FRAC is not searched: bodies do not mutate.) CHANGE_EVERY and MIN_POP are design choices (Section 2), not tuned.
- Draws use the "tuning_draw" stream of the trial's seed. Tuning never sees the test split.

---

## 6. Seeds

| Use | Seeds | How chosen |
|---|---|---|
| Unit tests and smoke runs | 3900–3999 | plan range for pilots, set aside for tests |
| Pilot runs | 3000–3099 | plan range |
| Pilot evaluation episodes | 3100–3199 | plan range |
| Tuning runs | 4000–4799 | plan range |
| Statistics (bootstrap) | 4800 | from the unassigned gap 4800–4899, fixed now |
| Validation environments and episodes | 4900–4999 | plan range |
| Final runs | 5000 onward, in order | plan range; the same seed list for every method |
| Test environments and their episodes | 5900–5999 | plan range |

Final seeds: 10 per method, 5000–5009. The rough pilot-based precision calculation used three seeds per method; ten final runs provided limited precision for equivalence at the 0.05 margin.

*Rule as drafted before the pilot:* chosen at design stage by a power analysis on pilot variance (`scripts/part3/pilot.py`: the run-to-run spread of the adaptation AUC on the pilot's stand-in shifted environments): the smallest N (at least 10 per method, as the plan requires) for which the expected 95% CI half-width of the primary difference D (Section 8.1) is at most half the margin (0.025), so that every verdict in Section 8.1, including "no meaningful difference", is reachable. The adopted design retained ten runs and the original margin despite the resulting limited precision. The analysis and its inputs are reported.

Each seed is split into named streams (`src/evo2/seeds.py`); a test checks that no two streams share a key and that seed ranges do not overlap.

---

## 7. Measures

### 7.1 Competence
Mean food eaten per agent in an episodic episode (16 agents, 500 steps), averaged over 8 episodes per environment (design stage). PPO and ES field 16 copies of their policy. OE fields 16 agents sampled from its living population, a new sample per episode. Actions are deterministic (SPEC Section 3.1).

### 7.2 Normalization (per test environment)
normalized = (competence − random) ÷ (best − random), where
- **random** = the random policy's competence (default body, actions uniform within bounds) on the same episode seeds;
- **best** = the highest competence measured in that environment in the final evaluation, over every core method, final seed and checkpoint (zero-shot and adaptation).
If best ≤ random in an environment, it is left out of every normalized measure for every method, and this is reported.

### 7.3 Protocols
- **Zero-shot:** competence (7.1); persistence: 32 agents (sampled from OE's population, or copies of a PPO or ES policy) in ecological mode, births without mutation, no floor, 5,000 steps; extinct or not, and steps survived.
- **Adaptation:** each method keeps improving in the test environment for the adaptation budget (OE keeps evolving with the environment fixed, floor included, with floor births recorded; PPO keeps training with a fresh Adam state and its reward-normalization statistics carried over; ES keeps optimizing). Point 0 is the zero-shot competence (the same number as above); points 1–10 are measured after each tenth of the budget: 11 points. Evaluation episodes are the same for every method and run; the adaptation's own randomness comes from the run's seed.

### 7.4 Primary metric
For each run and each T2 and T3 environment: the area under the normalized competence curve over the 11 points, by the trapezoid rule, with the x-axis = fraction of the adaptation budget (0 to 1), so a flat curve at height h has area h (`evo2.metrics.curve_auc`). Aggregated with the IQM over all (run, environment) pairs of a method.

### 7.5 Secondary measures
- zero-shot normalized competence (point 0);
- final normalized competence (point 10);
- time to 90% (descriptive, not tested): the first point (fraction of the adaptation budget) at which competence ≥ 0.9 × the run's pre-shift competence (its mean zero-shot competence over the T1 environments); "not reached" if never. Reported per method as the share of (run, environment) pairs that reach it and the median time among those that do;
- extinction rate: the fraction of persistence tests that end extinct;
- genotypic diversity (OE only): mean pairwise distance between the weight vectors, and between the trait vectors, of 16 sampled agents;
- behavioral diversity (OE only): mean pairwise distance between 16 sampled agents' actions on a fixed set of 256 probe observations (drawn once from validation episodes with seed 4801, "probe" stream);
- performance profiles of the primary metric (the share of (run, environment) scores above each threshold), per method, descriptive;
- compute: agent-steps, wall-clock time, hardware, cost.

PPO and ES deploy one policy, so their diversity is 0 by design; no test is run on it (plan rule).

---

## 8. Analysis

### 8.1 Primary comparison (H1)
D = IQM(OE) − IQM(PPO-DR) of the primary metric, each IQM over all (run, environment) pairs of the method. 95% percentile CI by bootstrap over **whole runs**: each method's runs are resampled with replacement, each keeping all of its environments, because one trained run is scored on every environment and the run is the independent unit (`evo2.metrics.iqm_difference_ci`; 50,000 resamples; `numpy.random.default_rng(4800)`). Also reported: each method's IQM with its run-bootstrap CI, and the probability of improvement P(OE > PPO-DR) on per-run mean scores (rliable). rliable's own bootstrap is not used: it resamples runs separately within each environment, which treats one run's environments as independent, and it ignores the random state passed to it.

**Verdict (pre-set; the verdict is the result).** Checked in this order; the first that applies is the verdict (`evo2.metrics.h1_verdict`), with margin m = 0.05 (5% of the gap between the random policy and the best observed score):
1. **Support:** the CI lies entirely above 0. If it also lies below +m, the verdict is "support (below the margin)": OE is better, by less than the margin.
2. **Against** (PPO-DR adapts faster): the CI lies entirely below 0; "against (below the margin)" if it also lies above −m.
3. **No meaningful difference:** the CI lies inside [−m, +m] (and contains 0).
4. **Inconclusive:** anything else.
The margin stays 0.05 (design stage; with 10 runs per method verdict 3 is practically unreachable, Section 6).

### 8.2 Secondary comparisons (Holm)
One family, Holm-corrected at α = 0.05. Each test: two-sided Mann-Whitney U on per-run scores (each run's mean over the listed environments). The effect is the probability of improvement P(X > Y) on the same per-run scores (it equals U ÷ (n_X × n_Y), so it always agrees with the test). Reported next to each: the IQM difference with its run-bootstrap CI.

| | Comparison | Measure | Environments |
|---|---|---|---|
| S1 | OE vs ES-DR | primary metric | T2, T3 |
| S2 | OE vs OE-fixed | primary metric | T2, T3 |
| S3 | PPO-DR vs PPO-fixed | primary metric | T2, T3 |
| S4 | OE vs PPO-DR | zero-shot normalized competence | T2, T3 |
| S5 | OE vs PPO-DR | final normalized competence | T2, T3 |
| S6 | OE vs PPO-DR | primary metric | T1 |
| S7 | OE vs PPO-DR | extinction rate (per-run fraction) | T2, T3 |

A hypothesis with a predicted direction is supported only if the Holm-corrected test rejects and P(X > Y) is on the predicted side of 0.5. A rejection on the other side is reported as evidence against it. Hypotheses without a predicted direction (S4, S6) report the two-sided result.

### 8.3 Variety vs method (H2b)
The four methods form a 2 × 2 design (evolution or PPO, variable or fixed training). On the primary metric's IQMs:
- variety main effect V = ½ [(OE − OE-fixed) + (PPO-DR − PPO-fixed)];
- method main effect M = ½ [(OE − PPO-DR) + (OE-fixed − PPO-fixed)];
- interaction I = (OE − OE-fixed) − (PPO-DR − PPO-fixed).
Each with a 95% CI by the whole-run bootstrap of 8.1 (all four methods' runs resampled). The comparison is Δ = |V| − |M|: "variety matters more" if Δ's CI lies above 0, "method matters more" if it lies below 0, otherwise "not decided". Because of the absolute values, Δ's CI is unreliable when V or M is near 0; the CIs of V and M are always shown next to it. If the interaction's CI excludes 0, the verdict is reported with that caveat.

### 8.4 Not tested
Quantities fixed by design (for example PPO's and ES's diversity) and descriptive results (training curves from the snapshots, diversity over time, performance profiles, time to 90%) are reported without tests. **Training curves (defined 3 Nov 2026, before any final run):** each snapshot (every tenth of training) is scored like a tuning trial: mean over the 24 validation environments of (competence − random competence), zero-shot, 8 episodes, validation episode seeds; computed on the pods right after training. Diversity over time uses the OE snapshots.

### 8.5 Planned extensions
Lifetime-learning and diversity ablations were not run and contribute no tests to the reported comparison.

---

## 9. Exclusion rules

- A run is excluded only if it fails technically: it crashes and cannot resume, or its parameters become NaN or infinite. The failure is reported with the run.
- An excluded run is replaced by the next unused final seed, for that method only, and the replacement is reported.
- If more than 20% of a method's runs fail, stop and record a protocol amendment.
- OE runs are never excluded for low population or extinction; floor births and extinctions are reported.
- No run is excluded for its results.

---

## 10. Amendment triggers

Stop and record a protocol amendment before going on if:
- the pilot shows a measure with no room to move (for example, the random policy scores as well as trained ones, or every method eats nearly all the food);
- OE's population lives mostly on floor births at the end of the pilot budget *(fired in P1 for OE-fixed: 67–102 floor births per 1,000 steps against 11–21 own births; OE about half and half. The floor was retained on 3 Nov 2026 before final runs)*;
- repeat runs on the GPU are not bit-identical and the difference is more than 1% of a run's competence;
- any change would alter what an experiment means.

---

## 16. Addendum A: ES-fixed (written 4 Nov 2026, after the core results, before any ES-fixed run)

### 16.1 Why, and what it can and cannot answer

Written after the core results (`docs/RESULTS_part3.md`, 3 Nov) were known. The report answers Milano, Carvalho and Nolfi (2019), *Artificial Life* 24(4):277–295: agents evolved by score-based evolution (a (μ+1) evolution strategy) in environments that vary across generations outperformed agents evolved in a fixed environment. Our H2-OE (S2) went the other way for survival-based OE (OE − OE-fixed = −0.121 [−0.180, −0.068]). ES-DR is score-based evolution with the environment redrawn every generation (tuned: one episode per candidate per generation, every candidate in a generation on the same environment). The missing control is the same algorithm in one fixed environment: ES-fixed. ES-fixed's result is unknown at the time of writing.

- **Can answer:** in our world, does training in varied environments help score-based evolution (ES) the way Milano et al. found, and does the effect of variety differ between ES and OE?
- **Cannot answer:** why the effect differs, if it does. ES and OE differ in more than selection (ES scores candidates in separate episodes with 16 copies each; OE lives in one shared world with births and deaths; OE has no repeated trials within a life). Also not answered: Milano et al.'s own task, other rates of change, Q3 and Q4.

### 16.2 Method and tuning

- **ES-fixed:** ES-DR's algorithm and code (evosax `Open_ES`, antithetic sampling, centred-rank shaping, common random numbers within a generation, the same network, the default body), trained only in the canonical environment (L 60, F 40, uniform, ρ 0.03), as OE-fixed and PPO-fixed are. Training budget 2 × 10⁸ agent-steps, counted as for every method. Code: `ADDENDUM_METHODS` in `src/evo2/run.py`; the core list `METHODS`, and so the core analysis, is unchanged.
- **Tuning:** the core procedure (Section 5) unchanged: 8 trials, seeds 4500–4507 (method index 5), settings drawn from the ES space (SIGMA, LR, POP_SIZE, EPISODES) with each seed's `tuning_draw` stream, starting from the same untuned settings every core method started from, a quarter of the training budget per trial, scored zero-shot on the 24 validation environments (8 episodes), ties to the lower trial number. The test split is not read.
- **Where the chosen settings go:** `tuned_addendum.yaml` in the results folder (`tune.py --write-addendum`), not `configs/part3/tuned.yaml`, which stays as it was. Tuning and the final runs happen in one job on one machine, so the final runs read the settings from that file (`train.py --tuned-extra`); each run records them in its `run.json`, and the analysis checks that they equal the tuning's choice. This differs from the core methods, whose tuned settings were committed before their final runs.

### 16.3 Runs and evaluation

- **Runs:** ES-fixed seeds 5000–5009 (the seeds of every core method). Training curves (`curves.py`) and the final test evaluation (`evaluate.py --split test --final --adapt`) exactly as for the core runs: the same 30 test environments, episode seeds, adaptation budget (2 × 10⁷) and checkpoints. Every look at the test split is appended to the test-access log.
- **Check reruns:** ES-DR seeds 5000 and 5001 are rerun on the same machine and commit, after the ES-fixed runs. The core ES-DR runs are the ones compared; the reruns only check that the new code and machine give ES-DR's level (the same seed on another machine is not bit-identical). Rule: if a rerun's mean primary score (T2, T3) lies more than 0.01 outside the range of the 10 core ES-DR runs, the report flags A1 and A2 as possibly shifted by machine or code, next to their verdicts. A missing rerun is reported and does not stop the analysis.
- **Exclusion:** an ES-fixed seed whose run wrote `failed.json`, crashed, or did not finish (for example, cut off by a money or time stop) is reported and left out. There are no replacement seeds in this addendum (budget). If more than 2 of the 10 are left out: no verdicts; the runs that finished are reported descriptively.

### 16.4 Analysis

`scripts/part3/analyze_addendum.py`; the functions are in `src/evo2/analysis.py` and tested in `tests/test_part3_addendum.py`.

- **Normalization:** with the core analysis's random-policy and best values (`results/part3/final/analysis.json`), so no core number changes. An ES-fixed score above the core best gives a normalized value above 1. If the new runs' random-policy competence differs from the core's by more than 10⁻³ in any environment: stop and record a protocol amendment.
- **A1 (our adaptation question):** D = IQM(ES-DR) − IQM(ES-fixed) on the primary metric (T2, T3), 95% CI from 50,000 whole-run bootstrap resamples (seed 4800), verdict by the H1 rule with the same margin, 0.05: support / support (below the margin) / against / against (below the margin) / no meaningful difference / inconclusive. **Prediction from Milano et al.: ES-DR > ES-fixed.** Limit: the primary metric is an area under the adaptation curve in which the zero-shot point has weight 0.05, and ES adapts fast (core ES-DR on T2: 0.711 at point 0, 0.841 after one tenth), so A1 can show little of a difference that exists before adaptation; A3 measures that.
- **A2 and A3:** one Holm family (α = 0.05): two-sided Mann-Whitney U on per-run means, with the labels of Section 8.2.
  - A2: ES-DR vs ES-fixed, primary metric, T2 and T3; predicted ES-DR > ES-fixed.
  - A3 (Milano et al.'s claim): ES-DR vs ES-fixed, normalized zero-shot competence on T1 (their kind of measure: no adaptation, new environments drawn from the training ranges); predicted ES-DR > ES-fixed.
- **A4 (descriptive, not a verdict):** I = (IQM(ES-DR) − IQM(ES-fixed)) − (IQM(OE) − IQM(OE-fixed)) on the primary metric (T2, T3), with a CI from one whole-run bootstrap over the four methods' runs (as H2b). Its label (CI above 0, "variety effect larger for ES than for OE"; below 0, "variety effect larger for OE than for ES"; otherwise "no difference found") compares the two effects only. Because OE − OE-fixed is already known (−0.121), I is positive unless ES-fixed beats ES-DR by more than about 0.065, so A4 adds little beyond A1 and is not used in the reading guide. It also mixes the kind of evolution with adaptation speed (ES adapts fast; OE got worse during adaptation).
- **Reading guide (fixed now).** Two questions, read separately:
  - **Milano et al.'s claim (A3).** "Supported": in our world, score-based ES evolved in varied environments did better on new environments from the training ranges, before any adaptation, as Milano et al. found; set against S2 and the exploratory OE result, their direction holds for score-based ES and not for survival-based OE, and the kind of evolution (with the other ES/OE differences in 16.1) is the likely reason; which part of it is not tested. "Evidence against": ES-fixed did better, so our world did not reward variety for score-based evolution either. "Not supported": no difference found; the reversal for OE cannot be put down to the kind of evolution.
  - **Our adaptation question (A1, with A2).** "Support" (CI above the margin): varied training made ES adapt better. "Support (below the margin)": a real but small effect, under 5% of the random-to-best range; reported as "a small advantage, below the pre-set margin", not as a meaningful difference, and it does not by itself support Milano et al.'s direction. "Against" or "against (below the margin)": ES-fixed adapted better (the same distinction applies). "No meaningful difference" or "inconclusive": as worded. With H2-PPO's null (S3), a non-support result means our world may not reward variety in adaptation for any method.
  - If A3 and A1 disagree, both are reported as they are: A3 answers Milano et al.'s claim, A1 our adaptation question.
- **Descriptive:** means by test set, raw zero-shot competence by set, the ES-DR check reruns against their core runs, the tuning trials, training hours, the test-access log's entries for this stage, and the stage's spend.
- **Checks before any verdict (the analysis stops otherwise):** every ES-fixed run trained and was evaluated from one clean commit; its settings equal the tuning's choice; its evaluation used the pre-registered settings and budget (the core analysis's own checks, `analyze.py: read_eval`). The OE vs OE-fixed zero-shot T1 comparison computed on 4 Nov (raw competence 13.68 vs 19.35, after the results) stays exploratory and is not part of this addendum's tests.

### 16.5 Budget and stops

The extension used one RTX 4090. Prespecified expenditure and runtime limits could stop the stage early. ES-fixed seeds ran before the ES-DR check reruns; unfinished seeds would be omitted under Section 16.3. All ten ES-fixed runs and both check reruns completed.
