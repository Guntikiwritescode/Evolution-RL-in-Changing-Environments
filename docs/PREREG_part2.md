# Trait experiment: scientific protocol

Excerpts from the study protocol, retaining original section numbers and scientific rules. Original document SHA-256: `71a125c9ebb937d103fb86033cd9f4229d6b433865ee6c533100ab0f06b0d842`.

Sections 0 and 12–17 of the original protocol are omitted. References to those sections identify historical material rather than sections in this excerpt. The implementation notes in Sections 9.2 and 9.3 describe the supplied code and selected configurations where the amendment text is unavailable.

## 1. The question, and what this design can and cannot answer

**Question:** In this trait-only blob world, when a population moves to a new environment, does it matter whether it can still change? If being able to change helps, where does the help come from?
- the starting point;
- variety present from the start;
- variety built up during training;
- new mutations after the move.

**Can answer** (for this world only): how each group does after the move, what it costs in compute, and which of the sources above explain any difference. This only works if B actually separates groups; Section 6 says how that is checked.

**What the main arm can and cannot show.** Part 1 shows that ET nearly fills B: it is at 24.27 of a possible 25, and it never died out (Section 0, point 4). So in the main arm:
- any group can do at most about 0.7 blobs better than ET on P1, which is below the 2.5-blob threshold;
- on P2, a group can only be 10 points better than ET if ET dies out in at least 10% of runs, and it died out in 0% in Part 1.

So in the main arm, rows 1–5 of the guide (Section 7) can only come out as "ET better", "the same" with a ceiling label, "mixed" or "inconclusive". The main arm can show a group *failing* where ET doesn't. It cannot show a group *beating* ET.
- On P1, that limit comes from B's food count (25).
- On P2, it comes from ET's Part 1 record: 0 of 30 runs died out. With 30 runs, the true rate could still be up to about 12% (exact 95% upper bound 11.6%), so this limit is likely but not certain.

Arm 3 (Section 9.2) is a harder B where P1 results can go either way. P2 will probably stay one-sided there too, because the arm 3 rule allows ET at most 3 extinctions in 30.

**Cannot answer:** anything about neural-network agents, learning within a lifetime, or other worlds. That is Part 3.

---

## 2. Setup (fixed)

**Frozen from Part 1:** the `world`, `env_A`, `env_B`, `blob_default`, `mutation_sd` and `phases` sections of `configs/part1/final.yaml`. That means:
- mutation size 1.0 for speed, size and sense, and 0.25 for reach;
- time limit 200 ticks per generation;
- blob-eating off (main arm);
- A: 500 × 500, 50 food, uniform;
- B: 300 × 300, 25 food in 5 clusters, spread 15, spots fresh each generation, blobs starting at random edge points;
- 100 generations in A, then 50 in B.

A hash of exactly these sections, as they are in `configs/part1/final.yaml`, is checked before launch. Arms 2 and 3 apply one B-phase change each on top (Section 9). Those changes are written in their own config files and checked separately, so the hash check is on the sections before any arm's change. Nothing is recalibrated.

**Phases.** 100 generations in A, then 50 in B, for every group except ORACLE_B, which starts directly in B (Section 4). The population that starts B generation 0 is the one made at the end of A generation 99, so that reproduction uses A's mutation settings (Part 1 rule T1).

**Deployment streams.** For each run seed there are three deployment streams, shared by every group in every arm:
- food: one child stream for each phase and generation;
- behavior: one child stream for each phase (it also drives reproduction, as in Part 1);
- starting spots: one child stream for each phase.

ORACLE_B uses the B-phase streams.


---

## 3. The grid search (fixed)

**Grid.** Four traits, 8 points each, evenly spaced on a log scale from 0.25× to 4× the default:
- speed and size: 2.5 to 40;
- sense: 5 to 80;
- reach: 0.25 to 4.

**Merging.** In every grid point, reach is replaced by its effective value, max(reach, size ÷ 4). Points that are then the same are merged into one design. That leaves **1,472 distinct designs**, each carrying its effective reach as its real reach.
- For clones this changes nothing, because clones behave by effective reach.
- It matters for GS_pop_mut, and the choice is an **assumption**, stated now. Its offspring start from the effective reach, so a reach mutation can change behavior right away.
  - Picking another merged point (a lower reach) would hide some reach mutations below size ÷ 4, as happens to ET: its default blob has reach 1, below size ÷ 4 = 2.5.
  - The chosen reach also stays when a child's size shrinks, which can give GS_pop_mut a small head start on reach compared with ET.
  - Neither choice is neutral. The effect is limited, because a size mutation can hide reach again.

**Streams.** In each stage of a run seed's search, every design uses the same streams (food, behavior, starting spots). A design on both score lists is run once.

**Stage 1 (screening, in A).** Every distinct design runs as 50 clones for 30 generations on one shared set of streams. Generations 0–9 are burn-in.
- **Population score:** mean population over generations 10–29 (a generation with no blobs counts as 0).
- **Food score:** mean over generations 10–29 of (food eaten + blobs eaten) ÷ population. A generation with no blobs scores 0. This is the Part 1 definition (`gridsearch.score_log`, rule GS4), over the same 30-generation horizon as Part 1.

**Stage 2 (re-scoring, in A).** For each score separately, the 32 best designs from stage 1 run again on 8 new shared sets of streams. Both lists use the same 8 sets.
- **Population score:** designs on the population list run for **100 generations**, scored on generations 10–99, so the winner must last the whole of training.
- **Food score:** scored on generations 10–29, the same window as stage 1. Designs only on the food list run for 30 generations, since later generations don't count. A design on both lists runs once, for 100 generations; its first 30 generations are the same as a 30-generation run on the same streams.
- Each score is averaged over the 8 runs. The stage-1 run is not included, so stage-1 luck can't carry into the stage-2 score.

**How 2025-style GS_food_frozen still is.** It keeps the 2025 goal (food per blob, with extinct generations scoring 0) and the 30-generation horizon. It differs from the Part 1 search in these ways:
- the grid has a reach axis (0.25 to 4; Part 1 fixed reach at 1);
- there are 1,472 merged designs instead of 512;
- there is a burn-in (the plan asks for one);
- there is a top-32 screen re-scored on 8 streams, with tie rules;
- there is one search per run seed (the plan asks for this), instead of one search shared by all runs.

**Tie rules** (applied at the stage-1 cut for the top 32, and when picking the winner):
1. The higher score.
2. Then the higher worst generation: the lowest population in any scored generation, over all runs used for that score.
3. Then a random tie-break key. At the start of each search, the run seed's tie-break stream gives every design one random number, drawn once. The design with the higher key wins every tie in that search: at the cut, at the winner, and for the stage-1-only winners in Section 18.5. No other draws are made, so adding or removing a later use can't change an earlier pick.

The number of designs tied after rule 1 is reported for every run.

**Winners.**
- The **population winner** has the best stage-2 population score.
- The **food winner** has the best stage-2 food score.
- If both are the same design in a run, their deployed runs are identical; how often this happens is reported.

**Search in B (for ORACLE_B and for check U2).** The same procedure, with these differences:
- clones start in B;
- stage 1 runs for 30 generations, scoring generations 10–29;
- stage 2 runs the top 32 for 50 generations (the length of B), scoring generations 10–49 on 8 sets of streams;
- the winner is chosen by population score only.

**U2 check runs.** The B winner and the A population winner each run as 50 clones in B for 50 generations, on 8 **fresh** sets of streams ("U2 sets"). They are scored on generations 10–49. Fresh streams are used because the B winner was chosen as the best of 32 on the stage-2 streams, so its score on those streams is biased upward.

**Cost.**
- The A search's creature-steps (both stages, both lists) are charged to every GS group.
- The B search is charged to ORACLE_B.
- The U2 check runs are reported but not charged.
- One blob moving one tick counts as one step (guess K1).

**Limitation, stated now:** mutation adds noise of a fixed size to a trait. A change of 1.0 is 40% of a speed of 2.5 but 2.5% of a speed of 40. So how strongly GS_pop_mut can change depends on which design won.

---

## 4. Groups (fixed)

| Name | Starts with | Can mutate? | What it tests |
|---|---|---|---|
| ET | 50 copies of the default blob | Yes, in A and B | Evolution, as in 2025 |
| ET_freezeB | Same as ET; identical through A, same seed | In A yes; **in B no** | **New.** Do new mutations after the move matter? |
| ET_cloneB | Same as ET through A. At the move to B, its population is replaced by the same number of clones of **one of its own blobs**, picked at random (each blob starting B generation 0 equally likely, from the run seed's clone-pick stream). If ET died out in A, so has this group | In A yes; in B no | **New.** Compared with ET_freezeB: does the variety built up in A matter, or would copies of one typical member do as well? |
| GS_food_frozen | 50 clones of the food winner | No | Food-score selection |
| GS_pop_frozen | 50 clones of the population winner | No | Population-score selection |
| GS_pop_mut | 50 clones of the population winner | Yes, in A and B | Population-score selection followed by mutation |
| SV_frozen | 50 random designs (each trait log-uniform over the grid range, drawn separately for each blob from the run seed's SV stream) | No | Is starting variety alone enough? |
| ORACLE_B | 50 clones of the B winner, placed directly in B for 50 generations | No | Reference only: what re-searching for the new world gets, and what it costs |

All groups that mutate use the frozen mutation sizes. ET, ET_freezeB and ET_cloneB are identical through the end of A generation 99.

**Why a random member, not the average design.** An earlier revision used clones of the average of each trait. A pre-final diagnostic on Part 1 populations at B generation 0 (n = 30) found that such a clone uses about 24% less energy per move than the population's average (ratio 0.76, range 0.65–0.86). That is because energy cost grows mainly with size³ × (speed × size)² (`kernel.motion_cost`), so averaging the traits doesn't average the cost. Averaging raw reach also lowered effective reach in 27 of 30 runs. So "the average design" is not an average blob. A randomly picked member is, on average over runs, a typical blob.

**A-checks for rows 5 and 6** (descriptive, no tests, not charged). These show whether a difference in rows 5 and 6 is special to the move or would happen anyway. From ET's population at the end of A, three runs continue **in A** for 50 more generations, on the run seed's A-check streams:
- the population with mutation, like ET;
- the population without mutation, like ET_freezeB;
- clones of the same randomly picked blob, like ET_cloneB.

For each, mean population over the last 10 of those generations is reported. The differences, with 95% CIs, are shown next to rows 5 and 6. A is near its own cap of 50 (ET averaged 48.0 over A generations 90–99 in Part 1), so these checks can be at a ceiling too.

---

## 5. Outcomes (fixed)

**Primary** (from the plan):
- **P1, population in B:** mean population over B generations 40–49. A generation with no blobs counts as 0, and so does every B generation of a run that died out in A.
- **P2, extinct at the end of B:** yes if no blob would start B generation 50, meaning survivors plus births in B generation 49 add up to 0. A run that died out in A counts as yes. (Part 1's `extinct_end` used the population at the start of B generation 49; see Section 15, C16.)

**Secondary:**
- **S2, survival curves in B:** the share of runs with blobs, by B generation.
- **S3, compute cost:** creature-steps, including search cost (guess K1).
- **S4, food per blob in B:** mean over B generations 40–49, among generations with blobs; the number of such generations is reported. **Warning:** this goes up when a population shrinks, so it never stands alone.
- **S5, reached B:** whether B generation 0 started with any blobs.
- **S6, population over B:** mean population over B generations 1–49. Generation 0 is left out because it reflects A.

**Descriptive only (no tests):**
- S1, the first B generation, counting from generation 1, with 23 or more blobs. Part 1 suggests this sits at its floor (Section 0, point 4).
- Trait variety over time (CV² and Shannon H, using the frozen Part 1 definitions on the raw traits).
- ET's mean traits over time.
- Search checks: tie-set sizes; search score vs deployed performance; and the B stage-1 score of every design tied for best in A. That last one shows whether being best in A predicts doing well in B; it costs nothing extra, because the B search scores every design.

---

## 6. Statistics (fixed)

**Planned comparisons.** Each pair is tested on P1 and P2. Differences are always the first group minus the second.

| # | Comparison (first − second) | Why |
|---|---|---|
| 1 | GS_food_frozen − ET | plan |
| 2 | GS_pop_frozen − ET | plan |
| 3 | GS_pop_mut − ET | plan |
| 4 | SV_frozen − ET | plan |
| 5 | ET_freezeB − ET | new: do new mutations matter? |
| 6 | ET_cloneB − ET_freezeB | new: does variety built up in A matter? |
| 7 | GS_pop_frozen − GS_food_frozen | the plan's interpretation guide uses it |
| 8 | GS_pop_mut − GS_pop_frozen | the plan's guide needs it; same design, only mutation differs |

ORACLE_B is reference only: no test.

**Families.** Each arm is its own Holm family:
- main arm: 8 pairs × 2 outcomes = **16 tests**;
- arm 2: comparison 7 is left out, because its data are identical to the main arm by construction (Section 9.1), so **14 tests**;
- arm 3: **16 tests**.

Below, k is the family size.

**Tests and effect sizes:**
- **P1.** Two-sided permutation test on the difference in means: 10,000 permutations; p = (number at least as extreme + 1) ÷ 10,001, where "at least as extreme" means |permuted difference| ≥ |observed difference| − 10⁻¹².
  - Effect: the difference in means, with a **studentized bootstrap** CI (20,000 resamples, drawn separately within each group, using the Welch standard error). If both groups are constant, the CI is the point estimate.
  - Why this method (decided 1 Nov, before any run). A pre-final simulation examined a boundary case: one group like ET in Part 1, the other the same but with about 10% of runs crashing to 0, so the true difference is exactly 2.5. The 99.375% percentile bootstrap called this "equivalent" too often, and so did the Welch t interval.

    The recorded check used 800 simulated datasets per case and 10,000 resamples; its script was not retained:

    | Runs | Percentile | Welch t | Studentized |
    |---|---|---|---|
    | 100 | 1.4% | 1.9% | 0% |
    | 200 | 0.75% | 1.4% | 0% |

    With partial crashes instead of zeros, the studentized method was 0% at 100 runs and 0.13% at 200. The target is at most 0.31%.

    When two groups truly don't differ, the studentized method still calls them equivalent nearly as often as the others: 35% vs 44% at 100 runs, and 92% vs 94% at 200, with 5% zeros in both groups. Section 18.3 checks this again on the final code.
  - The plan's Mann-Whitney U and Cliff's delta are also reported, but they don't decide verdicts (Section 15, C12).
- **P2.** Two-sided Fisher's exact test. Effect: the difference in the share of runs extinct, with a Newcombe (hybrid score) CI.
- **CI levels.** Every effect is reported with a 95% CI, as the plan asks. Equivalence decisions use a second CI at level 1 − 2 × 0.05 ÷ k (99.375% for k = 16; 99.29% for k = 14).
- **Holm correction** across each arm's family.
- If both groups have the same constant value on an outcome, the p-value is 1. Both tests give this anyway; the code sets it so nothing breaks.
- Every permutation and bootstrap uses its own stream, split from seed 2999 by arm, comparison and outcome.
- No tests on values fixed by design.

**Smallest differences that matter:**
- P1: **2.5 blobs**, which is 10% of B's 25 food items.
- P2: **10 percentage points.** This matches P1: with ET near 24.3 blobs, 10 more runs in 100 dying out lowers mean P1 by about 2.4 blobs.
- A difference "reaches" the threshold if its absolute point estimate is at least the threshold. Only the point estimate is checked, not the whole interval. A CI is "inside" if both ends are within ± the threshold, ends included (with a tolerance of 10⁻⁹ for rounding).
- **Exact arithmetic.** Every group has the same number of runs N. So threshold and label checks are done in whole numbers, which avoids rounding errors such as 21/200 − 1/200 coming out just under 0.10:
  - P1 is stored as each run's sum over its 10 generations.
  - A P1 difference reaches 2.5 if the two groups' totals differ by at least 25 × N.
  - The ceiling and floor checks use totals of at least 225 × N and at most 25 × N.
  - A P2 difference reaches 10 points if the extinction counts differ by at least N ÷ 10.

**Verdict for each comparison and outcome.** Checked in this order; the first that applies is the verdict:
1. **Meaningfully different:** Holm-adjusted p < 0.05, and the point estimate reaches the threshold.
2. **Equivalent:** the equivalence CI (level 1 − 2 × 0.05 ÷ k) is inside ± the threshold. This is a two one-sided equivalence test at the Bonferroni level. If the Holm-adjusted p is also below 0.05, it is labeled "equivalent, with a small real difference".
3. **Different, size unclear:** Holm-adjusted p < 0.05, but neither 1 nor 2 applies.
4. **Inconclusive:** none of the above. Never reported as "no difference".

**Labels on "equivalent" P1 verdicts.** These mark cases where the cap or the floor decides part of the result. Each label is checked in this order:
- **At the ceiling** (main arm, arm 3 and arm 4, where 25 is a hard cap): both groups' mean P1 is at least 22.5 (25 minus the threshold). Neither group can be 2.5 above the other, so both halves of the equivalence hold by construction. Both did about as well as this B allows.
- **Capped on one side** (main arm, arm 3 and arm 4): only the higher group's mean P1 is at least 22.5. Then the lower group can't be 2.5 *above* it, so that half of the equivalence holds by construction. Only the other half, "the lower group is not 2.5 *below*", is evidence.
- **At the floor** (all arms): both groups' mean P1 is at most 2.5. Both did about as badly as possible.
- In arm 2, blob-eating means 25 is not a hard cap, so the two ceiling labels aren't used there; the share of runs at 22.5 or more is still reported.

**P2 equivalence has a floor built in.** With 100 runs, "equivalent" on P2 is only possible when both groups have at most 4 extinctions in 100 (see Power, below), so then it means "both groups rarely die out". With 200 runs it is possible up to about 15% extinctions in both groups (30 of 200). When the group with fewer extinctions has under 10%, the other group can't be 10 points better. Both facts are reported next to every P2 "equivalent", not as a label.

**"Better"** means higher P1 or lower P2. **"Significantly worse"** means a verdict of "meaningfully different" or "different, size unclear" in the worse direction.

**Overall verdict for each comparison** (combining P1 and P2):
- **First group better:** meaningfully better on at least one outcome, and not significantly worse on the other.
- **Second group better:** the mirror of that.
- **Mixed:** meaningfully better on one outcome, and significantly worse on the other.
- **The same:** equivalent on both outcomes. It carries the P1 label if there is one: **"the same (at the ceiling)"**, **"the same (capped on one side)"** or **"the same (at the floor)"**.
- **Inconclusive:** anything else.

**A-phase label.** If any run reached B in one group but not in the other (S5 differs at all), the overall verdict is labeled **"includes A-phase losses"**. The report gives the two counts of runs that reached B, and how many of each group's P2 extinctions happened in A. P1 and P2 count runs that died out in A, as the plan's outcomes do.

**When the difference is all from A.** A verdict of "first group better", "second group better" or "mixed" that carries the A-phase label is checked against the runs-that-reached-B analysis (below). The row's conclusion gets the ending "but the difference arose in A, before the move" if either of these holds:
- fewer than 30 run seeds have both groups reaching B;
- for each outcome that was meaningfully different, the runs-that-reached-B point estimate does not reach the threshold in the same direction.

Otherwise the conclusion gets the note "part of this difference happened before the move".

**Positive control: "B separates groups"** (per arm). It holds if at least one planned comparison passes all three of these:
- both groups reached B in every run;
- both groups' mean B-generation-0 population is at least 40 (80% of the 50 starting blobs). ET averaged 47.4 in Part 1 (lowest single run 40);
- the overall verdict is "first group better", "second group better" or "mixed".

The first two conditions keep out groups that only limp into B. Comparisons 5 and 6 pass them whenever ET passes them, because their groups are identical through A. If the control does not hold, every verdict is still reported, but the interpretation rows that rest on "the same" give **no conclusion**, with the reason "this B does not separate any groups".

**Secondary analyses** (reported, never decide verdicts):
- **Paired by run seed.** All groups with the same run seed share deployment streams, so runs pair by seed:
  - P1: a sign-flip permutation test (10,000 flips) on the per-seed differences, with a 95% bootstrap CI that resamples seeds;
  - P2: the exact McNemar test.

  These may have more power than the main tests. They are reported because the main tests ignore the pairing.
- **Runs that reached B.** P1 and P2 differences are recomputed over the run seeds where both groups reached B, with 95% CIs. This conditions on something that happens after the start, so it can be biased; it is shown next to any verdict labeled "includes A-phase losses".

**Power for P2.** Exact Fisher at 0.05 ÷ 16, the strictest Holm step; the other steps have more power. Computed before any run, with 100 runs per group:

| True extinction, group 1 vs group 2 | Power |
|---|---|
| 0% vs 10% | 0.55 |
| 0% vs 15% | 0.94 |
| 5% vs 20% | 0.55 |
| 10% vs 30% | 0.68 |

With 100 runs, a Fisher test at this level can only pass when the gap is at least 10 runs in 100, so this power is also the chance of a "meaningfully different" verdict. With 200 runs (Section 18.2) the two differ. For example, at 0% vs 10% the test passes 99.6% of the time, but the point estimate reaches 10 points only 53% of the time.

With 50 runs per group, all four would be 0.33 or lower. "Equivalent" on P2 would then be impossible even if no run died out in either group: the 99.375% CI for 0 of 50 vs 0 of 50 is ± 0.13, wider than the 0.10 threshold. With 100 runs it is ± 0.07. P2 can then be "equivalent" only in these cases (computed before any run): both groups have the same number of extinctions, from 0 to 4 in 100, or numbers that differ by 1 with the larger at most 3. P1 power depends on its spread, which the pilot measures.

---

## 7. Interpretation guide (per arm)

**The plan's guide, copied unchanged:**
- ET > GS_pop_mut: evolving from a simple start beats starting at the optimum and evolving.
- ET ≈ GS_pop_mut, and both > frozen groups: the advantage comes from being able to change, not from evolution vs. search.
- SV_frozen ≈ ET: starting variety alone is enough, and new mutations aren't needed.
- GS_pop_frozen > GS_food_frozen: part of the 2025 result came from the grid search aiming at the wrong goal.
- ORACLE_B: shows what a fixed design gets if the environment is re-searched, and the extra compute it costs.

**The same guide, made exact.** It uses the overall verdicts from Section 6.
- "Mixed" or "inconclusive" gives **no conclusion** for that comparison.
- "The same" gives the conclusion in the last column only if the positive control holds, and, where marked, U2 has not fired (Section 8).
- "The same (at the ceiling)" gives instead: "both groups did about as well as this B allows; this B can't show whether one is better".
- "The same (capped on one side)" gives instead: "[the lower group] is not meaningfully worse than [the higher group]; this B can't show whether it is better". This needs the positive control, and U2 where marked, like "the same".
- "The same (at the floor)" gives instead: "both groups did about as badly as possible; no conclusion".
- A verdict labeled "includes A-phase losses" keeps its row's conclusion with the ending or note set in Section 6 ("but the difference arose in A, before the move", or "part of this difference happened before the move"). The runs-that-reached-B numbers are shown next to it.
- Rows 5 and 6 are shown next to their A-checks (Section 4).
- **Row 4's qualifier.** The number of distinct designs at B generation 0 is reported for SV_frozen, ET and GS_pop_mut in every run. SV_frozen can't mutate, so selection in A may leave only one design. If SV_frozen has one distinct design at B generation 0 in more than half its runs, row 4 is reworded:
  - "first group better" → "the design that selection in A picked from 50 random starts beats evolution";
  - "second group better" → "evolution beats the design that selection in A picked from 50 random starts";
  - "the same" → "the design that selection in A picked from 50 random starts does as well as evolution".

  In that case row 4 no longer measures variety in B.

| # | If the first group is better | If the second group is better | If the same |
|---|---|---|---|
| 1 GS_food_frozen − ET | The 2025-style search beats evolution here; the 2025 claim fails | Evolution beats the 2025-style search, as in 2025 | The two do equally well |
| 2 GS_pop_frozen − ET | Population-score selection beats evolution | Evolution beats the best fixed design | Being able to change doesn't help a well-chosen design here (needs U2 not fired) |
| 3 GS_pop_mut − ET | Starting at the A optimum and evolving beats starting simple | Evolving from a simple start beats starting at the optimum and evolving | Once both can evolve, the starting point doesn't matter |
| 4 SV_frozen − ET | Variety without change beats evolution | Evolution beats variety without change | Starting variety alone is enough, and new mutations aren't needed |
| 5 ET_freezeB − ET | Stopping mutation after the move helps | New mutations after the move help | New mutations after the move don't matter (needs U2 not fired) |
| 6 ET_cloneB − ET_freezeB | Copies of one typical member do better than the varied population | Variety built up in A helps after the move | Copies of one typical member do as well; variety from A doesn't matter (needs U2 not fired) |
| 7 GS_pop_frozen − GS_food_frozen | Aiming at population beats aiming at food per blob: part of the 2025 result came from the wrong goal | Aiming at food per blob does better | The search's goal doesn't matter |
| 8 GS_pop_mut − GS_pop_frozen | Being able to change helps a good fixed design | Being able to change hurts | Being able to change doesn't matter for this design (needs U2 not fired) |

**The plan's combined rows:**
- "ET ≈ GS_pop_mut, and both > frozen groups" holds if all of these hold:
  - rows 2 and 4 are "second group better" (ET better);
  - row 8 is "first group better";
  - row 3 is "the same", "the same (at the ceiling)" or "the same (capped on one side)".

  If row 3 is "the same (at the ceiling)", the conclusion carries that label: "both groups that can change did as well as this B allows, and the frozen groups did worse". (Rows 2, 4 and 8 already show that this B separates groups.) "The same (capped on one side)" is accepted the same way, with that label. Accepting these labels here departs from the plan's plain "ET ≈ GS_pop_mut"; the departure was recorded before the final runs.
- This is narrower than the plan's wording: GS_pop_mut is not tested against SV_frozen or GS_food_frozen (Section 15, C14).
- "SV_frozen ≈ ET" is row 4 "the same".

---

## 8. Checks that would make results useless, fixed now

These are prespecified diagnostic checks. Each is run on the pilot and on the final data, per arm, and every result is reported.

**U1. Ceiling and floor.** Built into the verdict labels (Section 6). Also reported for each group: the share of runs with P1 of 22.5 or more, and the share with P1 of 2.5 or less.

**U2. B doesn't call for a different design.** In each run, compare the B winner's score on the U2 sets with the A population winner's score on the same sets (Section 3). U2 fires if the B winner is ahead by less than 2.5 blobs in at least half the runs.
- *If it fires:* "the same" in rows 2, 5, 6 and 8 gives no conclusion, with the reason "B doesn't call for a different design".
- *In the pilot:* if it fires, stop and record a protocol amendment.

**U3. The search winner still doesn't hold up.** Averaged over runs, compare the population winner's stage-2 score (generations 10–99) with GS_pop_frozen's deployed mean population over A generations 10–99 (died-out generations count as 0). U3 fires if the deployed value is lower by more than 2.5 blobs. The average over runs is used so that single-run noise doesn't trigger it. If it fires in the pilot, the rule in Section 10 applies.

**U4. GS groups die out before reaching B.** If a group dies out in A in more than half its runs, the report adds its B outcomes among the runs that reached B, as descriptive only.
- *In the pilot:* if this happens for GS_pop_frozen, stop and record a protocol amendment.

**U5. Degenerate tests.** Handled in Section 6: p = 1 only when both groups have the same constant value.

---

## 9. Other arms

### 9.1 Arm 2: blob-eating on in B (tests the food assumption)

**What.** The whole experiment again, with blob-eating **on in B only**. A is the same as in the main arm, because calibration clearly rejected blob-eating in A. Everything else is frozen.

**Why B only.** Part 1 left a conflict: the training numbers favor blob-eating off, but the paper's B numbers need more food than B places. Turning it on in B only tests that question directly, without bringing back the A setting that calibration rejected.

**Which runs change.**
- Identical clones can never eat each other. In `kernel.py` (lines 295 and 420), a blob can only eat another if its size × 0.8 is at least the other's size. Equal sizes fail that check before the random draw at line 302. The turn-order shuffle (lines 243–247) is drawn every tick whether blob-eating is on or off.
  - So GS_food_frozen, GS_pop_frozen, ET_cloneB and ORACLE_B should be identical to the main arm.
  - A test checks this before arm 2 runs; if it fails, they are rerun.
- ET, ET_freezeB, GS_pop_mut and SV_frozen are rerun from the start of B. Their A phase is identical to the main arm.
- The searches are shared, because they only evaluate clones.

**Built-in difference, stated now.** In arm 2, a blob can only eat another blob that is at most 80% of its own size. Clone groups have equal sizes, so they never can; groups with varied sizes can.
- **Clone vs varied: comparisons 1, 2, 6 and 8.** Here arm 2 differences partly come from one group's access to this extra food, not only from adapting to B.
- **Varied vs varied: comparisons 3, 4 and 5.** Both groups can eat each other, but how much depends on how spread out their sizes are. SV_frozen's sizes span 2.5 to 40. GS_pop_mut starts as clones and spreads only through mutation. In all 30 Part 1 runs, ET at B generation 0 had at least one pair that could eat each other (checked on the Part 1 files).
- For every group in arm 2, the number of blobs eaten per B generation is reported. The note above is shown next to every affected comparison.

**Analysis.** Arm 2's family has 14 tests (comparison 7 left out; Section 6).
- **Holds or changes.** For this check only, verdicts in both arms are recomputed with a fixed level for every test: p < 0.05 ÷ 16, and the equivalence CI at 99.375%, instead of Holm. This way a verdict can't change between arms only because *other* tests changed. A comparison **holds** if its overall verdict is in the same category in both arms. The categories are: first better, second better, mixed, the same, inconclusive. The labels are notes, not categories. Otherwise the result **changes**.
- Comparison 7 is reported once, as identical in both arms by construction.

**Seeds.** The same run seeds as the main arm. This is deliberate: the A phases are identical, so only B differs. It stays within the Part 2 seed range.

**Descriptive extra.** Whether ET's mean P1 goes above 25. The paper reports about 29.

### 9.2 Arm 3: a harder B (original proposal; required amendment adoption under original Section 16, omitted here)

**Why.** In the main arm no group can beat ET (Section 1). A B where ET does not fill the food supply lets results go both ways.

**What.** B with one change: a larger map. It still has 25 food in 5 clusters, spread 15, fresh spots each generation, random edge starts, 50 generations and blob-eating off. Blobs start further from the food, so finding it is harder. Everything else is as in the main arm, including the same run seeds. A is identical, so only B is rerun.

**How the map size is set (before any final run, by a rule fixed now):**
- Candidate sides: 400, 500, 600, 800, 1,000.
- Seeds 2800–2829, set aside for this only. Each is split into the same named child streams as a final run seed (Section 10).
- For each seed, ET runs through A once (as in the main arm). From its end-of-A population, it then runs 50 generations in each candidate B, on the same B-phase streams.
- Choose the smallest side at which ET's mean P1 is at most 20 (80% of the 25 food) and ET is extinct at the end of B (P2, which also counts deaths in A) in at most 3 of the 30 runs. That leaves room above ET (up to 25) and below it on P1.
- If no candidate qualifies, stop and record a protocol amendment.
- All 30 A runs and 150 B runs are reported in an appendix, including every run.

Only ET is used, and no other group is run on these seeds. The final seeds are separate, so picking the side doesn't bias comparisons between groups. It only means ET's final mean P1 in arm 3 may come out a little above 20, since the side was picked where ET happened to score at most 20 on these 30 seeds.

**Analysis.** Arm 3 has its own B search (for ORACLE_B and U2), its own Holm family of 16 tests, the same rules and labels as the main arm, and the same holds/changes check against the main arm (with the fixed level from Section 9.1).

**Seeds.** The same run seeds as the main arm, so A phases are identical and only B differs.

**Implementation note: adopted map setting.** The supplied `configs/part2/base.yaml` and `scripts/part2/maprule.py --round 2` define a second candidate set of 650, 700 and 750, evaluated on seeds 2830–2859 with the same mean-P1 and extinction criteria above. `configs/part2/arm3.yaml` records side 750 and rule revision `e5423a8ec25b0368a06f318bf617aba04042d4c0`; `RESULTS_part2.md` reports that setting for the final experiment. The associated result path is `results/part2/maprule_round2/maprule.json`. These are implementation and result-record details, not a reconstruction of the omitted amendment's wording.

### 9.3 Arm 4: a B that calls for a different design (added after the pilot, at the design amendment, 1 Nov 2026)

**Why.** In the pilot (seeds 2900–2909), check U2 fired in the main B and in arm 3's B, in all 10 runs. The best design found by searching directly in B was no better, or under 1 blob better, than the best design found in A.
- The A winner was a small, fast, cheap design (speed 40, size 3.7).
- Every B tried also rewarded small, fast, cheap foragers.
- So none of them tested whether being able to change helps when the world asks for something different.

B was redesigned before any final run.

**What.** The main B with one change: **less energy per generation in B** (`world.energy_per_generation`, 500 in A). Moving fast costs much more energy than moving slowly (`kernel.motion_cost`), so a low budget should favor a different, slower or cheaper design. Everything else is as in the main B: 300 × 300, 25 food in 5 clusters, spread 15, fresh spots, random edge starts, tick limit 200, 50 generations, blob-eating off.

**How the energy level is set (clones only; no compared group is used):**
1. **Probe** (seeds 2870–2871). For energy 500, 400, 300, 250, 200, 150 and 100, compare the stage-1 landscapes of A and B, and see where the top A designs rank in B. This probe preceded the final runs. The candidate list for step 2 is chosen from the probe, and the probe and the choice are both reported.
2. **Rule** (seeds 2860–2869; round 2 on seeds 2872–2881 with a median test, at the selected setting: original Section 17, 2 Nov 2026, omitted here). For each candidate, run the full A search and B search (Section 3) and the U2 runs (Section 8).
   - Choose the highest energy (the smallest change from the main B) where, averaged over the 10 seeds, the B winner beats the A winner by **at least 5 blobs** (twice U2's 2.5) on the U2 sets.
   - The B winner's U2 score must also average **at least 20**, so a well-suited design can still live there.
   - Exact arithmetic.
   - If no level qualifies, stop and record a protocol amendment.
   - The rule writes `configs/part2/arm4.yaml`.

**Implementation note: adopted energy setting.** In `scripts/part2/b_redesign.py`, round 2 evaluates energy levels 30, 25, 20 and 15 on seeds 2872–2881. It selects the highest level with a median B-winner-minus-A-winner U2 gap of at least 5 blobs and a mean B-winner U2 score of at least 20. This replaces the first round's mean-gap criterion with a median-gap criterion. `configs/part2/arm4.yaml` records energy 25, round 2 and rule revision `e2619663f841c0b28cb60fefe127c28b2e01b0f7`; `RESULTS_part2.md` reports that setting for the final experiment. The associated result path is `results/part2/b_redesign/rule_round2.json`. This note describes the supplied implementation and selected configuration; the historical rule above is retained.

**Analysis.** As arm 3:
- its own B search, U2 runs and ORACLE_B (searched in arm 4's B);
- its own Holm family of 16 tests;
- the same rules and labels as the main arm (cap 25);
- the holds/changes check against the main arm.

Exploratory items run in arm 4's B too: ET_scratchB, the stage-1-only winners, the mutation-size reruns and the longer B.

**Seeds.** The same run seeds as the main arm.

**New pilot.** Before any final run, the pilot is repeated on new seeds 2910–2919 with arm 4 included. The 2900–2909 pilot stays reported. The same pilot decision rules apply to arm 4.

---

## 10. Seeds, number of runs, and pilot

**Seeds** (Part 2 range 2000–2999):

| Use | Seeds |
|---|---|
| Final runs | 2000–2099 (100 per group, every arm), or 2000–2199 if Section 18.2 picks 200 |
| Analysis-code check on made-up data (Section 18.3) | 2950 |
| Arm 3 map-size rule | 2800–2829; round 2: 2830–2859 (dated pre-final amendment) |
| Pilot | 2900–2909; second pilot with arm 4: 2910–2919 (Section 9.3) |
| Arm 4 energy rule: probe; rule; rule round 2 (Section 9.3; round 2: Section 17, 2 Nov) | 2870–2871; 2860–2869; 2872–2881 |
| Smoke tests | 2990–2998 |
| Bootstrap and permutation | 2999 |

Each run seed is split into named child streams. Within a run seed, no stream is used for two purposes:
- A search stage 1;
- A stage-2 sets (8, or 16 after the pilot rule);
- B search stage 1;
- B stage-2 sets 1–8;
- U2 sets 1–8;
- tie-breaks;
- deployment (food, behavior, starting spots; Section 2);
- SV designs;
- ET_cloneB's clone pick;
- A-checks (food, behavior, starting spots);
- the landscape check's B stage-1 searches, one set per suite variant (Section 18.4).

Some streams are shared **on purpose**, so that runs differ only in what is being tested:
- the suite variants, the mutation-size reruns, the longer B, ET_scratchB and the stage-1-only winners reuse the deployment streams of their run seed;
- the stage-1-only winners use the same tie-break keys as the confirmatory search, with no new draws.

A test lists every stream key the code uses and checks that each key serves only the purposes listed here.

Arms 2 and 3 use the same streams as the main arm. Arm 3's B search and U2 runs use the same child streams as the main arm's, in the larger map.

**Number of runs:** N per group per arm, with N = 100 or 200 by the rule in Section 18.2. Power is in Sections 6 and 18.2.

**Pilot.** It runs after adoption of the protocol amendment, on seeds 2900–2909, all groups. It covers:
- the main arm;
- arm 2's B reruns, plus the clone check from Section 9.1;
- arm 3, after its map-size rule.

It measures:
- wall time per seed for each arm;
- U1–U4;
- the SD of P1 in each group;
- the size of the tied sets;
- whether B separates any groups (per arm). In the pilot this is judged without tests, mirroring the final rule: in at least one **planned comparison**, both groups reached B in all 10 runs, both have a mean B-generation-0 population of at least 40, and they differ by at least 2.5 in mean P1 or by at least 3 of 10 runs in extinction.

The pilot may change **only** this:
- **If U3 fires in the pilot** (averaged over its 10 runs), stage 2 of the A search grows to the 64 best designs per list and 16 sets of streams. This happens once, for the A search only. The pilot then reruns its 10 seeds with the larger search, measures the time again, and judges U2, U3, U4 and the positive control again on the new winners. If U3 still fires in the final data, that is reported.

The number of runs is set by the rule in Section 18.2 (100 or 200), which uses the pilot's timing only.

**Pause final runs and record a protocol amendment** if any of these happens:
- U2 fires in an arm (5 or more of the 10 pilot runs);
- an arm's B separates no groups in the pilot;
- U4 fires for GS_pop_frozen;
- the projected time for the confirmatory final runs at N = 100 is over 24 hours on this machine. The projection is the measured process time per pilot seed for each confirmatory arm (after the U3 rule, if it applied), × 100, divided by the 2 CPUs, added up over the arms (Section 18.2);
- any simulation error.

Pilot runs are not used in the final analysis. They are reported in an appendix.

**Held-out (test) data for Part 2:** every B-phase outcome on the final seeds. Nothing is tuned on them. The pilot does see B outcomes on its own seeds, which is why it may change only the item above. The arm 3 map-size rule sees only ET, on its own seeds.

---

## 11. Exclusions and failures (fixed)

No run is excluded. Any simulator change after final runs begin must be documented as a dated protocol amendment. After such a change, **all** runs in every arm are rerun from one clean commit, and both the problem and the change are reported.

---

## 18. Dated additions (1 Nov 2026, before any Part 2 run)

Everything here was written before any Part 2 run.

### 18.1 Two kinds of result: confirmatory and exploratory

- **Confirmatory:** the tests in Sections 6–9 (main arm, arm 2, arm 3, and arm 4 from 2 Nov 2026). Only these give verdicts.
- **Exploratory:** everything in Sections 18.4–18.7. These are reported as effect sizes with 95% CIs. They are never part of a Holm family and never change a confirmatory verdict, and the report labels each one "exploratory". They do not carry the confirmatory tests' error control.

### 18.2 Number of runs: 100 or 200, chosen by time only

After the pilot, the number of confirmatory runs per group, N, is the larger of 100 and 200 whose projected confirmatory time is at most 24 hours on this machine.
- The projection uses the measured process time per pilot seed for the confirmatory arms (main arm, arm 2, arm 3, and arm 4 in the second pilot), divided by the 2 CPUs (Section 10).
- If even 100 is over 24 hours, stop and record a protocol amendment, as in Section 10.
- The exploratory work in Sections 18.4–18.7 always uses seeds 2000–2099, whatever N is, so it can't reduce N.
- The choice depends only on measured time. Pilot outcomes can change the time only through the U3 rule (Section 10), and pilot seeds are never final seeds.
- N is fixed once chosen and written into the pilot report before any final run.
- While final runs are going, progress checks report only how many seeds are done and any errors, never outcomes.
- Final seeds: 2000–2099 for 100 runs, 2000–2199 for 200.

**Why 200 is better** (computed before any run, exact Fisher at 0.05 ÷ 16):
- The chance of a "meaningfully different" P2 verdict rises from 0.94 to 0.985 at 0% vs 15%, and from 0.55 to 0.94 at 5% vs 20%.
- P2 can be "equivalent" with up to about 15% extinctions in both groups, instead of at most 4%.
- The CI for 0 vs 0 extinctions shrinks from ± 0.070 to ± 0.036 (Newcombe at 99.375%). An earlier draft of this line said ± 0.05; that used a wrong formula.

### 18.3 Testing the analysis code on made-up data, before the final runs

The full analysis code is written, tested and committed **before the pilot**. That covers computing outcomes from the raw logs, the tests, verdicts, labels, overall verdicts, positive control and holds/changes. It is then run on made-up datasets where the true answer is known. Numbers come from seed 2950, and no simulator is used.
- Checks that only use p-values (V1 and V3) may use 2,000 permutations per test to save time.
- Checks that use intervals (V2 and V4) use the real analysis's 20,000 resamples. With fewer, the 0.31% tails of a 99.375% interval rest on a handful of resamples.

| Check | Made-up data | Pass if |
|---|---|---|
| V0. Outcomes from raw logs | Small hand-made logs with hand-computed answers: P1 with a run that died in A; P2 when the last generation dies (survivors plus births = 0); S5; S6; ET_cloneB's pick; selection-gradient fitness | Every value matches the hand computation exactly |
| V1. False differences | 1,000 families of 16 tests with no true difference anywhere, at N = 100 and 200. Done three times: like ET in Part 1 (P1 mean 24.3, SD 0.4, capped at 25; P2 extinction 2%); with more spread (P1 mean 15, SD 5; extinction 20%); and with rare crashes (as the first, with 5% of runs at 0) | A verdict of "meaningfully different" or "different, size unclear" appears in at most 6.4% of families (5% plus 2 standard errors) |
| V1b. Holm with real effects mixed in | 1,000 families where 4 of the 16 comparisons have large true differences and 12 have none | The 12 null comparisons give a false difference in at most 6.4% of families |
| V2. False sameness | 1,000 datasets for each case where the true difference sits exactly on the threshold, at N = 100 and 200: P1 normal (SDs 0.4 and 5); P1 zero-inflated (as in Section 6); P1 with partial crashes; P2 exactly 10 points apart | "Equivalent" in at most 1% of each case |
| V3. Power | P2 at 0% vs 15%, and 5% vs 20%, with 100 and with 200 runs; 5,000 datasets each | The rate of "meaningfully different" is within 4 standard errors of the exact values in Section 18.2 |
| V4. Every verdict path | Built cases for each verdict and label: meaningfully different; equivalent; "equivalent, with a small real difference"; different, size unclear; inconclusive; mixed; both at the cap; one at the cap; both at the floor; deaths in A in one group; all deaths in A (the "arose in A" rule, including fewer than 30 seeds); constant identical groups; differences exactly on a threshold; positive control holding and failing; U2 fired; arm 2's 14-test family without ceiling labels | Each gives exactly the verdict and label the rules in Sections 6–9 say |
| V5. Holds/changes | The same comparison put into two different families | The fixed-level holds/changes check gives the same verdict both times |
| V6. Numbers in this document | The power tables, interval widths and P2 equivalence cases in Sections 6 and 18.2, recomputed with the analysis code's own functions | Each matches the document to the shown rounding; any mismatch is corrected as a dated change |

If a check fails because of a code bug, the bug is fixed and every check is run again. If it fails because a rule itself is wrong (for example, the studentized interval still fails V2), stop and record a protocol amendment. All results are reported.

### 18.4 Robustness suite: do the conclusions hold in other new worlds? (exploratory)

The confirmatory result is for one or two versions of B. To see how far it reaches, every group except ORACLE_B also runs in these B variants. Each variant starts from the same end-of-A populations and the same search winners on the final seeds, so A is not rerun.

| Variant | What changes from the main B |
|---|---|
| Map side 400, 500, 600, 800, 1,000 | Food is further away. Together with the main B (300) and arm 3, this gives a difficulty curve. Arm 3's side stays confirmatory |
| Food 15 | Less food (side 300) |
| Food 40 | More food (side 300) |
| Time limit 100 ticks | Shorter days (side 300) |
| Uniform food | Same 25 food, spread evenly instead of in clusters (side 300) |
| Blob-eating on at arm 3's side | Arms 2 and 3 combined |

**Data reuse.** Two variants are not run again, because their data already exist:
- arm 3's side is arm 3;
- for clone groups, "blob-eating on at arm 3's side" is the same as arm 3, by the clone check in Section 9.1.

**Thresholds and labels per variant.**
- The P1 threshold is 10% of the variant's food count: 1.5 blobs with 15 food, 4 with 40, and 2.5 otherwise.
- The ceiling cut-off is the food count minus that threshold.
- The blob-eating variant has no hard cap, so it gets no ceiling labels.
- The share of runs near the cap is reported per group.

**Reported for each variant:**
- P1, P2 and S6 per group.
- The 8 planned differences with 95% CIs, computed on run seeds where both groups reached B. All variants share the same A phase, so any loss in A would otherwise be copied into every variant.
- A summary for each comparison: the number of variants where the 95% CI excludes 0 in the same direction as the main arm, the number where it excludes 0 in the other direction, and the number where it reaches the variant's threshold. If the main arm's CI includes 0, only the counts by direction are given.

**Landscape check** (10 seeds, 2000–2009, every variant).
- The B search's stage 1 (generations 10–29, population score) is run for all 1,472 designs, and compared with the A search's stage-1 population score (generations 10–29).
- Reported:
  - Kendall's tau-b between A and B scores, over all designs and over designs with an A score above 0. Tau-b is used because many designs tie at 0 and at the cap.
  - The A winner's B rank, as a percentile, using average ranks for ties.
- This shows directly how much each variant calls for a different design.
- The pilot runs the landscape check for every variant on 2 pilot seeds (2900–2901) to measure its time.
- ORACLE_B is not run in the suite.

### 18.5 Sensitivity checks (exploratory)

- **Mutation size.** All mutation sizes × 0.5 and × 2, for every group that mutates (ET, ET_freezeB in A, ET_cloneB in A, GS_pop_mut), rerun from the start in the main B, arm 3's B and arm 4's B. The searches are unchanged, because they only use clones. Reported: the 8 differences with 95% CIs, next to the frozen setting.
- **Thresholds.** Verdicts are recomputed with half and double the thresholds (P1: 1.25 and 5 blobs; P2: 5 and 20 points). The report says how many verdicts change.
  - The label cut-offs move with the threshold: the ceiling is 25 minus the threshold, and the floor equals the threshold.
  - The U2, U3 and positive-control cut-offs stay as they are.
  - With N = 100, P2 can never be "equivalent" with a 5-point threshold, because the 0-vs-0 interval is ± 0.07. This is stated next to those results.
- **Longer B.** In the main arm, arm 3 and arm 4, every group's B phase continues to B generation 199 (ORACLE_B too). P1 and P2 still use generations 0–49. A test checks that generations 0–49 are bit-for-bit the same as in a 50-generation run. Reported: population and survival curves to generation 199, and P1 and P2 recomputed at generations 190–199.
- **What the second search stage bought.** The stage-1-only winners are also deployed as frozen clones, in the main arm, arm 3 and arm 4. They are the designs ranked first in the stage-1 ordering already used for the cut (score, then worst generation, then tie-break key), for each score. No new random draws are made. Reported: their P1 and P2 next to the two-stage winners'. This measures the fix for Part 1's search problem.

### 18.6 Extra reference group (exploratory)

**ET_scratchB:** 50 default blobs placed directly in B, with mutation, for 50 generations (main B, arm 3's B and arm 4's B; B-phase streams). Compared with ET, this shows whether training in A helps or hurts in B, compared with starting fresh. No test, like ORACLE_B.

### 18.7 Why groups succeed or fail (exploratory)

- **Selection gradients** (Lande and Arnold, 1983): which traits each world favors, and whether B favors different traits than A.
  - **Fitness.** A blob's fitness is the number of blobs it puts into the next generation: alive at the end of the generation × (1 + [food items + prey ≥ 2]). This is exactly what `world.reproduce` does.
    - It is computed from the per-blob logs (`b_alive_end`, `b_food_balls`, `b_prey`), not from parent links. Survivors keep their old parent ID, so counting parent links gives wrong numbers.
    - "Alive at the end" comes from `b_alive_end`, because some blobs die while holding 2 food.
    - A test checks that the summed fitness equals the next generation's population in every generation.
  - **Traits:** log speed, log size, log sense, and log(effective reach ÷ size). The last one is used instead of log effective reach, because effective reach is often just size ÷ 4, which makes it almost the same as size: a pre-final diagnostic found correlations above 0.95 in 9 of 30 Part 1 runs.
  - **Method:**
    - each trait is turned into a z-score within its generation;
    - relative fitness is fitness ÷ the generation's mean fitness, and generations with mean fitness 0 are skipped;
    - in each run, relative fitness is regressed on the 4 traits by ordinary least squares, pooled over the window's blob-generations;
    - a trait with no spread in the window is dropped;
    - a run gives a gradient only if it has at least 30 blob-generations, and the trait matrix has a condition number under 1,000.
  - **Windows:**
    - ET and GS_pop_mut: A generations 90–99 and B generations 0–9;
    - ET_freezeB: B generations 0–9 only (its A is ET's);
    - SV_frozen: A generations 0–9 (where its variety is), and B generations 0–9 if any variety is left.
  - **Reported:** the number of runs that give a gradient; the mean gradient over those runs, with a 95% bootstrap CI over runs.
- **First-generation deaths in B:** the share of blobs that die in B generation 0, and the traits of those that survive vs those that die, per group.
- **Energy use:** mean energy cost per move and ticks used per generation, per group.
- Trait means and variety over time, as in Section 5.

### 18.8 Code and analysis integrity

- **Tests**, all passing before the pilot:
  - Part 1 reproduces bit-for-bit;
  - same-seed groups share food layouts generation by generation;
  - per-phase settings apply only in their phase;
  - clone runs are identical with blob-eating on or off;
  - ET, ET_freezeB and ET_cloneB are identical through A;
  - the merged grid has 1,472 designs;
  - tie rules;
  - the longer B leaves generations 0–49 unchanged;
  - every random stream key is used for only one purpose.
- **Provenance.** Every results file stores the config, its hash, the git commit, package versions and hardware. One command reruns Part 2, and every script has a `--quick` mode (plan rules).
- **Run integrity and restart behavior:**
  - Final runs are launched from a separate clean worktree at the launch commit, checked in the same command as the launch.
  - Each seed's results are written to a temporary file, then renamed into place, then marked done with a small file holding the commit and config hash. A stop mid-write can't leave a file that looks finished.
  - On restart, the runner loads every finished file and checks its commit and config hash, and only then skips that seed. Any mismatch stops the runner.
  - Interrupted runs resume under the restart checks above, and interruptions are reported.
  - Times for Section 18.2 come from each seed's own process time, so that pauses don't count.
  - Every run, including failed ones, is reported.
- **Report.** Every result sentence is generated from the data (Part 1 practice). Confirmatory results come first; exploratory ones are labeled.

### 18.9 Extra predictions (written before any run; no part in the analysis)

| # | Prediction | Probability |
|---|---|---|
| 17 | In the difficulty curve, ET's lead over GS_pop_frozen on P1 grows with map size | 0.5 |
| 18 | The A–B landscape rank correlation is lower in arm 3's B than in the main B | 0.6 |
| 19 | ET_scratchB does worse than ET in the main B (training in A helps) | 0.7 |
| 20 | The stage-1-only population winner dies out in A more often than the two-stage winner | 0.6 |
| 21 | In B generations 0–9, ET's selection gradient on size has the opposite sign to A generations 90–99 | 0.3 |

### 18.10 Time

Section 18 adds, per run seed:
- the suite's B reruns: up to 10 variants × 7 groups × at most about 255,000 steps (up to 402,000 in the food-40 variant, 58% more), so at most about 20 million;
- the sensitivity reruns, including the stage-1-only winners: at most about 15 million;
- the longer B: at most about 20 million;
- ET_scratchB: about half a million.

That is at most about 1 minute per seed, including the start-up cost of about 100 extra short runs.

The landscape check is 100 B stage-1 searches (10 variants × 10 seeds): at most about 23 billion steps (about 2.5 hours). It is likely about 3 billion steps, going by the main B's typical cost. That could take 20–40 minutes, because short runs in Part 1 ran up to 2× slower per step.

The pilot measures all of this, and Section 18.2 uses the measured times.

