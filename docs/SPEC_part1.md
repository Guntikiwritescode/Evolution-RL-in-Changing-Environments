# Trait simulator rules and calibration

This specification describes the calibrated Python reconstruction used in Study 1. The final settings are in [`configs/part1/final.yaml`](../configs/part1/final.yaml), which extends [`base.yaml`](../configs/part1/base.yaml); selected metric definitions are in [`metric_defs.yaml`](../configs/part1/metric_defs.yaml). Section numbers and rule identifiers are retained for code references.

**Sources and tags.** ER denotes the source manuscript, Abhinav R. Guntaka's *Evolving Intelligence: What Biology Can Teach AI About Getting Smarter*, referred to in the historical records as the 2025 paper. S1 denotes Jasper Palfree's MinuteLabs.io Evolution Simulator, pinned at commit `55af956`; the Rust reference wrapper is in `reference/s1ref/`. `[PAPER]` identifies a paper statement, `[SOURCE]` an implementation rule, `[AUTHOR]` a study-design decision, and `[GUESS]` a reconstruction assumption. Source paths in the rule tables refer to S1. Reconstruction choices are distinguished from S1 defaults below.

## 3. The world

| # | Rule or setting | Value | Tag and evidence |
|---|---|---|---|
| W1 | Stage shape | A square from (0,0) to (L,L). Blobs can't leave it. After a move, x and y are each clamped into [0, L] separately. So a blob pushing against a wall slides along it rather than stopping where its path hit the wall. | [SOURCE] `stage/mod.rs:24-42, 65-70`; `simulation/behaviours/move.rs:9-13` |
| W2 | Environment A size | L = 500 | [PAPER] ER p. 3; [SOURCE] default `src/store/simulation.js:41` |
| W3 | Environment A food | 50 food items each generation, each at a uniformly random spot on the stage | [PAPER] ER p. 3; [SOURCE] `src/store/simulation.js:39`, `simulation/mod.rs:133-139`, `stage/mod.rs:46-51` |
| W4 | Food timing | All food is placed fresh at the start of each generation. Eaten food does not grow back during the generation. Leftover food is thrown away. | [SOURCE] `simulation/generation.rs:23-26`, `simulation/food.rs`, `simulation/mod.rs:96-113` |
| W5 | Environment B size | L = 300 | [PAPER] ER p. 3 |
| W6 | Environment B food count | 25 food items in 5 clusters | [PAPER] ER p. 3 |
| W7 | Food per cluster | 5 (25 split evenly) | [GUESS] The paper does not specify the allocation. |
| W8 | Cluster centers | Uniformly random inside the stage, staying at least 2σ_c from the edges. Drawn fresh every generation (like W4). | [GUESS] The paper does not specify cluster timing. Fixed centers were evaluated as a sensitivity check. |
| W9 | Food spread in a cluster | Each food item = cluster center + random offset, normal in x and y with standard deviation σ_c, then clipped to the stage | [GUESS] The paper does not specify the distribution. |
| W10 | σ_c | 15 (5% of the stage width); sensitivity checks use 7.5 and 30. | [GUESS] Fixed before final evaluation; B outcomes were excluded from calibration. |

---

## 4. A blob

### 4.1 Traits (the "genes")

| Trait | Start value (ET) | Selected mutation SD | Lowest allowed | Tag and evidence |
|---|---|---|---|---|
| speed | 10.0 | 1.0 | smallest positive normal float64 (about 2.2 × 10⁻³⁰⁸) | Start and bound: [SOURCE] `src/store/simulation.js:19`, `creature/mutatable.rs:43-46`; start: [PAPER] ER p. 5; SD: [GUESS], calibrated |
| size | 10.0 | 1.0 | smallest positive normal float64 | Start and bound: [SOURCE] `src/store/simulation.js:20`, `creature/mutatable.rs`; start: [PAPER] ER p. 5; SD: [GUESS], calibrated |
| sense range | 20.0 | 1.0 | 0 | Start and bound: [SOURCE] `src/store/simulation.js:21`, `creature/mutatable.rs:29-32`; start: [PAPER] ER p. 5; SD: [GUESS], calibrated |
| reach | 1.0 | 0.25 | smallest positive normal float64 | Start: [SOURCE] `src/store/simulation.js:22`, [PAPER] Fig. 3; SD: [GUESS], calibrated |
| flee distance | 10¹² | 0 | 0 | [SOURCE] `src/store/simulation.js:23`; effectively covers every visible predator |

S1's mutation SD is 0.5 for speed, size and sense, and 0 for reach. The reconstruction calibrated these values because the paper reports changing traits, including reach. S1's lifespan setting is 10⁴, but its old-age rule is disabled (`api/mod.rs:26-36`); the reconstruction likewise has no old-age mortality.

Mutation size denotes standard deviation: S1's interface labels it “σ²,” but `Normal::new` receives a standard deviation (`src/components/creature-template-config.vue:46-48`; `creature/mutatable.rs:5-8`). Mutation sizes remain constant across generations. Calibration candidates and the selected values are documented in Section 12.

### 4.2 Numbers worked out from the traits

| # | Quantity | Formula | Tag |
|---|---|---|---|
| B1 | Distance moved per tick ("effective speed") | speed × size / 10 | [SOURCE] `creature/mod.rs:171` |
| B2 | Reach used in every reach check (food, prey, and reaching home to fall asleep) | the larger of (reach, size / 4) | [SOURCE] `creature/mod.rs:175-177, 307, 321` |
| B3 | Can see a point | distance ≤ sense range | [SOURCE] `creature/mod.rs:296-298` |
| B4 | Can reach a point | Yes if the point is within reach of where the blob is now. Also yes if the point is within reach of the straight path it just walked (last position → now), **but only if the point sits "beside" that path**: drop a line from the point straight onto the path, and that line must land on the path itself, not past either end. So a point near the *last* position but behind it does not count. This is not the usual "distance to a line segment". | [SOURCE] `creature/mod.rs:300-309, 320-322`, `math/mod.rs:13-39` (the "outside the segment" check is lines 33-36) |
| B5 | Energy cost per tick | 0.0001 × (size³ × effective speed² + sense range) | [SOURCE] `creature/mod.rs:10, 214-218` (comment says this is Primer's formula) |
| B6 | Energy at the start of every generation | 500 (full refill each generation) | [SOURCE] `src/store/simulation.js:25`, `creature/mod.rs:125-168` |

Example: a default blob (speed 10, size 10, sense 20) moves 10 units a tick, pays about 10.0 energy a tick, and so has about 50 ticks of energy. A blob with the paper's generation-100 averages (speed 26.6, size 6.9, sense 31.2) moves about 18.4 a tick, pays about 11.1, and has about 45 ticks.

The movement term scales as size⁵ at a fixed speed trait: size³ × (speed × size / 10)². Thus changing size affects both distance traveled and energy use.

---

## 5. One generation, tick by tick

### 5.1 Start of a generation
| # | Rule | Tag |
|---|---|---|
| G1 | Very first generation: each blob is placed at a random point on the stage edge (pick a random spot, then move it to the nearest edge). | [SOURCE] `api/square_world.rs:31-33`, `stage/mod.rs:53-63` |
| G2 | Later generations: every blob starts at its "home" spot. That is the edge point that was its home at the end of the last generation (see Q6). A newborn starts at its parent's home spot. | [SOURCE] `creature/mod.rs:125-168` |
| G3 | Every blob starts awake, with 0 food eaten and full energy. | [SOURCE] `creature/mod.rs:86-110, 125-168` |
| G4 | Food is placed (W3 or W8–W10). | [SOURCE] / [GUESS] |
| G5 | A blob with an effective speed of exactly 0 dies. | [SOURCE] `simulation/behaviours/starve.rs:16-23` |

### 5.2 Each tick, in this order
The reconstruction repeats ticks until no blob is awake or 200 ticks have completed, then applies the end-of-generation rules. S1 has no ordinary tick cap: it runs until every blob is asleep or dead and raises an error at tick 1,000,000 (`simulation/generation.rs:7, 31, 62-67, 85`). Section 5.6 describes the calibrated cap.

1. **Shuffle** the list of blobs into a random order. Every later step goes through the list in this order. [SOURCE] `generation.rs:79-82, 88`
2. **PRE:** clear each living blob's goal. Set each living blob's "home" to the nearest point on the nearest stage edge. [SOURCE] `simulation/mod.rs:24-41`; `simulation/behaviours/edge_home.rs:11-39`
3. **ORIENT (choose a goal):** four rules run in this order, **for awake blobs only**. Each one may offer a goal. A blob keeps the goal with the highest urgency (5.3). [SOURCE] order from `api/mod.rs:26-36`; awake-only filters at `wander.rs:21`, `cannibalism.rs:12-23`, `scavenge.rs:8, 63`, `satisfied.rs:28`
   - a. **Wander.** Turn by a random angle between −45° and +45° from the current heading. If a point 1 unit ahead in that direction is inside the stage, aim there; otherwise aim at the stage center. Urgency: "mild want" (the lowest). [SOURCE] `simulation/behaviours/wander.rs:16-33`. The "current heading" is the direction of the last move, or the +x direction if the blob hasn't moved yet (`creature/mod.rs:220-239`).
   - b. **Predators and prey.** See 5.4. [SOURCE] `simulation/behaviours/cannibalism.rs:35-81`
   - c. **Look for food.** Only blobs with fewer than 2 food do this. Find the single nearest uneaten food. If the blob can see it, aim at it. Urgency: "vital want" if it has 0 food, "medium want" if it has 1. [SOURCE] `simulation/behaviours/scavenge.rs:7-29, 57-65`
   - d. **Go home.**
     - 0 food: no home goal (keep searching until food or energy runs out).
     - 2 or more food: aim home, "strong want".
     - 1 food: work out a "homesick number" = (energy left ÷ cost per tick) − (distance home ÷ effective speed). Above 10: no home goal. Above 5: "mild want". Above 0: "strong want". 0 or below: "vital want".
     - If the blob got a home goal and can reach home (B4), it **falls asleep** and stops for this generation.
     [SOURCE] `simulation/behaviours/satisfied.rs:7-41`, `simulation/behaviours/homesick.rs:7-27`
4. **MOVE:** every awake blob moves one effective-speed step straight toward its goal (or straight away from it, if the goal is "run away"), then is clamped back inside the stage (W1). If the goal is exactly where the blob already is, it keeps its last heading (or +x). It pays the full energy cost (B5) even if the clamp shortened the move. If its energy left is 0 or below, it **dies on the spot**. [SOURCE] `simulation/behaviours/move.rs:7-24`, `creature/mod.rs:207-239, 328-334`
5. **ACT (eat):**
   - a. **Predators eat prey** (5.4). [SOURCE] `cannibalism.rs:84-93`
   - b. **Food.** Going through the shuffled list: each awake blob with fewer than 2 food checks only its single nearest uneaten food. If it can reach it (B4), it eats it and the food disappears. [SOURCE] `scavenge.rs:31-39, 67-82`
6. **POST:** if no uneaten food is left anywhere, every awake blob that has eaten nothing dies now. [SOURCE] `simulation/behaviours/starve.rs:25-35`

### 5.3 Goal urgency (lowest to highest)
"mild want" < "mild avoid" < "medium want" < "medium avoid" < "strong want" < "strong avoid" < "vital want" < "vital avoid".
If two goals have the same urgency, the closer one wins. If they are the same distance, the first one stays. [SOURCE] `creature/mod.rs:27-41, 241-262`
"Running away from a predator" is "vital avoid", the highest of all.

**Goal precedence.** Wander always offers a "mild want" goal at a point only 1 unit ahead. Any other "mild want" goal is almost always farther away, so it loses the tie. In practice this means:
- a blob with 1 food and a homesick number between 5 and 10 does **not** head home; it keeps wandering (though it still falls asleep if home happens to be within reach);
- a predator that already has 2+ food does **not** chase prey (though it still eats any prey it happens to reach, C4).
The exception is when wander's point is outside the stage and wander falls back to the stage center, which may be farther away than the other goal. [SOURCE] `wander.rs:25-29`, `creature/mod.rs:250-257`. See Q10.

### 5.4 Predators and prey (S1 calls this "cannibalism")
| # | Rule | Tag |
|---|---|---|
| C1 | A blob can eat another blob if the other's size is **at most 0.8 ×** its own size. Both must be awake. | [SOURCE] `api/mod.rs:30`, `cannibalism.rs:16-24` |
| C2 | ORIENT: a prey blob that can see a possible predator (within its flee distance, which in practice means "within sight") runs away from a slightly randomized copy of the predator's position (random turn of −45° to +45°). | [SOURCE] `cannibalism.rs:40-52`, `creature/mod.rs:311-318` |
| C3 | ORIENT: a predator that can see prey offers a goal to chase it. Going through pairs in list order, it only offers a new chase goal if that prey is slower (lower effective speed, B1) than the last one it offered. Urgency depends on food eaten: 0 → "vital want", 1 → "medium want", 2+ → "mild want". These goals compete with all others by the rules in 5.3, so between equally urgent goals the closer one wins. | [SOURCE] `cannibalism.rs:54-79`, `creature/mod.rs:241-262` |
| C4 | ACT: if a predator can reach prey (B4), it eats it. The prey dies. **The meal counts as one food item for the predator.** There is **no fullness check**, so a predator can end up with more than 2 food. There is also no limit per tick: a predator can eat several prey in the same tick. | [SOURCE] `cannibalism.rs:84-93`, `creature/mod.rs:272-274` |
| C5 | Identical clones (all GS runs, and ET at the very start) can never eat each other, because their sizes are equal. | Follows from C1 |
| C6 | Predation is disabled in the selected reconstruction. S1 enables it by default, and the paper does not mention it. The implementation retains the switch `world.cannibalism`. | [GUESS], selected during calibration; S1 default: [SOURCE] `api/mod.rs:26-36, 52-55` |

### 5.5 End of a generation
| # | Rule | Tag |
|---|---|---|
| E1 | Every living blob that ate 0 food dies. | [SOURCE] `starve.rs:37-43` |
| E2 | A blob that ran out of energy is dead, **even if it had already eaten 2 food**. It does not survive or reproduce. | [SOURCE] `creature/mod.rs:328-334`, `simulation/behaviours/reproduction.rs:24-34` |

### 5.6 Generation duration

| # | Rule | Tag |
|---|---|---|
| L1 | S1 has no ordinary tick cap; a generation ends when all blobs are asleep or dead, subject to its error limit. | [SOURCE] `simulation/generation.rs:62-67, 85` |
| L2 | The reconstruction uses **200 ticks per generation** in A and B. At the cap, living blobs with 0 food die, those with 1 food survive, and those with 2+ food survive and produce one offspring. | [GUESS], selected by ET training calibration; `configs/part1/final.yaml`, `src/evosim/kernel.py` |

Generation length matters because an agent with one food item can keep wandering until its energy runs low. Calibration compared S1's uncapped rule with caps of 50, 75, 100, 150 and 200 ticks, then refined the best setting (Section 12). The selected cap is a reconstruction parameter; the source paper does not specify one.

---

## 6. Between generations

| # | Rule | Tag |
|---|---|---|
| R1 | Dead blobs are removed. | [SOURCE] `reproduction.rs:25-27` |
| R2 | Every living blob survives into the next generation, unchanged except for age + 1. | [SOURCE] `reproduction.rs:30-31`, `creature/mod.rs:141-168` |
| R3 | Every living blob with **2 or more** food (food items plus prey eaten) also makes **exactly one** offspring. | [SOURCE] `reproduction.rs:10-20` |
| R4 | The offspring's traits = the parent's traits, each with independent normal noise added (Section 4.1 sizes), then clipped to the lowest allowed value. | [SOURCE] `creature/mod.rs:125-138`, `creature/mutatable.rs` |
| R5 | The offspring starts at the parent's home spot. | [SOURCE] `creature/mod.rs:136` |
| R6 | So: 0 food = die, 1 food = survive, 2+ food = survive and make one offspring. No blob makes more than one offspring per generation. | [SOURCE] `reproduction.rs:10-34` |
| R7 | A run ends early if no blob is alive. An extinct run stays extinct. | [SOURCE] `simulation/mod.rs:96-113` |

---

## 7. The two conditions

### 7.1 ET (evolutionary training)
| # | Setting | Value | Tag |
|---|---|---|---|
| ET1 | Starting group | 50 identical default blobs (Section 4.1) | [PAPER] ER p. 3; [SOURCE] defaults |
| ET2 | Training | 100 generations in Environment A, numbered 0–99 | [PAPER] ER p. 3, Fig. 1 x-axis |
| ET3 | Mutation | SD 1.0 for speed, size and sense; 0.25 for reach (Section 4.1). | [SOURCE] + [GUESS] |
| ET4 | Selection | Only the survival and reproduction rules above. No score is used. | [PAPER] ER p. 3; [SOURCE] |

### 7.2 GS (grid search)
| # | Setting | Value | Tag and evidence |
|---|---|---|---|
| GS1 | Grid size | 8 × 8 × 8 = 512 designs | [PAPER] ER p. 3 |
| GS2 | Which 3 traits | speed, size, sense range. Reach fixed at 1.0. | [GUESS] These three are the traits S1's settings screen lets you change (`src/components/creature-template-config.vue:30-32`). The paper lists 4 traits for ET, so **ET evolves one trait (reach) that GS can't search**. The later trait experiment searches all four traits. |
| GS3 | Grid values | 8 points spaced evenly on a log scale, from 0.25× to 4× each default. Speed and size: 2.5, 3.71, 5.52, 8.20, 12.19, 18.11, 26.92, 40.0. Sense: 5.0, 7.43, 11.04, 16.41, 24.38, 36.23, 53.84, 80.0. | [GUESS] The paper does not specify the spacing or range. These settings were fixed before final evaluation. |
| GS4 | How a design is scored | Mean food per blob (Section 8, M1) over 30 generations as 50 clones in A; extinct generations score 0. Designs share the search seed; exact ties go to the first design in grid order. | [GUESS] The paper says the search found the "best-performing configuration" and that GS beat ET on "raw fitness" (ER p. 3–4). Grid-search scoring uses food per agent. |
| GS5 | How long each design is run, and whether there is one search or one per seed | Each design is evaluated for 30 generations in A; one search, seeded with 999, supplies the winner shared by all final runs. | `configs/part1/base.yaml` (`gs.eval_generations`, `gs.search`, `gs.search_seed`), inherited by `configs/part1/final.yaml`; `src/evosim/gridsearch.py`. |
| GS6 | Deployment | The winning design as 50 identical clones, mutation size 0 for every trait, 100 generations in A, then transfer. Clones still live, die and reproduce by the same rules, so their number changes. | [PAPER] ER p. 3, Fig. 2 |
| GS7 | Starting spots | Random edge points, as in G1 | [SOURCE] |

### 7.3 Transfer to Environment B
| # | Setting | Value | Tag |
|---|---|---|---|
| T1 | What moves to B | The whole population as it would start generation 100 (after the last A reproduction step), traits unchanged | [PAPER] ER p. 3; Fig. 7 starts near 49 for ET and 37 for GS |
| T2 | Length | 50 generations, numbered 0–49 | [PAPER] Figs. 6–7 x-axes |
| T3 | Where blobs start in B | Fresh random edge points of the 300×300 stage. (Their old home spots are on the edge of the 500 stage, which is outside B.) | [GUESS] S1 has no rule for this. |
| T4 | ET keeps mutating in B; GS clones keep mutation 0 | as in A | [PAPER] Fig. 3, ER p. 3 |

---

## 8. Metrics

Metrics are computed per run, then summarized across 30 runs as mean ± standard deviation. The final comparison uses the selected definitions in [`metric_defs.yaml`](../configs/part1/metric_defs.yaml). Calibration first scored simulation settings with fixed primary definitions, then selected scalar metric definitions in a second stage; both sets are described below. A generation with no blobs has population 0 and undefined food-per-blob, survival and trait metrics; averages of those metrics omit undefined generations and report contributing runs.

| # | Metric | Definition used in the final comparison | Tag and evidence |
|---|---|---|---|
| M0 | Population | Number of blobs that start the generation | [SOURCE] `api/mod.rs:130-176` |
| M1 | Fitness (food per blob) | (Food items eaten + prey eaten) ÷ M0 | [GUESS] S1 records both food sources (`creature/mod.rs:272-274`). Predation is disabled in the final reconstruction, so prey contributes 0. |
| M2 | Median fitness | Median per-blob food count in training generation 99 | [PAPER] Fig. 11; definition [GUESS] |
| M3 | Final training fitness | Mean M1 over training generations 90–99 | [GUESS] Selected in calibration stage 2; the primary calibration definition used generation 99 alone. |
| M4 | Survival rate | Living blobs at generation end ÷ M0, averaged over training generations 90–99 | [GUESS] Selected in stage 2; the primary definition averaged generations 0–99. |
| M5 | Transfer fitness, “gen 1–10” | Mean M1 over B generations 0–9 | [GUESS] The first ten plotted transfer generations in Fig. 6 |
| M6 | Transfer fitness, “last 10” | Mean M1 over B generations 40–49 with population above 0; undefined if none contribute | [GUESS] The source table repeats the first-ten-generation values in this row. |
| M7 | Generalization gap | M3 − M5, using the same final-training window as M3 | [PAPER] ER p. 9; the additional first-five-transfer-generation version is reported separately. |
| M8 | Transfer trend slope | Least-squares slope of M1 against B generation number, 0–49, using generations with population above 0 | [GUESS] The source text and table assign the reported slope 0.02 to different groups; the reconstruction computes its own value. |
| M9 | Trait diversity (CV²) | Sum of (sample SD ÷ mean)² over speed, size, sense and reach, averaged over training generations 0–99 | [GUESS] Selected in stage 2; primary calibration used the same four-trait sum at generation 99. Per-trait values are also reported. |
| M10 | Shannon diversity (H) | −Σ p ln p, where p is the fraction of blobs with an identical four-trait vector; averaged over training generations 0–99 | [GUESS] Exact types and training-mean aggregation were selected in stage 2; primary calibration used exact types at generation 99. |
| M11 | Energy efficiency | Relative decrease in mean energy cost per unit distance from generation 0 to 99: (u₀ − u₉₉) ÷ u₀ | [GUESS] Candidate E2 selected in stage 2. The source paper gives no definition. |
| M12 | Traits over time | Mean trait value across blobs that start each generation | [PAPER] Fig. 3; definition [GUESS] |
| M13 | Population curves | Mean across all runs with extinction carried forward as 0; surviving-run means are also computed separately | [GUESS] The distinction matters when runs go extinct. |
| M14 | Recovery time | Descriptive summary only | [AUTHOR] |

For M11, each blob's cost per unit distance is its per-tick energy cost divided by effective speed, and u is the mean across blobs. The five calibration candidates were: E1, relative decrease in mean per-tick cost from generation 0 to 99; E2, the same decrease in cost per unit distance; E3, the mean relative decrease in cost per unit distance over all 100 training generations; E4, the coefficient of variation of per-tick cost at generation 99; and E5, the coefficient of variation of cost per unit distance at generation 99. These definitions give 0 for unchanged clones.

Selecting metric definitions to match ET training targets contributes to the calibration match. Those definitions are reconstruction choices, not recovered definitions from the source paper. [`src/evosim/metrics.py`](../src/evosim/metrics.py) computes all candidates.

---

## 9. Statistics

| # | Rule | Tag |
|---|---|---|
| S1 | Welch's t-test, two-tailed, 30 vs 30 runs | [PAPER] ER p. 3, Fig. 11 |
| S2 | Degrees of freedom: Welch–Satterthwaite formula | [PAPER] Fig. 11 df column is consistent with this formula within the rounding of the printed SDs |
| S3 | Cohen's d = (mean_ET − mean_GS) ÷ √((SD_ET² + SD_GS²) ÷ 2). With 30 per group this equals t × √(2/30). | [PAPER] Fig. 11 |
| S4 | Bonferroni correction for 10 comparisons (multiply p by 10, cap at 1) | [PAPER] ER p. 3. The table has 9 rows, and two are identical, so only 8 distinct comparisons are shown. Two of the 10 are unaccounted for (one, if the duplicate row stands for a real comparison). The correction uses 10 as stated. |
| S5 | Report **both** the uncorrected p (what the table's p column actually shows) and the corrected p (what its stars show). | [PAPER] Fig. 11 |
| S6 | Tests on values that are 0 by design (GS diversity, GS CV²) are computed for comparison but marked "not meaningful". When one group has 0 spread, Welch's df becomes 29, as in the table. | [AUTHOR] |

---

## 10. Simulation-step accounting

| # | Rule | Tag |
|---|---|---|
| K1 | Count one creature-step each time one awake blob moves one tick. Totals include all 30 runs and both phases; the GS total additionally includes its single shared search. | [GUESS] `src/evosim/runner.py`, `scripts/part1/run_part1.py` |
| K2 | Also record ticks per generation and wall-clock time. | [AUTHOR] |

The paper reports approximately 7.9 million ET steps and 33.5 million GS steps (ER p. 11) without defining the counting unit or search horizon. These totals informed the reconstruction's creature-step definition and shared 30-generation search. GS performance and transfer outcomes were excluded from calibration; the reported compute totals therefore provide context rather than an independent validation of those choices.

---

## 12. Calibration

Calibration used ET in Environment A on seeds 1000–1029. The target definitions and search procedure are in `calibration/targets.json`, generated by [`scripts/part1/build_targets.py`](../scripts/part1/build_targets.py). Every evaluated setting is recorded in `calibration/log.csv`; `calibration/selected.json` records the selected setting and metric definitions.

### 12.1 Search and selection

| Setting | Stage-1a candidates | Selected |
|---|---|---|
| Shared mutation SD for speed, size and sense | 0.5, 0.75, 1.0, 1.25, 1.5, 2.0 | 1.0 |
| Reach mutation SD | 0.1, 0.25, 0.5, or the shared trait SD | 0.25 |
| Tick cap | No cap, 50, 75, 100, 150, 200 | 200 |
| Predation | Off, on | Off |

Stage 1a comprised 288 combinations. Stage 1b evaluated 25 combinations around the best setting, with offsets of −0.125, −0.0625, 0, +0.0625 and +0.125 in trait SD, and −25, −12, 0, +12 and +25 in tick cap; reach SD and predation remained fixed. An uncapped winner would remain uncapped. Candidates below SD 0.25 or cap 25 were excluded.

For each target, error is (simulated mean − paper value) ÷ tolerance. The score is the mean squared error within each of five groups—fitness, population, survival, traits and diversity—then averaged equally across groups. The lowest score across both stages selects the simulation setting; exact ties go to the first logged setting. A target passes when its absolute error is at most one tolerance unit. The prespecified criterion required all targets to pass; the selected setting did not meet it.

Stage 2 held the simulation fixed and selected, for each scalar metric, the candidate mean closest to its paper target. Candidates included final-generation or last-ten-generation fitness and median; all-training, last-ten or final-generation survival; three or four traits, population or sample SD, and final-generation or training-mean CV²; three- or four-trait Shannon types, exact or rounded to one decimal or whole numbers, and final-generation or training-mean aggregation; and E1–E5 for energy efficiency. The selected definitions are in Section 8.

### 12.2 Settings fixed before final evaluation

GS grid traits, spacing, range, scoring and search horizon (GS2–GS5), together with the transfer layout and placement rules (W7–W10, T3), were not tuned to GS performance or transfer outcomes. Transfer sensitivity checks used cluster SD 7.5 or 30 instead of 15, and fixed cluster centers instead of redrawing them each generation.

### 12.3 Calibration targets (ET, Environment A only)

- Fitness curve: about 0.87 at generation 0, peak about 1.12 at generations 4–5, back to about 1.03 by generation 12, then flat (Fig. 1 chart). Final 1.03 ± 0.03, median 1.00 ± 0.00 (Fig. 11).
- Population curve: 50 at generation 0 → about 41.5 at 1 → dip to about 37 at generations 3–4 → about 47 by generation 10 and 48.5 by 14 (Fig. 2 chart) → about 48 at generation 100 (ER p. 4).
- Survival rate: 66.5% (ER p. 4).
- Traits at generation 100: speed 26.6, size 6.9, sense 31.2 (ER p. 5); reach about 2 (Fig. 3 chart).
- Diversity: H 3.77 ± 0.06, about 3.8 by generation 10 (Fig. 5 chart); CV² 0.41 ± 0.29 (Fig. 11).

### 12.3a Calibration outcome (31 October 2026)

Per-target calibration results are in `calibration/runs/sd1_reach0.25_cap200_cann0__1a.json`; selected scalar definitions and candidate means are in `calibration/selected.json`.

- Selected by the written rule: **trait mutation 1.0, reach mutation 0.25, time limit 200 ticks, blob-eating off** (`configs/part1/final.yaml`). 70 of 76 targets pass with the primary definitions.
- Misses: reach rises too slowly (1.78 vs 2.12 at generation 99; also generations 60 and 80); CV² at generations 10 and 40 a bit low; final Shannon on the primary definition (3.88 vs 3.77 ± 0.06), which passes on the stage-2 definition (training mean, 3.75).
- **Sensitivity to calibration seeds.** The selected setting scores 0.294 on calibration seeds and 0.505 and 0.583 on two additional seed sets; nearby settings change rank (`calibration/score_noise_check.json`). The check compared trait SDs 0.875–1.0625 and caps of 175–225 ticks or no cap, with reach SD 0.25 and predation off throughout. These checks did not change the selected setting. In the original calibration search, the best predation-on setting passed 47 of 76 targets, compared with 70 for the selected setting.
- Stage-2 definitions: final fitness = mean of the last 10 training generations; survival = last 10; CV² = 4 traits, sample SD, training mean; Shannon = exact types, training mean; energy efficiency = E2 (drop in energy cost per unit distance), 0.28 ± 0.09 vs the paper's 0.20 ± 0.18.
- **Diversity contributions:** under the selected sample-SD definition, training-mean CV² is 0.058 for speed, size and sense together and 0.396 for all four traits. Reach contributes 0.338 in the reconstructed calibration trajectories.

### 12.4 Held-out evaluation

Final runs used seeds 0–29. The ET training recheck passed 67 of 76 primary calibration targets (`results/part1/final/comparison.json`, `targets_recheck`), compared with 70 on calibration seeds. GS performance and every transfer-phase outcome were held out from calibration. The compute-accounting assumptions are described in Section 10.

### 12.5 Seeds

Calibration used seeds 1000–1029; final ET and GS runs used 0–29; the shared GS search used 999. The score-noise check used additional sets 17000–17029 and 17030–17059 without changing the selection. Reference checks default to 9000–9199, and the quick run uses 9900–9901. ET and GS receive separate condition streams from each run seed; each lineage then splits streams for initial placement, A, transfer placement and B using NumPy SeedSequence.

---

## 13. Inherited implementation details

The reconstruction retains the following S1 behaviors. Predation behaviors are implemented but disabled in the selected configuration. The births statistic and zero-length-movement handling differ as noted below.

| # | Quirk | Where | Default |
|---|---|---|---|
| Q1 | The predator–prey pairing loop skips some pairs. When the blob at list position i is not awake, it skips pairing the blob at position i−1 with everyone after it. | `cannibalism.rs:11-12` | Retained |
| Q2 | A blob only checks its single nearest food when eating. It can pass within reach of a farther food item and not eat it. | `scavenge.rs:31-39` | Retained |
| Q3 | Energy is paid in MOVE, before eating in ACT. A blob that hits 0 energy on its last move dies before it can eat that tick. | `generation.rs:91-95` | Retained |
| Q4 | A blob that runs out of energy dies even after eating 2 food (E2). | `creature/mod.rs:328-334` | Retained |
| Q5 | Predators keep eating even when full, so a blob can have more than 2 food (C4). | `cannibalism.rs:84-93` | Retained |
| Q6 | "Home" moves every tick to the nearest edge from where the blob is now. | `edge_home.rs:32-39` | Retained |
| Q7 | Every blob's first heading each generation is +x, plus the random turn. | `creature/mod.rs:231-236` | Retained |
| Q8 | S1's "births" statistic also counts dead blobs that ate 2 food. | `api/mod.rs:157-159` | Not copied: births = living blobs with 2+ food (R3) |
| Q9 | Movement cost is the same for a full move and a move cut short at the edge. | `move.rs:7-14` | Retained |
| Q10 | "Mild want" goals (going home with a homesick number between 5 and 10; chasing prey when already full) almost always lose to wander's goal 1 unit ahead, so they rarely steer the blob (Section 5.3). | `wander.rs:25-29`, `creature/mod.rs:250-257` | Retained |
| Q11 | A blob pushing into a corner can make a move of zero length. Its next heading is then undefined (a division by zero gives "not a number"). Wander's "1 unit ahead" point then fails the inside-the-stage check, so the blob aims at the stage center. | `creature/mod.rs:231-238`, `wander.rs:25-29` | Aim at the center without relying on NaN arithmetic |

S1 uses Rust SmallRng; the reconstruction uses NumPy generators. Reference comparisons therefore test distributions and rule behavior, rather than identical random trajectories.
