# Trait simulator rules and calibration

Scientific excerpts from the dated study protocol. Operational checklists, private discussion and task lists are omitted. Original section numbers and decision rules are retained. These are edited archival excerpts, not a new registration. Original document SHA-256: `3858484a1b2ddbefd10f665c68b57f477bf63e0268ff1ce05999fe1bb25f51ee`.

## 3. The world

| # | Rule or setting | Value | Tag and evidence |
|---|---|---|---|
| W1 | Stage shape | A square from (0,0) to (L,L). Blobs can't leave it. After a move, x and y are each clamped into [0, L] separately. So a blob pushing against a wall slides along it rather than stopping where its path hit the wall. | [SOURCE] `stage/mod.rs:24-42, 65-70`; `simulation/behaviours/move.rs:9-13` |
| W2 | Environment A size | L = 500 | [PAPER] ER p. 3; [SOURCE] default `src/store/simulation.js:41` |
| W3 | Environment A food | 50 food items each generation, each at a uniformly random spot on the stage | [PAPER] ER p. 3; [SOURCE] `src/store/simulation.js:39`, `simulation/mod.rs:133-139`, `stage/mod.rs:46-51` |
| W4 | Food timing | All food is placed fresh at the start of each generation. Eaten food does not grow back during the generation. Leftover food is thrown away. | [SOURCE] `simulation/generation.rs:23-26`, `simulation/food.rs`, `simulation/mod.rs:96-113` |
| W5 | Environment B size | L = 300 | [PAPER] ER p. 3 |
| W6 | Environment B food count | 25 food items in 5 clusters | [PAPER] ER p. 3 |
| W7 | Food per cluster | 5 (25 split evenly) | [GUESS] Simplest split. Not stated. |
| W8 | Cluster centers | Uniformly random inside the stage, staying at least 2σ_c from the edges. Drawn fresh every generation (like W4). | [GUESS] Not stated. Fresh-each-generation matches how S1 handles food. Other option: fixed for the whole transfer phase. |
| W9 | Food spread in a cluster | Each food item = cluster center + random offset, normal in x and y with standard deviation σ_c, then pushed back inside the stage if outside | [GUESS] Not stated |
| W10 | σ_c | Main choice: 15 (5% of the stage width). Also reported as a side check: 7.5 and 30. | [GUESS] Not stated. B results are held out from calibration, so this must be fixed **before** any B run and can't be tuned. |

---

## 4. A blob

### 4.1 Traits (the "genes")

| Trait | Start value (ET) | Mutation size (standard deviation) | Lowest allowed | Tag and evidence |
|---|---|---|---|---|
| speed | 10.0 | 0.5 | smallest positive number (about 2.2 × 10⁻³⁰⁸) | [SOURCE] `src/store/simulation.js:19`, `creature/mutatable.rs:43-46`; [PAPER] start value ER p. 5 |
| size | 10.0 | 0.5 | smallest positive number | [SOURCE] `src/store/simulation.js:20`; [PAPER] ER p. 5 |
| sense range | 20.0 | 0.5 | 0 | [SOURCE] `src/store/simulation.js:21`, `creature/mutatable.rs:29-32`; [PAPER] ER p. 5 |
| reach | 1.0 | **S1 default is 0 (never changes). The paper's Fig. 3 shows reach rising from about 1 to about 2, so the 2025 port must have let it mutate.** | smallest positive number | Start value: [SOURCE] `src/store/simulation.js:22`, [PAPER] Fig. 3 (chart). Mutation size: **[GUESS]**, free for calibration, options 0.1 / 0.25 / 0.5 |
| flee distance | 10¹² (in effect: flee from any predator you can see) | 0 | 0 | [SOURCE] `src/store/simulation.js:23` |
| life span | 10⁴ | 0 | n/a | [SOURCE] `src/store/simulation.js:24`. Not used: the old-age rule is not switched on (`api/mod.rs:26-36`). |

Notes:
- **Mutation size is a standard deviation.** S1's interface labels it "σ²" (`src/components/creature-template-config.vue:46-48`), but the code passes it to `Normal::new(mean, std_dev)` (`creature/mutatable.rs:5-8`; R). So 0.5 means a standard deviation of 0.5.
- The mutation sizes themselves are passed to offspring unchanged; they do not evolve (`creature/mutatable.rs:17, 31, 45`).
- The speed, size and sense mutation sizes start at S1's defaults (0.5). They were treated as **free calibration settings**. S1's 0.5 is one of the options.

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

The energy budget changes a lot along the way. Using the trait averages read off Fig. 3 (approximate): about 105 ticks at generation 10, about 100 at generation 20, about 62 at generation 50, and about 190–230 in Environment B, where size falls to about 4.8. Cost grows with size to the 5th power (size³ × (speed × size/10)²), so small changes in size matter a great deal. Shrinking is the cheap way to get faster. That fits the paper's trait trends, but only the rebuild can confirm it.

---

## 5. One generation, tick by tick

### 5.1 Start of a generation
| # | Rule | Tag |
|---|---|---|
| G1 | Very first generation: each blob is placed at a random point on the stage edge (pick a random spot, then move it to the nearest edge). | [SOURCE] `api/square_world.rs:31-33`, `stage/mod.rs:53-63` |
| G2 | Later generations: every blob starts at its "home" spot. That is the edge point that was its home at the end of the last generation (see Q6). A newborn starts at its parent's home spot. | [SOURCE] `creature/mod.rs:125-168` |
| G3 | Every blob starts awake, with 0 food eaten and full energy. | [SOURCE] `creature/mod.rs:86-110, 125-168` |
| G4 | Food is placed (W3 or W8–W10). | [SOURCE] / [GUESS] |
| G5 | Any blob with an effective speed of exactly 0 dies. (Can't really happen, because speed and size can't reach 0.) | [SOURCE] `simulation/behaviours/starve.rs:16-23` |

### 5.2 Each tick, in this order
The generation repeats ticks until no blob is awake (all are asleep at home or dead) (`simulation/generation.rs:62-67`). The tick counter starts at 1 (`generation.rs:31`). S1 stops with an error if a generation reaches tick 1,000,000 (`generation.rs:7, 85`). S1 has **no other limit on generation length**; Section 5.6 explains why that matters.

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

**An important side effect.** Wander always offers a "mild want" goal at a point only 1 unit ahead. Any other "mild want" goal is almost always farther away, so it loses the tie. In practice this means:
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
| C6 | Is this rule on in the rebuild? **No** (your decision at design stage). It is on in S1's default setup, and the paper never mentions it. The code is kept behind the switch `world.cannibalism` and was checked against S1 (Section 15.1). | [AUTHOR]; S1 default: [SOURCE] `api/mod.rs:26-36, 52-55` |

### 5.5 End of a generation
| # | Rule | Tag |
|---|---|---|
| E1 | Every living blob that ate 0 food dies. | [SOURCE] `starve.rs:37-43` |
| E2 | A blob that ran out of energy is dead, **even if it had already eaten 2 food**. It does not survive or reproduce. | [SOURCE] `creature/mod.rs:328-334`, `simulation/behaviours/reproduction.rs:24-34` |

### 5.6 How long a generation lasts (open question)
Under S1's rules, a generation ends only when every blob is asleep or dead. A blob with exactly 1 food has no home goal while its homesick number is above 10. It also isn't killed when the food runs out, because that rule only kills blobs with 0 food. So it keeps wandering until its energy runs low. **A cheap-to-run blob can keep a generation going for a very long time.**

On the proposed GS grid (GS3), 84 of the 512 designs have more than 10,000 ticks of energy. The cheapest (speed 2.5, size 2.5, sense 5) has about 450,000. At the other end, 176 designs have fewer than 5 ticks and die almost at once.

This matters for two reasons:
1. **Compute.** Under S1's rules, the cheap designs alone would make the grid search cost far more than the paper's 33.5 million steps (Section 10).
2. **Your Section 0 item 4 asks for "length of a generation (time steps or seconds)".** That suggests the 2025 port may have had a fixed generation length, which S1 does not have.

| # | Rule | Tag |
|---|---|---|
| L1 | Main choice: S1's rule (no tick limit). | [SOURCE] |
| L2 | Other option: a limit of T ticks per generation. When the limit is hit, the generation ends and the normal end-of-generation rules apply: blobs still alive with 0 food die (E1); alive blobs with 1 food survive; alive blobs with 2+ food reproduce. Candidate T values: 50, 75, 100, 150, 200. This changes ET's results, so calibration on ET training numbers chooses it. | [GUESS], approved at design stage |

**What the quick test showed (design stage).** With no limit, the cheapest grid design (speed 2.5, size 2.5, sense 5) won the 2×2×2 test grid, tied with 3 others. Its 50 slow clones wander for hundreds of thousands of ticks until each finds exactly 1 food, so it scores exactly 1.0. Running it for 10 generations took about 53 million creature-steps per seed. And without a limit, ET generations in the reference check averaged about 410 ticks, because the most frugal blob keeps going. Both are hard to square with the paper's GS numbers (fitness 1.30, population 37) and its compute (about 43 steps per blob per generation). That is more evidence that the 2025 port had a time limit. A default blob has exactly 50 ticks of energy, so a limit near 50 would be a natural choice for a porter.

---

## 6. Between generations

| # | Rule | Tag |
|---|---|---|
| R1 | Dead blobs are removed. | [SOURCE] `reproduction.rs:25-27` |
| R2 | Every living blob survives into the next generation, unchanged except for age + 1. | [SOURCE] `reproduction.rs:30-31`, `creature/mod.rs:141-168` |
| R3 | Every living blob with **2 or more** food (food items plus prey eaten) also makes **exactly one** offspring. | [SOURCE] `reproduction.rs:10-20` |
| R4 | The offspring's traits = the parent's traits, each with independent normal noise added (Section 4.1 sizes), then clipped to the lowest allowed value. | [SOURCE] `creature/mod.rs:125-138`, `creature/mutatable.rs` |
| R5 | The offspring starts at the parent's home spot. | [SOURCE] `creature/mod.rs:136` |
| R6 | So: 0 food = die, 1 food = survive, 2+ food = survive and make one offspring. No blob makes more than one offspring per generation. | [SOURCE], and matches the plan's clue (Section 11) |
| R7 | A run ends early if no blob is alive. An extinct run stays extinct. | [SOURCE] `simulation/mod.rs:96-113` |

---

## 7. The two conditions

### 7.1 ET (evolutionary training)
| # | Setting | Value | Tag |
|---|---|---|---|
| ET1 | Starting group | 50 identical default blobs (Section 4.1) | [PAPER] ER p. 3; [SOURCE] defaults |
| ET2 | Training | 100 generations in Environment A, numbered 0–99 | [PAPER] ER p. 3, Fig. 1 x-axis |
| ET3 | Mutation | Section 4.1. Reach mutation size is the main free setting. | [SOURCE] + [GUESS] |
| ET4 | Selection | Only the survival and reproduction rules above. No score is used. | [PAPER] ER p. 3; [SOURCE] |

### 7.2 GS (grid search)
| # | Setting | Value | Tag and evidence |
|---|---|---|---|
| GS1 | Grid size | 8 × 8 × 8 = 512 designs | [PAPER] ER p. 3 |
| GS2 | Which 3 traits | speed, size, sense range. Reach fixed at 1.0. | [GUESS] These three are the traits S1's settings screen lets you change (`src/components/creature-template-config.vue:30-32`). The paper lists 4 traits for ET, so **ET evolves one trait (reach) that GS can't search**. The later trait experiment searches all four traits. |
| GS3 | Grid values | Main choice: 8 steps spaced evenly on a log scale, from 0.25× to 4× each default. Speed and size: 2.5, 3.71, 5.52, 8.20, 12.19, 18.11, 26.92, 40.0. Sense: 5.0, 7.43, 11.04, 16.41, 24.38, 36.23, 53.84, 80.0. | [GUESS] Not stated. Other option: evenly spaced on a normal scale over the same range. GS numbers are held out, so this must be fixed before any GS run. Section 14, question 6. |
| GS4 | How a design is scored | Mean food per blob (Section 8, M1) while running as 50 clones in Environment A. | [GUESS] The paper says the search found the "best-performing configuration" and that GS beat ET on "raw fitness" (ER p. 3–4). Grid-search scoring uses food per agent. |
| GS5 | How long each design is run, and whether there is one search or one per seed | | GS6 | Deployment | The winning design as 50 identical clones, mutation size 0 for every trait, 100 generations in A, then transfer. Clones still live, die and reproduce by the same rules, so their number changes. | [PAPER] ER p. 3, Fig. 2 |
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

Each metric is computed per run (per seed), then summarized across the 30 runs as mean ± standard deviation, as the paper does.

| # | Metric | Definition | Tag and evidence |
|---|---|---|---|
| M0 | Population of a generation | Number of blobs that **start** the generation | [SOURCE] S1's own statistics count every blob in the generation's list (`api/mod.rs:130-176`) |
| M1 | Fitness (food per blob) | (food items eaten + prey eaten) ÷ population (M0) | [GUESS]. S1 records both kinds in the same list (`creature/mod.rs:272-274`). Counting prey was selected to explain the Environment B numbers (Section 11.4). Other option: food items only. |
| M2 | Median fitness | Median of each blob's food count in that generation | [PAPER] Fig. 11 row 2; definition [GUESS] |
| M3 | Final training fitness | M1 in the **last training generation only** (generation 99) | [GUESS] The table's SD across seeds (ET 0.03, GS 0.10) matches the width of the shaded across-seed band at single generations in Fig. 1 (about ±0.03 for ET, ±0.10 for GS). The ET transfer rows, which are 10-generation averages, have a much smaller SD (0.01). Other option: mean of generations 90–99. |
| M4 | Survival rate | Blobs alive at the end of a generation ÷ blobs that started it | [GUESS] Which generations are averaged is not stated. Options: all 100 training generations, last 10, or the last one. Chosen by calibration. |
| M5 | Transfer fitness, "gen 1–10" | Mean of M1 over the **first 10 transfer generations (B generations 0–9)**, including the first one where fitness drops | [GUESS] Reading Fig. 6 (chart), the mean of ET's B generations 0–9 is about 1.03, which matches the table's 1.03. The mean of B generations 1–10 would be about 1.08, which doesn't. GS points the same way: about 1.16 for generations 0–9 and 1.23 for 1–10, against the table's 1.13. |
| M6 | Transfer fitness, "last 10" | Mean of M1 over B generations 40–49, using only generations where the population is above 0. A run that is extinct for all 10 gets no value. The number of contributing runs is reported. | [GUESS] The paper's two transfer rows are identical (likely a copy error). |
| M7 | Generalization gap | M3 − M5 | [PAPER] ER p. 9; check in Section 11.2. Also reported (labeled): M3 minus the mean of B generations 0–4, which reproduces the bar chart (Section 11.2). |
| M8 | Transfer trend slope | Straight-line (least squares) slope of M1 against generation number over B generations 0–49, using only generations with population above 0 | [GUESS] The text and table disagree on which group is 0.02 (plan, "Known contradictions"). The reconstruction reports its own value. |
| M9 | Trait diversity (CV²) | For each trait: (standard deviation ÷ mean)², summed over the traits | [GUESS] Options, chosen by calibration: (i) 4 traits (speed, size, sense, reach) or 3; (ii) population or sample standard deviation; (iii) last generation or average over training. **Note:** with speed, size and sense alone, a rough estimate gives about 0.1 or less, well under 0.41 (Section 11.6). CV² is also reported per trait. |
| M10 | Shannon diversity (H) | −Σ p·ln p over "types" of blob in the generation, where p = share of blobs of that type | [GUESS] What counts as a "type" is not stated. Options, chosen by calibration: (i) exact trait values, (ii) traits rounded to 1 decimal, (iii) traits rounded to whole numbers. If every one of 48 blobs were unique, H would be ln 48 = 3.87. The paper's 3.77 means some blobs share a type, which points toward rounding. |
| M11 | Energy efficiency | **Not defined in the source paper.** It was reconstructed during calibration. The paper's GS value is 0.00 ± 0.00, so the definition must give exactly 0 for clones that never change. Candidates, each computed per run on training generations: **E1** relative drop in mean energy cost per tick, generation 0 → 99; **E2** same, for cost per unit distance (cost ÷ effective speed); **E3** E2 averaged over all 100 training generations; **E4** spread (CV) of energy cost per tick across blobs at generation 99; **E5** same as E4 for cost per unit distance. Calibration picks the one closest to ET's 0.20 ± 0.18. | [AUTHOR] (include it) + [GUESS] (definition). The definition is a reconstruction. |
| M12 | Traits over time | Mean of each trait across living blobs at the start of each generation | [PAPER] Fig. 3; definition [GUESS] |
| M13 | Population over time in figures | Mean across runs, two ways: counting extinct runs as 0, and averaging only runs still alive. The paper's GS line in Figs. 6–7 stops at about B generation 26, which suggests the second way. | [GUESS] |
| M14 | Recovery time (for Figs. 6–7 text) | Reported descriptively only | n/a |

**About choosing metric definitions by calibration.** The plan says to pick unknown options using calibration. That is allowed here, but it means a match on these metrics is partly "made" by the choice of definition. These calibration choices limit the interpretation of an apparent match. Each metric definition only affects its own metric, so these choices don't interact with the simulation settings.

---

## 9. Statistics

| # | Rule | Tag |
|---|---|---|
| S1 | Welch's t-test, two-tailed, 30 vs 30 runs | [PAPER] ER p. 3, Fig. 11 |
| S2 | Degrees of freedom: Welch–Satterthwaite formula | [PAPER] Fig. 11 df column is consistent with this formula, within the rounding of the printed SDs (Section 11.1) |
| S3 | Cohen's d = (mean_ET − mean_GS) ÷ √((SD_ET² + SD_GS²) ÷ 2). With 30 per group this equals t × √(2/30), which the plan asks the code to reproduce. | [PAPER] all Fig. 11 rows match (Section 11.1) |
| S4 | Bonferroni correction for 10 comparisons (multiply p by 10, cap at 1) | [PAPER] ER p. 3. The table has 9 rows, and two are identical, so only 8 distinct comparisons are shown. Two of the 10 are unaccounted for (one, if the duplicate row stands for a real comparison). The correction uses 10 as stated. |
| S5 | Report **both** the uncorrected p (what the table's p column actually shows) and the corrected p (what its stars show). | [PAPER] Section 11.1 |
| S6 | Tests on values that are 0 by design (GS diversity, GS CV²) are computed for comparison but marked "not meaningful". When one group has 0 spread, Welch's df becomes 29, as in the table. | [PLAN] Step 5 |

---

## 10. Counting compute ("simulation steps")

| # | Rule | Tag |
|---|---|---|
| K1 | Count one **creature-step** each time one awake blob moves one tick. Sum over all generations of all 30 runs, both phases. For GS, add every creature-step of the search. | [GUESS] from the clue below. Medium confidence. |
| K2 | Also report plain **ticks** (one per tick of a generation, however many blobs) and wall-clock time. | [PLAN] Step 3 |

**ET clue.** ET used about 7.9 million steps (ER p. 11).
- 7.9M ÷ (30 runs × 150 generations) ≈ 1,756 per run-generation. ET's average population over A and B is about 41, so that's about 43 per blob per generation.
- A blob's energy lasts roughly 45 to 230 ticks, depending on its traits (Section 4.2). So "one step = one blob moving one tick" is a plausible reading.
- Plain ticks can't explain the number. Even with those energy budgets, they would add up to well under 1 million.
- **Whether both phases were counted can't be told.** Counting training only gives about 56 per blob per generation, which is also within the energy budget.
- **Confidence that a step is a creature-step: medium.**

**GS clue.** GS used about 33.5 million (ER p. 11).
- If the deployed clones also cost about 43 steps per blob per generation, deployment comes to roughly 30 × (100 × 37 + about 150) × 43 ≈ 5.0 million. That leaves about 28.5 million for the search. The 43 is an assumption: the real figure depends on which design won.
- Per design: 28.5M ÷ (512 × 30) ≈ 1,860 if the search ran once per seed. That is about one generation of 50 clones × 37 ticks.
- Or 28.5M ÷ 512 ≈ 55,700 if the search ran once in total. That is about 30 generations of 50 clones.
- **But under S1's rules, the grid in GS3 can't fit inside 28.5M either way.** The cheap designs keep generations running for thousands to hundreds of thousands of ticks (Section 5.6). Their energy budgets add up to about 6 million ticks for one pass through the grid. Even one long-lived blob per design per generation would give about 180 million steps for either option.
- So at least one of these must be true of the 2025 port: (i) a limit on generation length (L2), (ii) a narrower grid than GS3, or (iii) a different meaning of "step". The paper does not resolve this ambiguity.
- **Confidence: low.**
- **Note:** if any GS setting is chosen to fit this 33.5M number, GS compute is no longer a held-out check. All GS *performance* numbers stay held out either way.

---

## 12. Calibration plan (details go into `calibration/targets.json` at design stage)

### 12.1 Free settings (all [GUESS] items that can affect ET in Environment A)
| Setting | Options | Changes the simulation? |
|---|---|---|
| Reach mutation size | 0.1, 0.25, 0.5 (0 is S1's default, but it can't produce Fig. 3) | Yes |
| Speed, size, sense mutation sizes | 0.5 (S1), 1.0, 2.0 ([AUTHOR] allowed at design stage) | Yes |
| M11 energy efficiency | E1–E5 | No, metric only |
| Generation length (5.6) | no limit (S1, L1) / limit of 50, 100 or 200 ticks (L2), selected during calibration | Yes |
| M3 window | last generation / mean of last 10 | No, metric only |
| M4 window | all 100 / last 10 / last 1 | No, metric only |
| M9 options | 3 or 4 traits; population or sample SD; last generation or training average | No, metric only |
| M10 type definition | exact / 1 decimal / whole numbers | No, metric only |
| M1 prey counted | yes (main choice, from Section 11.4) / no | No, metric only |

Three settings change the simulation (reach mutation, the other mutation sizes, the time limit). Everything else is a choice of metric definition.

### 12.2 Settings that can't be calibrated (fixed now, before any run)
Every GS setting (GS2–GS5) and every Environment B setting (W7–W10, T3). The plan holds all GS and transfer numbers out of calibration, so nothing can tune these. These settings were fixed before final evaluation. For W10 (σ_c) and W8 (fresh or fixed clusters), Results are reported for the main choice and for the side-check values, all labeled.

### 12.3 Calibration targets (ET, Environment A only)
- Fitness curve: about 0.87 at generation 0, peak about 1.12 at generations 4–5, back to about 1.03 by generation 12, then flat (Fig. 1 chart). Final 1.03 ± 0.03, median 1.00 ± 0.00 (Fig. 11).
- Population curve: 50 at generation 0 → about 41.5 at 1 → dip to about 37 at generations 3–4 → about 47 by generation 10 and 48.5 by 14 (Fig. 2 chart) → about 48 at generation 100 (ER p. 4).
- Survival rate: 66.5% (ER p. 4).
- Traits at generation 100: speed 26.6, size 6.9, sense 31.2 (ER p. 5); reach about 2 (Fig. 3 chart).
- Diversity: H 3.77 ± 0.06, about 3.8 by generation 10 (Fig. 5 chart); CV² 0.41 ± 0.29 (Fig. 11).

### 12.3a Outcome (31 Oct; details in `calibration/REPORT.md`)
- Selected by the written rule: **trait mutation 1.0, reach mutation 0.25, time limit 200 ticks, blob-eating off** (`configs/part1/final.yaml`). 70 of 76 targets pass with the primary definitions.
- Misses: reach rises too slowly (1.78 vs 2.12 at generation 99; also generations 60 and 80); CV² at generations 10 and 40 a bit low; final Shannon on the primary definition (3.88 vs 3.77 ± 0.06), which passes on the stage-2 definition (training mean, 3.75).
- **The top settings can't be told apart.** On two extra seed sets the selected setting scores 0.51 and 0.58 (0.29 on the calibration seeds), and close neighbors swap places (`calibration/score_noise_check.json`). What holds on every seed set: blob-eating off (best with it on: 47/76), reach mutation 0.25, other mutations near 1.0 (twice S1's 0.5). A limit of 175–225 ticks and no limit score about the same.
- Stage-2 definitions: final fitness = mean of the last 10 training generations; survival = last 10; CV² = 4 traits, sample SD, training mean; Shannon = exact types, training mean; energy efficiency = E2 (drop in energy cost per unit distance), 0.28 ± 0.09 vs the paper's 0.20 ± 0.18.
- **Measured, not guessed any more:** with speed, size and sense alone, CV² is about 0.06; reach adds about 0.33. Most of the paper's "trait diversity" number comes from reach.

### 12.4 Held out (never used for tuning)
Every GS number. Every transfer-phase number. Compute numbers, except the documented GS5 calibration clue.

### 12.5 Seeds
Calibration: 1000–1029. Final runs: 0–29, per condition. [PLAN] Checks against the Rust original and the quick test use seeds 9000–13999, which no other purpose uses. These seeds were reserved for implementation checks. GS search: seed 999. The selected shared grid search uses seed 999. Each run splits its seed into separate random streams (for example search vs. simulation) with numpy's SeedSequence, so no stream is reused.

---

## 13. Quirks in S1's code, and what the rebuild does

The rebuild copies these by default, because the 2025 port most likely did. Each one gets a config switch so its effect can be checked.

| # | Quirk | Where | Default |
|---|---|---|---|
| Q1 | The predator–prey pairing loop skips some pairs. When the blob at list position i is not awake, it skips pairing the blob at position i−1 with everyone after it. This looks like a bug. | `cannibalism.rs:11-12` | Copy it |
| Q2 | A blob only checks its single nearest food when eating. It can pass within reach of a farther food item and not eat it. | `scavenge.rs:31-39` | Copy it |
| Q3 | Energy is paid in MOVE, before eating in ACT. A blob that hits 0 energy on its last move dies before it can eat that tick. | `generation.rs:91-95` | Copy it |
| Q4 | A blob that runs out of energy dies even after eating 2 food (E2). | `creature/mod.rs:328-334` | Copy it |
| Q5 | Predators keep eating even when full, so a blob can have more than 2 food (C4). | `cannibalism.rs:84-93` | Copy it |
| Q6 | "Home" moves every tick to the nearest edge from where the blob is now. | `edge_home.rs:32-39` | Copy it |
| Q7 | Every blob's first heading each generation is +x, plus the random turn. | `creature/mod.rs:231-236` | Copy it |
| Q8 | S1's "births" statistic also counts dead blobs that ate 2 food. | `api/mod.rs:157-159` | Not copied: births = living blobs with 2+ food (R3) |
| Q9 | Movement cost is the same for a full move and a move cut short at the edge. | `move.rs:7-14` | Copy it |
| Q10 | "Mild want" goals (going home with a homesick number between 5 and 10; chasing prey when already full) almost always lose to wander's goal 1 unit ahead, so they rarely steer the blob (Section 5.3). | `wander.rs:25-29`, `creature/mod.rs:250-257` | Copy it |
| Q11 | A blob pushing into a corner can make a move of zero length. Its next heading is then undefined (a division by zero gives "not a number"). Wander's "1 unit ahead" point then fails the inside-the-stage check, so the blob aims at the stage center. | `creature/mod.rs:231-238`, `wander.rs:25-29` | Copy the outcome (aim at center), without relying on "not a number" math |

The random numbers will not match S1's bit for bit. S1 uses Rust's SmallRng; the rebuild uses numpy. Only the rules match.

---

