# Neural environment and implementation specification

Scientific excerpts from the dated study protocol. Operational checklists, private discussion and task lists are omitted. Original section numbers and decision rules are retained. These are edited archival excerpts, not a new registration. Original document SHA-256: `c1061810041008bec5a3ca4e00ba6889d0c516cdbfaca650f3e0c896ff176b15`.

## 3. The world (`src/evo2/env`)

A continuous, square 2D arena with walls. Time moves in discrete steps. All constants live in `configs/part3/*.yaml`.

### 3.1 Agents

**Body traits** (provisional ranges; the default is the start for evolving methods):

| Trait | Range | Default | Effect |
|---|---|---|---|
| max speed | 0.25–2.0 units/step | 1.0 | upper bound on the speed action |
| size | 0.5–2.0 | 1.0 | eating reach = size; energy cost grows with size³ |
| sense range | 4–20 units | 10 | how far the sensors see; costs energy |

**Sensors** (the brain's inputs, 30 numbers in total). Directional sectors were selected before training.
- K = 8 sectors, each 45°, covering 360° around the agent's heading. Sector 0 is centred on the heading (−22.5° to +22.5°); sector i is centred i × 45° to the left (counter-clockwise).
- In each sector, three values, one each for food, other agents and walls. Each is 1 − d / range for the nearest object of that kind within sense range, or 0 if there is none. Distances are between centres. An object at exactly the sense range reads 0.
- Walls: d is the distance to the wall along the sector's centre line.
- Own energy ÷ the birth energy, clipped to [0, 2]. Previous action (2 numbers, the squashed values in [−1, 1]). The 3 body traits, scaled to [0, 1].
- Dead agents' slots read all zeros.

**Actions** (2 numbers). Each is squashed into its range with tanh, so the bounds hold whatever the network outputs:
- turn rate = (π/4) × tanh(a₁), in [−π/4, π/4] per step;
- speed = max speed × (tanh(a₂) + 1) / 2, in [0, max speed].
- **[core build]** At evaluation every method acts deterministically: the brain's output is the action. PPO samples Gaussian noise around that output only while training.

**Energy** (provisional constants; same structure as Part 1: cost grows with size³ × speed², plus a sense cost):
- cost per step = 0.5 + 1.0 × size³ × speed² + 0.05 × sense range;
- eating a food item gives 40;
- energy at birth or episode start is 100;
- an agent dies at energy ≤ 0.

With default traits at full speed, the cost is 2.0 per step, so an agent lasts 50 steps without food.

**Eating.** A food item within an agent's reach (centre distance ≤ size) is eaten. If several agents reach the same item in one step, the one with the lowest random draw that step gets it. One item can be eaten only once. An agent can eat several items in one step.

**Order of events in one step [core build]** (both modes; `src/evo2/env/world.py`):
1. Alive agents turn, then move; walls stop movement (positions are clipped to the arena).
2. Eating, at the new positions.
3. Every agent alive at the start of the step pays its energy cost.
4. Death: energy ≤ 0, or (ecological mode) age ≥ 1,000.
5. Births (ecological mode).
6. Food: eaten items regrow, then present items drift (bouncing off walls), then, every P steps, the cluster or patch centres and the uneaten items move.

### 3.2 Food and environment parameters

| Parameter | Meaning |
|---|---|
| arena side L | square arena, side L |
| food count F | food items present when fully grown |
| layout | uniform; clusters (k centres, Gaussian spread σ); patches (k discs of radius r, uniform inside). Centres are drawn uniformly in [0.15 L, 0.85 L]²; food positions are clipped to the arena |
| regrowth rate ρ | each eaten item comes back with probability ρ per step, at a new position drawn from the layout |
| drift speed | every food item moves this far per step in its own fixed random direction, bouncing off walls |
| seasonal period P | every P steps the cluster or patch centres move to new random places, and so do uneaten items |
| hazards (optional, not in the core) | discs that drain extra energy per step |

### 3.3 Two modes

- **Ecological mode** (OE's training, and persistence tests):
  - an agent with energy ≥ 200 gives birth: the child starts with 100 and appears 1 unit from the parent in a random direction (clipped to the arena), and the parent loses 120 (20 is the cost of birth);
  - when several agents can give birth in one step, they are taken in a random order and matched to free slots in slot order;
  - the child's genome is the parent's, mutated (Section 4.2);
  - agents die at energy ≤ 0 or at age 1,000 steps;
  - the arrays hold at most 128 agents: when full, no births happen, and the parent keeps its energy;
  - there is no fitness function anywhere.
- **Episodic mode** (training PPO and ES, and every competence measurement):
  - 16 agents, 500 steps, no births;
  - dead agents stay dead until the episode ends;
  - all agents start with energy 100 at random positions;
  - score: food eaten per agent.

### 3.4 Environment splits (locked before any training)

Provisional ranges; the locked values go in the config files and the pre-registration.

| Split | Where defined | Contents |
|---|---|---|
| Train distribution | `configs/part3/train.yaml` | L 40–80; F 20–60; layout any (k 2–5, σ 3–8, patch radius 4–10); ρ 0.01–0.05; no drift; no seasonal moves |
| Canonical environment (PPO-fixed, fixed-world ablation) | `train.yaml` | L 60, F 40, uniform, ρ 0.03 |
| Validation set | `configs/part3/validation.yaml` | 20 fixed environments drawn from the train ranges with environment seeds 4900–4919, plus 4 held-out corner combinations. Used only for tuning and sanity checks |
| Test set | `configs/part3/test_split.yaml` (locked) | **T1** in-distribution: 10 environments from the train ranges, environment seeds 5900–5909. **T2** near shift: smaller (L 30–40), scarcer (F 10–20), clustered (k 2–3, σ 2–4), slower regrowth (ρ 0.005–0.01), 10 environments. **T3** far shift: drifting food (0.05–0.3 per step) with seasonal relocation (P 100–300), on train-range sizes, 10 environments |

**The lock.** Only `src/evo2/env/splits.py` reads `test_split.yaml`, and only when called with `final=True`. `scripts/part3/evaluate.py` passes that only with an explicit `--final` flag. A test checks that every training, tuning and pilot entry point fails if it tries to load the test file.

---

## 4. Agents' brains and genomes (`src/evo2/agents`)

### 4.1 Brain

- The same network for every method:
  - a small multilayer perceptron, 30 inputs → 32 hidden (tanh) → 32 hidden (tanh) → 2 outputs;
  - about 2,100 weights *(provisional size)*;
  - the outputs are squashed into the action ranges.
- No memory beyond the previous-action input, so every method gets the same kind of brain.
- PPO adds a separate value network of the same shape. It is used only in training and is not part of the policy.

### 4.2 Genome and mutation (evolving methods)

- **Genome** = brain weights + the 3 body traits.
- **Mutation at birth** *(provisional sizes)*:
  - each weight gets Gaussian noise (sd 0.02); weights have no bounds;
  - each trait gets Gaussian noise of 5% of its range, then is clipped to its bounds;
  - a test checks that the bounds always hold.
- **Initial population** (OE): 32 agents, each with a default body. Brains are drawn from the same initializer PPO uses, from the run's seed.

---

## 5. Methods (`src/evo2/methods`)

Every method uses the same brain, world, compute budget (Section 6), tuning budget and evaluation.

| Method | What it does | Training mode |
|---|---|---|
| **OE** (open-ended evolution) | One population lives in a shared world. The world's parameters are redrawn from the train distribution every 2,000 steps *(provisional)*; agents stay where they are (clipped into a smaller arena). No fitness function, only survival and births with mutation | ecological |
| **OE-fixed** (ablation, Q2) | OE in the canonical environment only | ecological |
| **PPO-fixed** | PPO in the canonical environment. All 16 agents share one network (parameter sharing), and each agent's experience is its own trajectory. Reward = food eaten that step | episodic |
| **PPO-DR** (main baseline) | PPO with each episode in an environment drawn from the train distribution | episodic |
| **ES-DR** | OpenAI-ES (evosax `Open_ES`, antithetic sampling, centred-rank fitness shaping) optimizes one parameter vector. Fitness = mean food per agent over the candidate's episodes, in environments drawn from the train distribution. All candidates in a generation share the same environments (common random numbers) | episodic |

**Fixed bodies.** The default body was selected for every method on 2 Nov 2026 (`body_option: b` in `configs/part3/methods.yaml`): maximum speed 1.0, size 1.0, sensing range 10. OE mutates controller weights only. Body traits remain fixed inputs.


**OE needs a population floor [core build].** Quick tries on test seeds 3980–3984 (OE as above, no floor, train distribution, random keys made the way real runs make them: the seed's "train_run" stream) went extinct every time, after 177, 176, 274, 177 and 177 steps, with 1, 0, 2, 0 and 0 births before that.  Newly initialized brains do not find enough food to breed before they starve. So the code has a population floor (`MIN_POP = 16`, selected before final runs):
- whenever fewer than 16 agents are alive after a step, empty slots are filled up to 16 with mutated copies of agents chosen uniformly at random from those alive and not born this step;
- the copies appear at random places with birth energy 100, which comes from outside the world;
- there is still no fitness function: every living agent is equally likely to be copied, so longer-lived agents are copied more only because they are alive longer;
- every floor birth is counted and reported, next to normal births.
In one 16,000-step quick try with the floor (seed 3980, default settings, "train_run" keys), floor births were 354 of the 489 births in the first 6,000 steps and 381 of the 471 in the last 6,000; the population averaged 18.3 agents, and 16.3 in the last 2,000 steps. So in this try the population still lived mostly on floor births after 16,000 steps (about 290,000 agent-steps, 0.15% of the provisional training budget). One run only. Whether OE breeds on its own later is a pilot question; if it does not, that is a stop-and-ask trigger (PREREG Section 10).

**Method details settled while building [core build]:**
- **PPO:** dead agents' steps are left out of every loss term (a `valid` mask); an agent's death ends its trajectory; the end of an episode (500 steps) also ends it, as in PureJaxRL. Optional running reward normalization (PureJaxRL's NormalizeVecReward) is a tuning setting. In the adaptation phase PPO continues from its trained weights with a fresh Adam state.
- **ES:** the policy after training is the ES mean (not the best candidate seen). Body traits (option (a)) are 3 numbers mapped to the trait ranges with a sigmoid, starting at the default body.
- **OE:** in the adaptation phase OE keeps evolving with the environment fixed to the test environment.

Lifetime learning and neural mechanism ablations were planned but not run.

**Implementations used:**
- PPO is adapted from PureJaxRL's `ppo_continuous_action.py` (Apache-2.0; credited in `src/evo2/methods/ppo.py` and in `README.md`), changed for multi-agent parameter sharing in this world.
- ES uses evosax 0.3.2 `Open_ES` (Apache-2.0).
- **Sanity checks before use**, on gymnax tasks:
  - PPO must solve Pendulum-v1 to a pre-set return;
  - ES must improve on a standard gymnax task;
  - the targets and their sources are fixed in the pre-registration.

---

## 6. Compute accounting

- **Training budget:**
  - every method gets the same total number of training agent-steps, counting every agent-step (*provisional*: 2 × 10⁸; the pilot sets it);
  - every method counts the same thing: agents alive at the start of each step (OE in its shared world; PPO in its parallel worlds; ES in every candidate's episodes). The first draft charged PPO and ES for dead agents' slots as well, which early in training is 4–5 times their living agent-steps, and would have favored OE;
  - so that PPO spends no simulation on empty worlds, an episodic training world ends its episode as soon as every agent in it is dead (or after 500 steps) and starts a new one; this also lets the parallel worlds drift out of step.
- **Adaptation budget** (after the move to a test environment): every method gets the same number of agent-steps (*provisional*: 2 × 10⁷).
- **Tuning:**
  - the same number of random-search trials for every method, scored on the validation set only;
  - tuning compute is reported separately;
  - search ranges are declared before tuning.
- **Reported for every run:** agent-steps, wall-clock time and hardware.
- **Stopping at the budget [core build]:** each method runs in chunks of units (one PPO update, one ES generation, one OE step). The next chunk is the full chunk only if even fully alive units could not pass the budget; otherwise it is the largest power of two of units that could not pass it, down to one unit. So no method passes the budget by a full unit, and a resumed run takes the same chunks as an uninterrupted one. At the default settings a unit is at most 65,536 agent-steps for PPO (0.03% of 2 × 10⁸), 1,024,000 for ES (0.5%) and 128 for OE. The actual count is recorded for every run.
- **Snapshots:** each run saves its policy after every tenth of the training budget, for learning curves and (OE) diversity over time.
- **Adaptation compute is large [core build].** At 2 × 10⁷ agent-steps per test environment, adaptation on all 30 test environments costs 6 × 10⁸ agent-steps per run, three times the training budget; on T2 and T3 only (the primary metric) it costs 4 × 10⁸. The selected protocol used 2 × 10⁷ adaptation agent-steps on all 30 test environments.
- **Compile time [after P0, 2 Nov 2026].** P0 showed that short GPU runs spent most of their time compiling (OE on the RTX 4090: 149 s for about 3 s of work). Measured on this machine: every chunk length compiled its own program (the budget logic uses the full chunk, then powers of two down to 1 near each target), and every adaptation built a new trainer, so it compiled again for every environment and run (OE: its chunk program compiled about 11 times per adaptation here, about 5 s each); for PPO and ES the test environment was also written into the compiled program. Changes:
  - one compiled program serves every chunk length (`evo2.methods.loop.run_dynamic`);
  - adaptation keeps the test environment in the state (PPO's worlds, ES's runner, OE's state), so one trainer per method serves every environment and run (`evo2.run.get_trainer`);
  - compiled programs are saved on disk and reused by later processes (`evo2.runio.enable_compile_cache`).

  Checked against the code before the change (test seed 3955, trained policies, environment P3[0]): all five methods' training, and OE, OE-fixed, PPO-fixed and PPO-DR adaptation, are bit-identical. ES-DR adaptation is not: with the environment written into the program, the compiler rounds some arithmetic differently in the last bit (first difference 9.5 × 10⁻⁷ in a position, at step 37 of a trained policy's episode; one float32 step at the arena size is 3.8 × 10⁻⁶), and the world, being chaotic, grows that into different episodes. Neither version is more exact. The new version gives every environment and every method the same compiled arithmetic. No ES adaptation result existed before the change.
- **Speed on this machine [core build]** (2 CPU cores, one quick run each, so rough; counted as all slots, before the accounting change above, and with smaller settings than `methods.yaml`): PPO about 74,000 slot-steps per second (16 worlds; `methods.yaml` has 64); ES about 250,000 (32 candidates × 2 episodes; `methods.yaml` has 64 candidates); OE about 300 steps per second, about 6,500 agent-steps per second at 16–40 agents. OE is slow here because one world steps one step at a time; on a GPU, many OE runs (seeds) step together. The pilot measures GPU speed.

---

## 7. Evaluation (`src/evo2/eval`) and metrics (`src/evo2/metrics`)

**Individual competence.** Food per agent in an episodic episode: 16 agents, 500 steps, in a fixed set of evaluation episodes per test environment, with evaluation seeds fixed in the pre-registration.
- For OE, 16 agents are sampled from the living population, a new sample for each episode (without replacement when at least 16 are alive).
- The random policy is scored on the same episodes as the policy. The adaptation curve's point 0 is the zero-shot competence (the same number). Evaluation episodes are the same for every method and run; the adaptation's own randomness comes from the run's seed, so it differs between runs.
- PPO's reward-normalization statistics are taken over living agents and carried into the adaptation phase with the weights.
- For PPO and ES, 16 copies of the policy are used. With option (a), PPO uses its tuned body and ES its optimized one.

**Normalization** per test environment: (score − random policy score) ÷ (best observed score − random policy score).
- The random policy draws actions uniformly within bounds and uses the default body.
- "Best observed" is defined exactly in the pre-registration.

**Protocols**, run on every test environment:
1. **Zero-shot.** Everything frozen: no learning, no mutation.
   - individual competence;
   - ecological persistence: 32 agents (32 sampled from an OE population, or 32 copies of a PPO or ES policy) placed in ecological mode, with births but no mutation, no population floor and no environment change, for 5,000 steps; extinction yes or no, and steps survived.
2. **Adaptation.** Each method continues its own improvement process in the test environment for the adaptation budget: OE keeps evolving, PPO keeps training, ES keeps optimizing. Competence is measured at 10 evenly spaced checkpoints *(provisional)*.

**Primary metric** (from the plan): the area under the normalized competence curve during adaptation on T2 and T3, aggregated with the interquartile mean (IQM) over test environments and seeds.

**Secondary metrics:** zero-shot competence; final competence after adaptation; time to reach 90% of pre-shift competence; extinction rate in ecological mode; genotypic and behavioural diversity; compute cost.

**Statistics:** rliable 1.2.0 (Apache-2.0): IQM with 95% stratified bootstrap CIs, probability of improvement, performance profiles. The exact definitions go in the pre-registration.

---

## 8. Seeds (plan: Part 3 pilots 3000–3999, tuning 4000–4999, final 5000–5999; never reused across purposes)

| Use | Seeds |
|---|---|
| Unit tests and smoke runs | 3900–3999 |
| Pilot runs | 3000–3099 |
| Pilot evaluation: stand-in shifted environments 3100–3109 and their episodes 3130–3199 (`configs/part3/pilot_shift.yaml`) | 3100–3199 |
| Statistics: bootstrap (4800) and the behavioral-diversity probe set (4801) | 4800–4899 |
| Tuning runs | 4000–4799 |
| Validation environments and their episodes | 4900–4999 |
| Final runs (count from the pilot's power analysis; at least 10 per method) | 5000–5499 |
| Test environments and their evaluation episodes | 5900–5999 |

Each seed is split into named streams with SeedSequence / JAX PRNG keys, and a test checks that keys never serve two purposes (the Part 2 stream-key test, carried over).

---

## 9. Libraries (checked 2 Nov 2026)

| Library | Version | License | Use |
|---|---|---|---|
| jax / jaxlib | 0.11.2 | Apache-2.0 | everything |
| flax | 0.12.12 | Apache-2.0 | networks |
| optax | 0.2.8 | Apache-2.0 | optimizers |
| distrax | 0.1.9 | Apache-2.0 | PPO action distributions |
| evosax | 0.3.2 | Apache-2.0 | ES-DR (`Open_ES`) |
| gymnax | 1.0.1 | Apache-2.0 | baseline sanity checks |
| rliable | 1.2.0 | Apache-2.0 | statistics |
| PureJaxRL (code adapted, not installed) | commit 31756b1 | Apache-2.0 | PPO |

Every version is pinned in `requirements_part3.txt`, with pandas held at 2.3.3 so that rliable's dependency arch 7.2.0 imports. rliable is used for the IQM and the probability of improvement; its bootstrap is not used (PREREG Section 8.1). Parts 1–2 keep their own environment (`requirements.txt`), so they stay reproducible.

---

