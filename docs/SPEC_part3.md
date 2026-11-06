# Neural environment and implementation specification

This document describes the implementation used for the reported experiments. It incorporates the final design choices and tuned settings; the protocol and dated ES-fixed extension are in [PREREG_part3.md](PREREG_part3.md). Section numbers are retained for cross-references from the code.

## 3. The world (`src/evo2/env`)

The environment is a continuous, square 2D arena with walls and discrete time steps. Constants are defined in `configs/part3/world.yaml`; environment distributions are in the other Part 3 configuration files.

### 3.1 Agents

All methods use a fixed body (`body_option: b`):

| Trait | Value | Bounds used to scale the input |
|---|---|---|
| Maximum speed | 1.0 units/step | 0.25–2.0 |
| Size | 1.0 | 0.5–2.0 |
| Sensing range | 10 units | 4–20 |

**Sensors: 30 inputs.** Eight 45° sectors cover the surroundings relative to the agent's heading. Sector 0 is centred on the heading; subsequent sectors run counter-clockwise. Each sector measures food, other living agents and walls as `1 − distance / sensing range`, or zero if none is in range. Food and agent distances are between centres; wall distance is measured along the sector centre line. The remaining inputs are energy divided by birth energy, clipped to [0, 2]; the two previous squashed actions; and the three body traits scaled to [0, 1]. Dead slots return zeros.

**Actions.** The network's two outputs determine turn rate `(π/4) × tanh(a₁)` and speed `maximum speed × (tanh(a₂) + 1) / 2`. Evaluation uses deterministic network outputs. PPO samples a Gaussian action distribution during training, before the environment applies these bounds.

**Energy.** Each living agent pays `0.5 + size³ × speed² + 0.05 × sensing range` per step. A food item gives 40 energy; birth or episode-start energy is 100. An agent dies at energy ≤ 0. With the fixed body at full speed, the energy cost is 2 per step.

**Eating.** A food item is reachable when its centre is within the agent's size radius. If several agents reach one item, the agent with the lowest random draw that step receives it. Each item is eaten once; an agent can eat several items per step.

**Step order** (`src/evo2/env/world.py`):

1. Living agents turn and move; positions are clipped at the walls.
2. Agents eat at their new positions.
3. Agents alive at the start of the step pay their energy cost.
4. Agents die from depleted energy or, in ecological mode, age ≥ 1,000.
5. Eligible agents give birth in ecological mode.
6. Absent food regrows, present food drifts, and seasonal relocation occurs when due.

OE then applies its population floor, mutations and any scheduled environment change (Section 5).

### 3.2 Food and environment parameters

| Parameter | Meaning |
|---|---|
| Arena side L | Side length of the square arena |
| Food count F | Number of food items when fully grown |
| Layout | Uniform; Gaussian clusters with k centres and spread σ; or k circular patches with radius r |
| Regrowth probability ρ | Independent probability per step that an absent item returns at a newly sampled position |
| Drift speed | Distance moved per step along each item's direction, reflecting at walls |
| Seasonal period P | Interval between relocation of the layout centres and present food; zero disables relocation |

Cluster and patch centres are uniform in [0.15 L, 0.85 L]². Patch positions are uniform within their discs. Food positions are clipped to the arena. Arrays provide 64 food slots and at most five layout centres.

### 3.3 Two modes

**Ecological mode** supports 128 agent slots. An agent with energy ≥ 200 can produce one child per step, losing 120 energy; the child starts with 100 energy at a position one unit from its parent, clipped to the arena. Eligible parents are ordered randomly and matched to free slots. When slots are exhausted, remaining parents keep their energy. Agents die at energy ≤ 0 or age 1,000. OE starts with 32 agents and uses the population floor described in Section 5.

**Episodic mode** starts 16 agents at random positions with energy 100 and has no births. Dead agents remain dead for the episode. PPO restarts a world after 500 steps or when all its agents are dead. ES and evaluation execute 500-step episodes, with dead slots contributing no agent-steps. Competence is mean food eaten per starting agent.

### 3.4 Environment splits

| Split | Definition |
|---|---|
| Training (`train.yaml`) | L 40–80; F 20–60; uniform, clustered or patch layouts; k 2–5; σ 3–8; patch radius 4–10; ρ 0.01–0.05; no drift or seasons |
| Canonical (`train.yaml`) | L 60, F 40, uniform, ρ 0.03; used by the fixed-environment methods |
| Validation (`validation.yaml`) | 20 training-distribution environments, seeds 4900–4919, plus four corners combining L ∈ {40, 80} and F ∈ {20, 60}, with three clusters, σ 5 and ρ 0.03 |
| T1 (`test_split.yaml`) | Ten training-distribution environments, seeds 5900–5909 |
| T2 (`test_split.yaml`) | Ten environments: L 30–40, F 10–20, clusters with k 2–3 and σ 2–4, ρ 0.005–0.01; seeds 5910–5919 |
| T3 (`test_split.yaml`) | Ten environments with training-distribution ranges plus drift 0.05–0.3 and seasonal relocation every 100–300 steps; seeds 5920–5929 |

`load_test_split` requires `final=True`. The evaluation entry point requires `--split test --final --adapt`, completed final runs and the full test split, and logs test access. Training and tuning use the training and validation configurations.

## 4. Brains and genomes (`src/evo2/agents`)

### 4.1 Controller

Every method uses a 30 → 32 → 32 → 2 multilayer perceptron with tanh hidden activations: **2,114 parameters**, including biases. The controller has no recurrent state; the previous action is part of its observation. Hidden weights use orthogonal initialization with gain √2, output weights use gain 0.01, and biases start at zero.

PPO adds a separate 30 → 32 → 32 → 1 value network and two learned, state-independent action log standard deviations. The value head uses initialization gain 1. These additions are used during training; the evaluated controller is the actor mean.

### 4.2 OE inheritance and mutation

Each agent stores controller parameters and the fixed body traits. At both ordinary and floor births, the child receives its parent's controller plus independent Gaussian noise on every weight and bias. The selected noise standard deviations are 0.03411858603369065 for OE and 0.005074746343511773 for OE-fixed (`configs/part3/tuned.yaml`). Body traits are copied unchanged. Initial controllers use the shared initializer above, with independent keys.

## 5. Methods (`src/evo2/methods`)

| Method | Training procedure |
|---|---|
| OE | Survival and reproduction in one ecological world. Parameters are redrawn from the training distribution every 2,000 steps; agents retain their positions, clipped to the new arena, and food is regenerated. |
| OE-fixed | The same ecological process in the canonical environment. |
| PPO-fixed | Parameter-sharing PPO in 64 parallel canonical worlds; reward is food eaten per step. |
| PPO-DR | The same PPO procedure, with a new training-distribution environment at each episode reset. |
| ES-DR | `evosax.Open_ES` optimizes one controller using antithetic sampling, centred-rank shaping and Adam updates to the mean. Candidates share environment and episode randomness within each generation; score is mean food per agent. |
| ES-fixed | The ES procedure in the canonical environment, added under protocol Section 16. |

**OE population floor.** If fewer than 16 agents remain after a world step, empty slots are filled with mutated copies of parents sampled uniformly from survivors not born that step. If none remain, the parent pool is the agents alive at the start of the step. Floor offspring appear at random positions with 100 externally supplied energy. Floor births and ordinary births are counted separately. OE has no scalar fitness ranking.

**PPO.** Each agent supplies a trajectory to the shared policy. Death and episode termination end that trajectory; steps from agents already dead are masked out of every loss. PPO-fixed uses 128 rollout steps, γ = 0.95 and no reward normalization; PPO-DR uses 64 rollout steps, γ = 0.99 and reward normalization over living agents. Both use eight update epochs. Full selected settings, including learning rates, are in `configs/part3/tuned.yaml`, overlaid on `methods.yaml` by `load_methods_config`.

**ES.** The returned controller is the distribution mean. ES-DR uses 64 candidates and one episode per candidate per generation; σ = 0.08987903644893716 and learning rate = 0.0073745955277558515. ES-fixed is tuned separately; its settings are loaded with `--tuned-extra` and stored in each run's `run.json`.

**Adaptation.** The test environment stays fixed. OE continues mutation and reproduction with the floor enabled. PPO carries over actor, critic, action variance and reward-normalization statistics, with a fresh Adam state. ES initializes a new optimizer from the trained mean controller. Each uses its selected training hyperparameters.

## 6. Agent-step budgets and saved outputs

Training uses **2 × 10⁸ agent-steps per run**, and adaptation uses **2 × 10⁷ per test environment**. Every method counts agents alive at the start of each simulated step, including all PPO worlds and ES candidates. Tuning uses eight trials per method at **5 × 10⁷ agent-steps per trial**, scored on the 24 validation environments.

Training proceeds in units of one OE step, one PPO update or one ES generation. `evo2.run.chunk_size` reduces chunk size near each target; overshoot is smaller than the maximum agent-step count of one unit. Actual counts are saved. The same chunk rule is used when resuming a run.

`src/evo2/train_loop.py` saves ten policy snapshots at the first chunk boundary reaching each tenth of the training budget, together with their actual step counts. Each run also records settings, seed, source commit, hardware, wall time and progress. `checkpoint.pkl` supports resuming; `policy.pkl` and `done.json` mark completed training.

## 7. Evaluation and analysis

**Competence** is mean food per agent over eight episodic evaluations per environment and checkpoint. PPO and ES use 16 copies of one controller. OE samples 16 living agents afresh for each episode, without replacement when possible. Evaluation keys are shared across methods and runs; adaptation randomness comes from each run's seed.

**Zero-shot evaluation** measures competence without learning and compares it with a random policy on the same episodes. The random policy uses the fixed body and draws turn and speed uniformly within their bounds. Persistence is measured separately: 32 sampled or copied agents run for 5,000 ecological steps with ordinary births, no mutation, no floor and no environment change. Outputs include extinction and survival time.

**Adaptation evaluation** contains the zero-shot value at point 0 and ten measurements after successive tenths of the adaptation budget. Each point records its actual agent-step count.

**Normalization** is `(competence − random) / (best − random)`. The denominator uses the best competence among all five core methods, final runs and checkpoints in that environment. ES-fixed uses the same core denominator. Environments with best ≤ random are excluded from normalized measures.

The primary metric is trapezoidal area under the normalized adaptation curve on T2 and T3, using nominal budget fractions 0, 0.1, …, 1 as the x-axis. Scores are aggregated by interquartile mean over run–environment pairs. Confidence intervals use 50,000 whole-run bootstrap resamples, preserving all environments of each sampled run. Secondary measures include zero-shot and final competence, time to 90% of pre-shift competence, persistence and OE diversity. Exact comparisons, exclusions and decision rules are in protocol Sections 7–8 and 16; implementations are in `src/evo2/metrics.py` and `src/evo2/analysis.py`.

## 8. Seeds

| Use | Seeds |
|---|---|
| Unit tests and smoke runs | 3900–3999 |
| Pilot runs | 3000–3099 |
| Pilot evaluation | 3100–3199 |
| Tuning | 4000–4799; trial t of method m uses 4000 + 100m + t |
| Bootstrap and diversity probes | 4800 and 4801 |
| Validation environments and episodes | 4900–4999 |
| Final runs | 5000–5009 for each method |
| Test environments and episodes | 5900–5999 |

`src/evo2/seeds.py` validates purpose-specific ranges and folds named stream identifiers into JAX keys. Streams separate environment generation, evaluation, tuning draws, training, adaptation and diversity probes. Test environment j uses episode seed 5930 + j, in T1, T2, T3 order.

## 9. Dependencies

The neural environment uses JAX; PPO adapts PureJaxRL, ES uses evosax 0.3.2, and IQM and probability-of-improvement calculations use rliable 1.2.0. The whole-run bootstrap is implemented locally. Training dependencies are listed in `environments/historical_requirements_part3.txt`; CPU setup is in [REPRODUCE.md](../REPRODUCE.md). Credits and license notices are in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
