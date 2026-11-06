# Part 3 tuning results

Recorded tuning summary. Trial records, scores and selected settings accompany this report. Protocol: PREREG_part3.md Section 5. The chosen settings are in `configs/part3/tuned.yaml`, written by `tune.py --write-tuned`.

Each trial trained for 5 × 10⁷ agent-steps (a quarter of the final budget) and was scored on the 24 validation environments only: mean of (competence − random-policy competence), zero-shot, 8 episodes. The highest score was chosen; ties go to the lower trial number. One trial per setting, so a score mixes the setting's effect with run-to-run luck.

## OE

Commit 732a643 (clean: True); hardware: NVIDIA GeForce RTX 4090, 595.91.07, 24564 MiB.

| Trial | seed | WEIGHT_SD | score | chosen |
|---|---|---|---|---|
| 0 | 4000 | 0.015 | 9.910 |  |
| 1 | 4001 | 0.0276 | 13.953 |  |
| 2 | 4002 | 0.0341 | 15.333 | yes |
| 3 | 4003 | 0.0236 | 9.750 |  |
| 4 | 4004 | 0.00736 | 12.659 |  |
| 5 | 4005 | 0.00635 | 14.276 |  |
| 6 | 4006 | 0.00711 | 11.278 |  |
| 7 | 4007 | 0.0193 | 11.378 |  |

## OE-fixed

Commit 732a643 (clean: True); hardware: NVIDIA GeForce RTX 4090, 595.91.07, 24564 MiB.

| Trial | seed | WEIGHT_SD | score | chosen |
|---|---|---|---|---|
| 0 | 4100 | 0.0452 | 11.463 |  |
| 1 | 4101 | 0.0434 | 6.172 |  |
| 2 | 4102 | 0.00788 | 15.161 |  |
| 3 | 4103 | 0.041 | 8.040 |  |
| 4 | 4104 | 0.0238 | 12.979 |  |
| 5 | 4105 | 0.0204 | 12.434 |  |
| 6 | 4106 | 0.0117 | 13.452 |  |
| 7 | 4107 | 0.00507 | 15.274 | yes |

## PPO-fixed

Commit 732a643 (clean: True); hardware: NVIDIA GeForce RTX 4090, 595.91.07, 24564 MiB.

| Trial | seed | ENT_COEF | GAMMA | LR | NORMALIZE_REWARD | NUM_STEPS | UPDATE_EPOCHS | score | chosen |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4200 | 0 | 0.99 | 0.000136 | False | 32 | 2 | 27.084 |  |
| 1 | 4201 | 0 | 0.95 | 0.000706 | False | 128 | 8 | 27.450 | yes |
| 2 | 4202 | 0 | 0.95 | 0.000118 | True | 64 | 2 | 26.643 |  |
| 3 | 4203 | 0.01 | 0.99 | 0.000266 | True | 128 | 8 | 27.134 |  |
| 4 | 4204 | 0.01 | 0.99 | 0.000991 | True | 64 | 2 | 26.359 |  |
| 5 | 4205 | 0.01 | 0.95 | 0.000331 | True | 64 | 4 | 27.134 |  |
| 6 | 4206 | 0.01 | 0.95 | 0.000292 | True | 128 | 2 | 26.810 |  |
| 7 | 4207 | 0.01 | 0.95 | 0.000117 | False | 64 | 4 | 26.932 |  |

## PPO-DR

Commit 732a643 (clean: True); hardware: NVIDIA GeForce RTX 4090, 595.91.07, 24564 MiB.

| Trial | seed | ENT_COEF | GAMMA | LR | NORMALIZE_REWARD | NUM_STEPS | UPDATE_EPOCHS | score | chosen |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4300 | 0 | 0.99 | 0.000263 | True | 64 | 4 | 26.404 |  |
| 1 | 4301 | 0.001 | 0.995 | 0.000177 | True | 32 | 2 | 25.841 |  |
| 2 | 4302 | 0.001 | 0.99 | 0.000307 | False | 32 | 2 | 26.478 |  |
| 3 | 4303 | 0.001 | 0.95 | 0.000899 | False | 32 | 8 | 26.441 |  |
| 4 | 4304 | 0.001 | 0.99 | 0.000248 | True | 32 | 8 | 25.978 |  |
| 5 | 4305 | 0 | 0.95 | 0.000248 | False | 32 | 8 | 26.841 |  |
| 6 | 4306 | 0.01 | 0.99 | 0.00011 | True | 32 | 4 | 26.556 |  |
| 7 | 4307 | 0 | 0.99 | 0.00018 | True | 64 | 8 | 26.928 | yes |

## ES-DR

Commit 732a643 (clean: True); hardware: NVIDIA GeForce RTX 4090, 595.91.07, 24564 MiB.

| Trial | seed | EPISODES | LR | POP_SIZE | SIGMA | score | chosen |
|---|---|---|---|---|---|---|---|
| 0 | 4400 | 4 | 0.00751 | 64 | 0.0316 | 20.047 |  |
| 1 | 4401 | 2 | 0.00402 | 32 | 0.091 | 22.040 |  |
| 2 | 4402 | 4 | 0.0044 | 32 | 0.0343 | 18.384 |  |
| 3 | 4403 | 4 | 0.00652 | 32 | 0.0115 | 17.971 |  |
| 4 | 4404 | 4 | 0.0191 | 64 | 0.0311 | 22.527 |  |
| 5 | 4405 | 4 | 0.0057 | 32 | 0.0252 | 20.018 |  |
| 6 | 4406 | 1 | 0.00379 | 128 | 0.0254 | 18.204 |  |
| 7 | 4407 | 1 | 0.00737 | 64 | 0.0899 | 23.777 | yes |

