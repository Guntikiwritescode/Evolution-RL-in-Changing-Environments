# Part 3 main experiment: results

We compared OE, OE-fixed, PPO-fixed, PPO-DR and ES-DR on three test suites. The primary outcome was normalized adaptation AUC on T2 and T3. AUC includes starting competence and performance during adaptation.

Runs per method: ES-DR 10; OE 10; OE-fixed 10; PPO-DR 10; PPO-fixed 10. Confidence intervals use 50,000 whole-run bootstrap resamples.

## Registered comparisons

| Hypothesis | Prediction | Test | Verdict |
|---|---|---|---|
| H1 | OE > PPO-DR in adaptation AUC (T2, T3) | 8.1 | against |
| H2-OE | OE > OE-fixed | S2 | evidence against |
| H2-PPO | PPO-DR > PPO-fixed | S3 | not supported |
| H2b | variety vs method (no direction) | 8.3 | method matters more |
| H3 | OE > ES-DR | S1 | evidence against |
| H4 | zero-shot: no direction; final: OE > PPO-DR | S4, S5 | S4 difference (Y higher); S5 evidence against |
| H5 | T1 (in distribution): no direction | S6 | difference (Y higher) |
| H6 | OE goes extinct less often than PPO-DR | S7 | evidence against |

## Adaptation AUC: OE versus PPO-DR (T2 and T3; H1)

D = IQM(OE) − IQM(PPO-DR) = -0.559, 95% CI [-0.609, -0.520] (margin 0.05). P(OE > PPO-DR) on per-run means: 0.000.

| Method | IQM of the primary metric | 95% CI |
|---|---|---|
| OE | 0.291 | [0.242, 0.329] |
| OE-fixed | 0.412 | [0.376, 0.443] |
| PPO-fixed | 0.840 | [0.831, 0.848] |
| PPO-DR | 0.850 | [0.844, 0.855] |
| ES-DR | 0.878 | [0.873, 0.882] |

## Secondary comparisons (one Holm family, α = 0.05)

Two-sided Mann-Whitney U on per-run scores; P(X > Y) = U ÷ (n_X n_Y); IQM difference with its whole-run bootstrap CI.

| | X vs Y | Measure | Sets | P(X > Y) | p | p (Holm) | IQM difference [CI] | Predicted | Result |
|---|---|---|---|---|---|---|---|---|---|
| S1 | OE vs ES-DR | primary | T2, T3 | 0.000 | 0.0001827 | 0.001102 | -0.587 [-0.636, -0.548] | X > Y | evidence against |
| S2 | OE vs OE-fixed | primary | T2, T3 | 0.090 | 0.002202 | 0.004404 | -0.121 [-0.180, -0.068] | X > Y | evidence against |
| S3 | PPO-DR vs PPO-fixed | primary | T2, T3 | 0.570 | 0.6232 | 0.6232 | 0.010 [-0.000, 0.020] | X > Y | not supported |
| S4 | OE vs PPO-DR | zero_shot | T2, T3 | 0.000 | 0.0001827 | 0.001102 | -0.465 [-0.557, -0.407] | none | difference (Y higher) |
| S5 | OE vs PPO-DR | final | T2, T3 | 0.000 | 0.0001827 | 0.001102 | -0.603 [-0.649, -0.564] | X > Y | evidence against |
| S6 | OE vs PPO-DR | primary | T1 | 0.000 | 0.0001827 | 0.001102 | -0.537 [-0.595, -0.480] | none | difference (Y higher) |
| S7 | OE vs PPO-DR | extinct | T2, T3 | 1.000 | 0.0001575 | 0.001102 | 0.375 [0.250, 0.483] | X < Y | evidence against |

S7's IQM difference is taken over the per-run extinction fractions (one value per run). "All tied" means every per-run score was equal, so the test cannot reject (p = 1).

## Environmental variation and method (H2b)

- variety: -0.056 [-0.085, -0.028]
- method: -0.494 [-0.524, -0.467]
- interaction: -0.131 [-0.191, -0.076]
- delta: -0.438 [-0.474, -0.406]

Δ = |variety| − |method|. These 2 × 2 contrasts use the primary metric's IQMs and compare separately tuned configurations. The interaction interval excludes zero, indicating that the effect of environmental variation differs between methods. The interval for Δ is sensitive to effects near zero.

## Descriptive results

| Method | primary T1 | primary T2 | primary T3 | zero-shot norm. (T2, T3) | final norm. (T2, T3) | reached 90% of pre-shift (T2, T3) | median time if reached | extinction T1 / T2 / T3 |
|---|---|---|---|---|---|---|---|---|
| OE | 0.396 | 0.321 | 0.276 | 0.367 | 0.250 | 43 of 200 | 0.100 | 0.48 / 0.94 / 0.76 |
| OE-fixed | 0.530 | 0.386 | 0.451 | 0.536 | 0.349 | 43 of 200 | 0.000 | 0.30 / 0.93 / 0.48 |
| PPO-fixed | 0.922 | 0.729 | 0.909 | 0.819 | 0.814 | 40 of 200 | 0.000 | 0.10 / 0.95 / 0.14 |
| PPO-DR | 0.929 | 0.761 | 0.886 | 0.811 | 0.816 | 40 of 200 | 0.000 | 0.03 / 0.83 / 0.15 |
| ES-DR | 0.923 | 0.874 | 0.869 | 0.783 | 0.880 | 40 of 200 | 0.000 | 0.03 / 0.89 / 0.22 |

Time to 90% compares raw competence with the run's own mean T1 zero-shot competence, so it depends on how much food each test environment holds. Environments where it was reached (number of runs): ES-DR T3-2 (10), T3-3 (10), T3-4 (10), T3-6 (10); OE T3-1 (7), T3-2 (9), T3-3 (6), T3-4 (9), T3-6 (10), T3-7 (2); OE-fixed T3-1 (4), T3-2 (10), T3-3 (9), T3-4 (10), T3-6 (10); PPO-DR T3-2 (10), T3-3 (10), T3-4 (10), T3-6 (10); PPO-fixed T3-2 (10), T3-3 (10), T3-4 (10), T3-6 (10).

Means over runs and environments (the tested measures above use IQMs and per-run scores). Time to 90% is counted on T2 and T3. Normalization uses the random policy's competence on the zero-shot episodes; adaptation points 1–10 are measured on other episodes of the same environment (the same for every method and run, PREREG 7.3).

**Persistence, steps survived** (mean over runs and environments, of 5,000): ES-DR T1 4963, T2 1274, T3 4151 (births during the test: 497 / 16 / 335); OE T1 3188, T2 776, T3 1893 (births during the test: 300 / 7 / 103); OE-fixed T1 4019, T2 1004, T3 3150 (births during the test: 387 / 11 / 232); PPO-DR T1 4945, T2 1413, T3 4430 (births during the test: 411 / 17 / 314); PPO-fixed T1 4744, T2 884, T3 4439 (births during the test: 505 / 12 / 362). Raw zero-shot competence per environment (mean over runs) and the random policy's are in `analysis.json`.

**OE populations.** Mean birth rates during training. Floor births are copies added to maintain the minimum population.

| Method | births per 1,000 steps | floor births per 1,000 steps | floor share of new agents | steps with every agent dead before the floor refilled |
|---|---|---|---|---|
| OE | 46.6 | 64.5 | 0.58 | 0 |
| OE-fixed | 28.7 | 41.7 | 0.59 | 0 |

During adaptation (per test environment, mean over runs; T1 / T2 / T3):

| Method | births | floor births | steps with every agent dead before the floor refilled |
|---|---|---|---|
| OE | 44973 / 1200 / 33259 | 81177 / 253243 / 124389 | 0 / 0 / 0 |
| OE-fixed | 47142 / 1013 / 40610 | 68664 / 250011 / 103768 | 0 / 0 / 0 |

**Learning curves** (validation score at each tenth of training, mean over runs):

| Method | 1/10 | 2/10 | 3/10 | 4/10 | 5/10 | 6/10 | 7/10 | 8/10 | 9/10 | 10/10 |
|---|---|---|---|---|---|---|---|---|---|---|
| OE | 9.48 | 10.48 | 10.67 | 10.92 | 10.60 | 10.07 | 9.52 | 10.32 | 9.56 | 10.61 |
| OE-fixed | 16.87 | 18.37 | 18.15 | 17.99 | 17.61 | 16.74 | 16.29 | 16.48 | 17.03 | 16.21 |
| PPO-fixed | 27.17 | 27.11 | 27.18 | 27.12 | 27.11 | 27.10 | 27.20 | 27.13 | 27.16 | 27.17 |
| PPO-DR | 26.44 | 26.43 | 26.39 | 26.33 | 26.27 | 26.33 | 26.40 | 26.31 | 26.30 | 26.43 |
| ES-DR | 21.23 | 23.73 | 24.27 | 24.77 | 24.80 | 25.14 | 25.33 | 25.50 | 25.46 | 25.64 |

**Diversity at the end of training** (16 sampled agents; mean over runs): OE: weights 5.947, traits 0.000, behavior 0.144; OE-fixed: weights 0.835, traits 0.000, behavior 0.056. PPO-fixed, PPO-DR and ES-DR each deploy one policy. All methods use the default body, so trait diversity is zero.

**Diversity over training** (OE snapshots, mean over runs, tenths 1–10):

| Method | measure | 1/10 | 2/10 | 3/10 | 4/10 | 5/10 | 6/10 | 7/10 | 8/10 | 9/10 | 10/10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OE | weights | 5.543 | 5.889 | 6.076 | 5.853 | 6.896 | 5.728 | 5.954 | 5.682 | 6.055 | 6.042 |
| OE | traits | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| OE | behavior | 0.257 | 0.193 | 0.185 | 0.180 | 0.188 | 0.162 | 0.162 | 0.152 | 0.152 | 0.146 |
| OE-fixed | weights | 0.816 | 0.900 | 0.851 | 0.775 | 0.917 | 0.812 | 0.847 | 0.810 | 0.813 | 0.830 |
| OE-fixed | traits | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| OE-fixed | behavior | 0.081 | 0.088 | 0.081 | 0.059 | 0.073 | 0.062 | 0.059 | 0.058 | 0.053 | 0.055 |

Probe observations for behavioral diversity come from the random policy playing one episode in each validation environment (seed 4801, "probe" stream).

**Adaptation curves** (normalized competence, mean over runs and environments, points 0–10):

| Method | set | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ES-DR | T1 | 0.902 | 0.933 | 0.917 | 0.914 | 0.921 | 0.930 | 0.921 | 0.913 | 0.928 | 0.940 | 0.927 |
| ES-DR | T2 | 0.711 | 0.841 | 0.858 | 0.866 | 0.908 | 0.892 | 0.895 | 0.854 | 0.938 | 0.893 | 0.883 |
| ES-DR | T3 | 0.854 | 0.861 | 0.848 | 0.875 | 0.873 | 0.883 | 0.856 | 0.864 | 0.883 | 0.876 | 0.877 |
| OE | T1 | 0.405 | 0.414 | 0.395 | 0.373 | 0.392 | 0.395 | 0.386 | 0.389 | 0.407 | 0.409 | 0.395 |
| OE | T2 | 0.438 | 0.423 | 0.380 | 0.353 | 0.359 | 0.316 | 0.284 | 0.258 | 0.259 | 0.236 | 0.234 |
| OE | T3 | 0.297 | 0.291 | 0.275 | 0.276 | 0.267 | 0.285 | 0.266 | 0.270 | 0.268 | 0.278 | 0.265 |
| OE-fixed | T1 | 0.592 | 0.576 | 0.534 | 0.524 | 0.524 | 0.519 | 0.518 | 0.505 | 0.528 | 0.524 | 0.509 |
| OE-fixed | T2 | 0.529 | 0.493 | 0.448 | 0.407 | 0.424 | 0.367 | 0.355 | 0.324 | 0.323 | 0.310 | 0.290 |
| OE-fixed | T3 | 0.544 | 0.498 | 0.477 | 0.459 | 0.443 | 0.448 | 0.433 | 0.432 | 0.421 | 0.423 | 0.408 |
| PPO-DR | T1 | 0.924 | 0.943 | 0.938 | 0.916 | 0.928 | 0.933 | 0.927 | 0.915 | 0.931 | 0.940 | 0.924 |
| PPO-DR | T2 | 0.724 | 0.755 | 0.747 | 0.746 | 0.802 | 0.755 | 0.775 | 0.712 | 0.806 | 0.777 | 0.748 |
| PPO-DR | T3 | 0.898 | 0.895 | 0.871 | 0.901 | 0.889 | 0.889 | 0.868 | 0.879 | 0.893 | 0.885 | 0.884 |
| PPO-fixed | T1 | 0.916 | 0.936 | 0.928 | 0.911 | 0.917 | 0.921 | 0.921 | 0.912 | 0.920 | 0.932 | 0.920 |
| PPO-fixed | T2 | 0.718 | 0.732 | 0.719 | 0.716 | 0.766 | 0.724 | 0.747 | 0.676 | 0.754 | 0.742 | 0.712 |
| PPO-fixed | T3 | 0.920 | 0.910 | 0.902 | 0.921 | 0.905 | 0.909 | 0.893 | 0.903 | 0.916 | 0.913 | 0.915 |

## Compute

Training and evaluation hardware: NVIDIA GeForce RTX 4090, 595.91.07, 24564 MiB.

Software: JAX 0.11.2; platform cuda 12090; CUDA driver 13.2; Python 3.13.18.

Training steps show the range across runs; other columns show means. Adaptation used a budget of 20,000,000 agent-steps per environment; adaptation wall time covers all 30 environments.

| Method | Training agent-steps | Training h | Adaptation agent-steps per environment | Adaptation h |
|---|---|---|---|---|
| OE | 200,000,003–200,000,031 | 0.37 | 20,000,010 | 1.01 |
| OE-fixed | 200,000,001–200,000,016 | 0.40 | 20,000,010 | 1.01 |
| PPO-fixed | 200,015,717–200,118,624 | 0.02 | 20,039,595 | 0.14 |
| PPO-DR | 200,002,202–200,043,807 | 0.05 | 20,017,188 | 0.20 |
| ES-DR | 200,002,830–200,312,317 | 0.01 | 20,144,256 | 0.02 |

**Per-run compute.** Each cell gives training agent-steps / training hours / adaptation hours (all 30 environments).

| Seed | OE | OE-fixed | PPO-fixed | PPO-DR | ES-DR |
|---|---|---|---|---|---|
| 5000 | 200,000,031 / 0.38 / 0.98 | 200,000,006 / 0.41 / 0.96 | 200,054,994 / 0.02 / 0.15 | 200,030,214 / 0.05 / 0.21 | 200,020,120 / 0.01 / 0.02 |
| 5001 | 200,000,015 / 0.38 / 0.99 | 200,000,012 / 0.40 / 0.95 | 200,118,624 / 0.02 / 0.14 | 200,042,853 / 0.05 / 0.20 | 200,276,575 / 0.01 / 0.02 |
| 5002 | 200,000,020 / 0.37 / 1.00 | 200,000,011 / 0.42 / 0.99 | 200,015,717 / 0.02 / 0.14 | 200,043,807 / 0.04 / 0.22 | 200,312,317 / 0.01 / 0.02 |
| 5003 | 200,000,004 / 0.39 / 0.99 | 200,000,007 / 0.41 / 0.98 | 200,042,971 / 0.02 / 0.14 | 200,033,396 / 0.05 / 0.20 | 200,002,830 / 0.01 / 0.02 |
| 5004 | 200,000,017 / 0.37 / 0.96 | 200,000,012 / 0.39 / 1.05 | 200,023,509 / 0.02 / 0.14 | 200,010,407 / 0.05 / 0.20 | 200,201,878 / 0.01 / 0.02 |
| 5005 | 200,000,012 / 0.38 / 0.99 | 200,000,001 / 0.37 / 1.00 | 200,021,436 / 0.02 / 0.14 | 200,015,128 / 0.05 / 0.20 | 200,052,709 / 0.01 / 0.02 |
| 5006 | 200,000,003 / 0.38 / 1.06 | 200,000,016 / 0.38 / 1.02 | 200,084,263 / 0.02 / 0.14 | 200,024,669 / 0.05 / 0.18 | 200,057,893 / 0.01 / 0.02 |
| 5007 | 200,000,004 / 0.37 / 1.07 | 200,000,015 / 0.38 / 1.03 | 200,072,327 / 0.02 / 0.14 | 200,011,167 / 0.05 / 0.18 | 200,216,359 / 0.01 / 0.02 |
| 5008 | 200,000,017 / 0.35 / 1.03 | 200,000,013 / 0.42 / 1.14 | 200,028,121 / 0.02 / 0.14 | 200,002,202 / 0.05 / 0.19 | 200,118,196 / 0.01 / 0.02 |
| 5009 | 200,000,012 / 0.34 / 1.04 | 200,000,004 / 0.46 / 0.99 | 200,027,878 / 0.02 / 0.13 | 200,023,408 / 0.05 / 0.18 | 200,181,324 / 0.01 / 0.02 |

## Figures

The figures include the later ES-fixed comparison where indicated.

- [Primary performance](../figures/publication/fig02_primary.png): IQM of normalized adaptation AUC with nominal 95% whole-run bootstrap intervals.
- [Fixed and varied training](../figures/publication/fig03_variation.png): adaptation and zero-shot comparisons for the separately tuned configurations.
- [Training and adaptation curves](../figures/publication/fig04_curves.png): validation and test means, with pointwise 95% whole-run bootstrap intervals.
- [Performance profiles and probability summaries](../figures/publication/fig05_profiles.png): T2/T3 AUC distributions and cross-run comparisons.
- [Population diagnostics](../figures/publication/figS01_diagnostics.png): weight and behavioral diversity over training and extinction in the separate persistence assay.

Sample counts and interval definitions are in [FIGURE_GUIDE.md](FIGURE_GUIDE.md).

