# Publication figure guide

Figure sources and captions. Rebuild: `python scripts/report/prepare_publication.py`. PDFs and SVGs are vector outputs; PNGs are 300 dpi. Shared style: `publication_style.py`. The manuscript uses seven of these ten figures. Neural bootstrap intervals resample training runs with fixed environments and normalization anchors; they exclude tuning and environment-sampling uncertainty. See the manuscript for sensitivity analyses.

## fig01_overview

Overview of the trait and neural experiments. Counts denote seeds or complete training runs. The neural contrast uses adaptation AUC and a nominal 95% whole-run bootstrap interval (50,000 draws). ES-fixed is a later extension.

Data: `results/publication/fig01_overview.json`.

## fig02_primary

Normalized adaptation AUC on T2/T3. Points show pooled IQMs across ten training runs and twenty environments; bars show nominal 95% whole-run bootstrap intervals (50,000 draws, seed 4800). Asterisk: ES-fixed extension.

Data: `results/publication/fig02_primary.json`.

## fig03_variation

Top: T2/T3 adaptation AUC, with run means (circles), pooled IQMs (diamonds) and nominal 95% whole-run intervals (50,000 draws). Bottom: T1 zero-shot run means and their mean (squares), with descriptive 95% intervals (20,000 whole-run draws, seed 20261105). A3 labels the pooled-IQM contrast; lower OE/PPO panels are descriptive. Each configuration was tuned separately. Asterisk: ES-fixed extension.

Data: `results/publication/fig03_variation.json`.

## fig04_curves

Top: validation competence above random during training. Bottom: normalized test competence during adaptation. Lines average ten runs; shading gives pointwise 95% whole-run percentile intervals (20,000 draws, seed 20261105). Axes use nominal living-agent steps; training observations begin at 20 million steps.

Data: `results/publication/fig04_curves.json`.

## fig05_profiles

Left: fraction above each AUC threshold across ten runs and twenty T2/T3 environments. Right: fraction of 100 cross-run comparisons in which the row method mean exceeds the column method mean, with ties counting one half. Asterisk: ES-fixed extension.

Data: `results/publication/fig05_profiles.json`.

## fig06_reconstruction

Calibrated reconstruction under the paper-described conditions. Solid lines show means over thirty ET trajectories; shading gives pointwise 95% whole-run percentile intervals (20,000 draws, seed 20261105). Dashed/dotted lines are approximate digitized curves from the original paper. All thirty reconstructed GS runs went extinct during training. The reconstruction uses different conditions from the recovered original implementation; its records are shown in figS02_original_archive.

Data: `results/publication/fig06_reconstruction.json`.

## fig07_trait_experiment

Eight groups across four arms, with 200 seeds per group. Top: mean late-B population (P1), including zero-population generations, with descriptive 95% whole-seed percentile intervals (20,000 draws). Bottom: terminal extinction proportion (P2), with Wilson 95% intervals. The contrast analyses use studentized P1 and Newcombe P2 intervals. Shared seeds couple groups and arms.

Data: `results/publication/fig07_trait_experiment.json`.

## fig08_long_training

Three exploratory OE runs trained to one billion steps, alongside the ten original 200-million-step runs (gray). Gain is the mean of checkpoints 9-10 minus the mean of checkpoints 1-2. None exceeds the planned 4.3-point threshold, which was based on an unreliable noise estimate. These are fresh runs, not continuations of the original runs.

Data: `results/publication/fig08_long_training.json`.

## figS01_diagnostics

Left/middle: diversity among sixteen sampled agents per training snapshot, averaged over ten runs. Weight diversity is mean pairwise Euclidean parameter distance; behavior diversity is mean action distance on 256 fixed validation probes. Right: extinction frequencies in the separate persistence assay without floor support. Asterisk: ES-fixed extension.

Data: `results/publication/figS01_diagnostics.json`.

## figS02_original_archive

Historical CSV records for thirty runs per group (seeds 1000-1029). Panels A/B show generation-end living agents, carrying extinction forward as zero; shading gives pointwise 95% whole-run percentile intervals (20,000 draws, seed 20261105). Panel C shows the surviving-run fraction. ET persists through all fifty transfer generations; every GS run is extinct by generation 26. The original code uses different transfer conditions from the report and disables predation through an inverted guard.

Data: `results/publication/figS02_original_archive.json`.
