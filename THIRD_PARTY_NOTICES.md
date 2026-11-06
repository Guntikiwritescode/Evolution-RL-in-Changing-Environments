# Third-party attribution

- The trait simulator (`src/evosim/`) is a Python port of the **MinuteLabs.io Evolution Simulator** by Jasper Palfree, pinned at commit `55af956`: https://github.com/minutelabsio/evolution-simulator. Its GPL version 3 terms are preserved in the root `LICENSE`. The energy formula and two-food home-return rule are credited to Primer's *Simulating Natural Selection* (Justin Helps, 2018). The original Rust wrapper is provided in `reference/s1ref/`.
- PPO (`src/evo2/methods/ppo.py`) adapts **PureJaxRL** by Chris Lu, commit `31756b1`, `purejaxrl/ppo_continuous_action.py`: https://github.com/luchris429/purejaxrl. The Apache 2.0 license is retained at `third_party_licenses/PureJaxRL-Apache-2.0.txt`. Modification notes remain at the top of the source. The pinned upstream root listing was checked during preparation; it has LICENSE and no separate root NOTICE file.
- The implementation uses evosax's OpenAI-ES and other installed dependencies. Their versions are recorded in `environments/`; dependency packages are obtained separately and retain their own licenses.

The root GPL software license is preserved from the existing project. This document does not newly license the manuscript or data, and does not imply an institutional affiliation or endorsement by the upstream authors.

Local scientific/reporting changes relative to the inspected project revision are documented in `provenance/code_changes.json`. These preparation changes date to 5–6 November 2026. They correct tied-sample metadata and reporting; the saved experimental outcomes are unchanged.
