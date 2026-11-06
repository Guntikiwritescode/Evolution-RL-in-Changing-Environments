# Third-party attribution

- The trait simulator (`src/evosim/`) is a Python port of the **MinuteLabs.io Evolution Simulator** by Jasper Palfree, pinned at commit `55af956`: https://github.com/minutelabsio/evolution-simulator. Its GPL version 3 terms are preserved in the root `LICENSE`. The energy formula and two-food home-return rule are credited to Primer's *Simulating Natural Selection* (Justin Helps, 2018). The original Rust wrapper is provided in `reference/s1ref/`.
- PPO (`src/evo2/methods/ppo.py`) adapts **PureJaxRL** by Chris Lu, commit `31756b1`, `purejaxrl/ppo_continuous_action.py`: https://github.com/luchris429/purejaxrl. The Apache 2.0 license is retained at `third_party_licenses/PureJaxRL-Apache-2.0.txt`. Modification notes remain at the top of the source.
- The implementation uses evosax's OpenAI-ES and other installed dependencies. Their versions are recorded in `environments/`; dependency packages are obtained separately and retain their own licenses.

