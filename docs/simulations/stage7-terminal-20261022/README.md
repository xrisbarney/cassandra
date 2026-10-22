# Final diagnostic outcomes

See the [terminal report](../../STAGE7_TERMINAL_REPORT.md) and
[independent audit](diagnostics.json). Both attempts stopped at their external
1,200-second limits. Each completed one chain and part of a second. No complete
four-chain posterior or partial posterior-draw archive exists; the preserved
files are execution evidence, not posterior samples.

Each case directory contains its run provenance, final periodic heartbeat,
supervisor exit record and sampling log. Heartbeat counts are lower bounds at
termination. `diagnostics.json` hashes these records and verifies the frozen
attempt budgets and source activation. Its `diagnostics_independently_recomputed`
flags are false because no complete posterior exists. There is no R-hat, ESS,
BFMI or recovery result to impute from progress alone.

The [pre-fit audit](../../checks/stage7-terminal-20261022/README.md) and full-size
coordinate verification passed. Three execution-guard tests passed. The
[final checks](final-checks.json) verify lint, lock, configuration, panel and
both source activations. No prior amendment or subsequent candidate validation
attempt was activated. The planned matched replications remain unused.

The terminal outcome is not admitted under the frozen gates and bounded
resources, with no more convergence repairs in this experiment. The per-attempt
cap was insufficient to obtain the required posterior in this local execution;
it is not evidence that the coordinate change cannot work. The original Stage 7
and supplement results remain separate, unchanged records. Held-out scoring is
still disabled.
