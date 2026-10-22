# Final bounded convergence effort

Status, 2026-10-22: **terminal effort closed; proposed model not admitted under
the frozen gates and bounded resources**. Both diagnostic attempts timed out
before four complete chains. Stage 8 and held-out scoring remain closed.

The [terminal plan](STAGE7_TERMINAL_DIAGNOSTIC_PLAN.md) caps the entire effort at
120 minutes, with a maximum of five attempts and 20 minutes per attempt. It
predeclares a terminal not-admitted outcome, rather than another repair loop.
The [activation](decisions/STAGE7_TERMINAL_DIAGNOSTICS_20261022.json) authorizes
only the first two diagnostics; any subsequent validation/prior branch must
satisfy the plan's prerequisites. The completed four-case supplement is unchanged.

The [saved-state audit](checks/stage7-terminal-20261022/README.md) checks both
preserved divergence callback points and two prescribed nearby points each.
Independent dense SciPy reconstruction, finite differences, native PyTensor and
compiled PyTensor/Numba agree within the predeclared tolerances. Maximum absolute
native/dense log-density differences are 1.13e-8 for the matched one-factor case
and 3.10e-8 for the masked two-factor case. These local checks find no discrepancy;
the saved callback points are not complete divergent trajectories.

The matched one-factor point has persistence 0.96518 (about 28.2 months of
e-folding correlation decay) and a category-level scale of 0.00335. The masked
two-factor point includes persistence 0.999484 (about 1,937 months), with level
scale 1.30954. The evidence therefore does not support attributing both failures
solely to factor memory longer than 108 months. With proper priors the likelihood's
constant-shift invariance is not necessarily a flat posterior: the independent
calculation gives positive prior curvature along that direction, particularly
large at the matched case's tiny level scale. Joint coordinates are worth testing
without first imposing a new scientific prior.

The isolated one-factor joint builder preserves the original normalized priors,
NB likelihood and Jacobian. It leaves all sealed production and earlier Stage 7
sources unchanged. Five full-size verification points pass: initial, perturbed
and near-unit for free persistence, plus initial and perturbed for fixed
persistence. The checks include canonical reconstruction, dense density,
native/compiled gradients, independent finite differences and mapping to the
original conditional-state coordinates. Three execution-guard tests pass; they
cover duplicate attempts, expiry and closed authority. Repeated test invocations
are not additional tests. Executed audit/verifier bytes are retained alongside
their hashes; subsequent unused-import/line-wrap cleanup is disclosed there.

Both posterior diagnostics use exactly the sealed matched seed-7101 observations,
including the independently modeled fallback level and its observed zeros. One
estimates persistence under the unchanged Uniform(-1,1) prior; the other conditions
on the generating persistence 0.65. Conditioning changes the posterior and is
diagnostic evidence only, never a production-model admission. Both use four
sequential chains, 2,000 warmup and 2,000 retained draws per chain, target 0.99,
depth 15, sampler seed 7103 and initialization seed 7301. An outer process enforces
the 20-minute cap even during compilation or a long transition.

The two additional matched datasets (seeds 7102 and 7103) were generated and
sealed before any diagnostic result, with the exact generator/RNG definitions
in the terminal plan. Reconstructing the seed-7101 matched dataset reproduces
the sealed supplement arrays. No dataset selection or seed search is allowed.
These are fixed-truth replications, not SBC. They are used only if the declared
diagnostic prerequisites pass and the global budget remains.

The approximation fits to two-factor truth remain robustness checks, not clean
one-factor recovery evidence. Their failures do not prove that misspecification
caused poor mixing. No extra draws are authorized beyond the frozen final
sampling budget, and no diagnostic gate is relaxed.

## Terminal outcome

The [independent result audit](simulations/stage7-terminal-20261022/diagnostics.json)
verifies activation/source hashes, exact sampling budgets, supervisor exits and
the preserved partial files. Both supervisors reached their 1,200-second limits
at 20:08:05 UTC. No fitting process remained afterward. Exactly two attempts
were started; the three remaining possible slots were not used. The 120-minute
global ceiling was not needed to extend or restart either attempt.

| Diagnostic | Last recorded completed work | Outcome |
|---|---|---|
| Joint coordinates, free persistence | First chain: 2,000 warmup + 2,000 retained; second: 2,000 warmup + 950 retained | 20-minute timeout |
| Joint coordinates, persistence fixed at 0.65 | First chain: 2,000 warmup + 2,000 retained; second: 2,000 warmup + 1,200 retained | 20-minute timeout |

These counts are lower bounds from the last periodic heartbeat, not exact counts
at process termination. No retained-divergence stop was recorded before the
timeouts, but neither run has four complete chains. R-hat, ESS, BFMI and recovery
are therefore not reported as admission results. The outer timeout terminated
the sampler before its final posterior serialization: logs, run provenance and
heartbeats survive, **not complete posterior files or saved partial draw arrays**.
No posterior diagnostics can be reconstructed from those progress records.

The per-attempt cap chosen for this effort proved insufficient for four chains
in this local execution. That is a resource limitation, not evidence that joint
coordinates cannot converge, nor evidence that fixing persistence resolves the
problem. Neither branch establishes the prerequisite for selecting a new prior.
No persistence-prior amendment or candidate validation fit was activated. The
restricted-prior constructor option was never used by either authorized fit.
The sealed replication datasets remain unused for fitting or scoring.

Convergence repair ends here for this experiment. Keep the proposed model's
forecast row **not admitted**, and keep the failed/incomplete attempts visible.
No reduced proposed structure has been validated at the intended dimensions by
this effort. The prior, production configuration and original Stage 7 results
remain unchanged. The manuscript must distinguish a verified implementation
from an admitted posterior and an empirically validated forecast.

The next evaluation work is to complete the outstanding named comparators
(RF, ETS, BSTS-U, Vuln4Cast, B-LSTM and the Bakdash adaptation), their ensemble
checks and development selection, using the validated controls already available.
Any reduced proposed structure needs its own existing evidence at the intended
dimensions; a small synthetic success or the fixed-persistence diagnostic is
not a substitute. Before held-out scoring, record the final comparison roster
and the proposed model's not-admitted status in the evaluation protocol. No
Stage 8 fit, economic calibration or 2024-2025 score was performed here.
