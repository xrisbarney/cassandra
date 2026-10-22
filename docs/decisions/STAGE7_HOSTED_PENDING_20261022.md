# Hosted execution of pending synthetic cases

Prepared and activated 2026-10-22 after full hosted OPT3 verification.
The corresponding [plan](STAGE7_HOSTED_PENDING_20261022.json) binds the exact
pending cases and inputs. A fresh verification job on the execution commit
must also pass before its fitting jobs can start.

The first Windows verification, run 39747925041 at commit db53e40, passed all
16 coordinate/guard tests, all eight fixture byte hashes and complete native/
Numba target comparisons. Its [record](../checks/stage7-hosted-verification-20261022/verification.json)
and original plan are preserved. Repeated fused evaluations took about 0.031
seconds on that host, substantially slower than the local measurement. Before
activation, compare NUMBA_OPT=0 and 3 on the same declared full-size deterministic
checks. This extends the earlier compiler-profile investigation to the hosted
environment; no profile can be selected by posterior acceptance or recovery.
Both profiles must pass the original numerical tolerances. The four active
local fits keep their existing compiler configuration and sampling schedules.

The [profile review](../checks/stage7-hosted-profile-20261022/selection.json)
selects OPT3 for the eight pending hosted cases. Its full-size median fused
evaluation is about 0.002 seconds; maximum absolute gradient discrepancy is
6.30e-8. These measurements used different fresh hosts and cache histories,
so they are not a controlled speedup estimate. The local OPT0 fits continue.
The second OPT0 job passed 23 checks but failed the exact regenerated transformed
truth hash for fallback seed 7103; no fitting ran. The OPT3 job passed all 24
checks and all original hashes. That mismatch is preserved, not overwritten.

Canonical local NPZ files are now committed for the three distinct datasets
used by these eight cases. Every fit reads these exact pinned observations and
truths, with their original hashes. Independent host regeneration is reported
separately: integer/bool arrays must be exact, floats agree to 1e-14 relative/
absolute tolerance, and every maximum difference is retained. Regenerated
floating-point arrays never replace the canonical fitting or recovery inputs.
This prevents machine-dependent generator rounding from changing an experiment.

Eight cases from the existing 43-primary/12-sensitivity design have not started
locally: primary fallback seed 7103; representative and fallback seed 7101 for
`level_tight`, `level_wide` and `dynamics_tight`; and the `dynamics_wide`
representative. These are existing cases, not extra replications or a new
selection grid. The four active local cases continue their prescribed schedules.

The repository is public and Actions is enabled. The prepared workflow uses
standard `windows-2022` runners, CPython 3.12.0 matching the local interpreter,
uv 0.13.4, the unchanged lock, and the same checksum-pinned MinGW 14.2 compiler.
It uses NUMBA_OPT=3, one numerical thread, four sequential chains, joint level
coordinates, the block solver and a diagonal metric. Priors, seeds, budgets,
observations, diagnostic thresholds and recovery requirements are unchanged.
Hardware and OS build differences will be recorded; timing comparisons must
distinguish local and hosted execution. Exact posterior draws need not coincide
across processors despite an unchanged statistical target and RNG seeds.

Before fitting, a verification job must pass independent dense/block coordinate
tests, exact byte hashes for all eight generated observation/truth fixtures,
and native/Numba complete-target and gradient comparisons on full-size count
and fallback fixtures, including near-unit persistence. The numerical source,
lock, compiler specification and original freeze hashes are pinned in the plan.
No real data, credentials from the local environment or held-out targets are
transferred. Actions receive only the repository, including the pinned synthetic
fixtures, and independently regenerate them for comparison.

The workflow has read-only repository permission and no automatic push/PR
trigger. The first dispatch is verification-only. If activated, each case must
finish its full prescribed schedule or be preserved as an interrupted execution;
the hosted job time limit is not a statistical failure criterion. A terminal
statistical failure is still a completed experiment, not an execution error.
The sampling step has a 320-minute execution limit inside a 355-minute job,
leaving time to upload partial evidence if execution times out. Such a case
remains unfinished and cannot be counted as an exhausted diagnostic schedule.
To avoid making the total three-attempt duration a single-job requirement,
hosted fitting is prepared as three successive jobs per case, one prescribed
attempt per job. Only a fully completed unsuccessful nonfinal attempt can be
sealed. Continuation verifies all artifact/source hashes, options, generated
fixtures and budgets, then starts the next original retry with its original
seed. It does not resume a partial Markov chain. Accepted or fully exhausted
cases are carried forward without refitting. Original run metadata and each
continuation's host metadata are retained; graph/preflight compilation costs
are counted once per execution process. A within-attempt timeout still requires
explicit preservation and resolution and cannot become a diagnostic failure.
Admitted saved posteriors are independently checked on the runner and again
after local transfer. Artifact hashes and actual host provenance are retained.

Hosted outputs will be downloaded and verified before they enter the local
canonical routes. Locally reserved case directories prevent the existing
coordinator from duplicating those cases; any temporary coordinator result for
a remote reservation is preserved and reconciled against the actual terminal
case results. No partial result is presented as a completed fit. Artifacts use
one-day retention and will be deleted from Actions after verified local transfer;
local evidence is preserved. No caches or larger runners are requested.

Standard hosted runner minutes are free for public repositories; artifact
storage has separate account allowances. GitHub limits hosted jobs to six hours.
These operational facts are documented in [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions)
and [Actions limits](https://docs.github.com/en/actions/reference/limits).
This is a computational scheduling decision, not a prior or evaluation amendment.
