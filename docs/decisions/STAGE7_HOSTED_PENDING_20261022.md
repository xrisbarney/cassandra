# Hosted execution of pending synthetic cases

Prepared 2026-10-22. **Verification only; fitting is not yet activated.**
The corresponding [plan](STAGE7_HOSTED_PENDING_20261022.json) refuses fitting
until its status is explicitly changed after successful hosted verification.

Eight cases from the existing 43-primary/12-sensitivity design have not started
locally: primary fallback seed 7103; representative and fallback seed 7101 for
`level_tight`, `level_wide` and `dynamics_tight`; and the `dynamics_wide`
representative. These are existing cases, not extra replications or a new
selection grid. The four active local cases continue their prescribed schedules.

The repository is public and Actions is enabled. The prepared workflow uses
standard `windows-2022` runners, CPython 3.12.0 matching the local interpreter,
uv 0.13.4, the unchanged lock, and the same checksum-pinned MinGW 14.2 compiler.
It uses NUMBA_OPT=0, one numerical thread, four sequential chains, joint level
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
transferred. Actions receive only the repository and generate synthetic fixtures.

The workflow has read-only repository permission and no automatic push/PR
trigger. The first dispatch is verification-only. If activated, each case must
finish its full prescribed schedule or be preserved as an interrupted execution;
the hosted job time limit is not a statistical failure criterion. A terminal
statistical failure is still a completed experiment, not an execution error.
The sampling step has a 320-minute execution limit inside a 355-minute job,
leaving time to upload partial evidence if execution times out. Such a case
remains unfinished and cannot be counted as an exhausted diagnostic schedule.
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
