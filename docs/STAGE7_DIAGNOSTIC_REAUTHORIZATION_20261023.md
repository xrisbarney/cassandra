# Reauthorization of the two unfinished diagnostics

User authorization, 2026-10-23: "go for it", following the recommendation to
rerun exactly the two unfinished diagnostics once with realistic wall limits and
completed-chain preservation. This supersedes the *resource closure* for this
narrow purpose. All prior records, including the timeouts, remain unchanged.
The proposed model is currently unadmitted; a final evaluation-protocol outcome
is deferred until these two attempts have been assessed.

## Unchanged experiment

Use the sealed matched one-factor fallback seed-7101 fixture, all ten categories
and 108 months, including every observed category-10 zero. Reuse the verified
joint builder and original priors. `joint_free` estimates persistence under
Uniform(-1,1); `joint_fixed` conditions on the generating value 0.65. The fixed
case is diagnostic evidence only. Four sequential chains, 2,000 warmup and
2,000 retained draws per chain, target acceptance 0.99, depth 15, sampler seed
7103 and initialization RNG 7301 remain unchanged. Keep one `pm.sample` call per
case so chain-seed allocation and initialization follow the original execution.
Do not split sampling into new calls or alter the model, priors, data or seeds.

Exactly two fresh attempts are authorized, one per case, in new reserved
directories. Previous attempts have no resumable state and are not continuations.
No automatic retry, extra seed, extra draw, prior amendment or validation-grid
expansion is authorized by this reauthorization. Stage 8 and held-out scoring
remain closed.

## Bounds informed by measured cost

Comparable completed supplement sampling calls took 2,766 and 2,119 seconds;
the latest partial joint runs took about 730 seconds to complete their first
chain, including startup. A four-chain completion therefore needs substantially
more than 1,200 seconds. Allocate **5,400 seconds (90 minutes) per attempt**,
including construction, compilation, checkpoint serialization and diagnostics.
An outer supervisor enforces the limit. Run the two cases concurrently, each
with sequential chains and one numerical thread. The global sampling deadline
is 120 minutes from the declared launch window, fixed in the activation before
either fit starts. The global cap always overrides a per-case cap. The 90-minute
allowance supplies headroom over the observed roughly 35-49 minute costs; it is
not a runtime guarantee and is not extended after results are observed.

## Preserve evidence without changing sampling

On the final callback of each completed chain (identified by draw index, not
the backend's `is_last` flag), copy the retained trace and sampler statistics,
excluding warmup. Write a one-chain NetCDF file and a SHA-256 marker only after
serialization succeeds. Preserve the original chain index. Also save its last
unconstrained point and the warning ledger available at that point; these are
audit artifacts, not a complete resumable NUTS state. Checkpointing must not
change RNG streams, sampler calls, draws or tuning. Verify this on small seeded
models before activation, including interruption after one complete chain.
Existing checkpoints or output directories may not be overwritten.

An early retained divergence/depth saturation still stops that attempt under
the unchanged rule, preserving completed chains and the failure state. A timeout
preserves completed checkpoints and progress but cannot admit an incomplete
chain set. Four complete chains remain mandatory. No extrapolated R-hat or ESS,
and no concatenation of chains across distinct attempts, is permitted.

## Review and decision

Independently recompute diagnostics from complete saved posteriors and verify
all four per-chain checkpoint arrays against the final posterior. Preserve all
failures. The unchanged gates are finite diagnostics, R-hat <1.01, bulk and
tail ESS >400, BFMI >0.3, zero retained divergences and depth saturations, and
no unresolved numerical warnings. Examine chain exploration separately before
using an admitted free-persistence posterior for recovery. Conditional recovery
from the fixed fit cannot validate a free-persistence production model.

Report whether the matched fixture is now numerically admitted, still fails,
or remains resource-incomplete. Do not turn one successful fixture into general
validation; the sealed additional replication cases remain outside this task.
Stop after these two outcomes. No persistence prior or evaluation roster is
changed automatically. Preserve the earlier budget decision as history, and
keep January 2024 through December 2025 unscored.
