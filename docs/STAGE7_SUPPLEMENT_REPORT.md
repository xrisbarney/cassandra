# Bounded Stage 7 convergence supplement: execution record

Status, 2026-10-22: **activated after preflight; posterior results pending**.
The [declaration](STAGE7_REPAIR_SUPPLEMENT.md) fixes four cases and one attempt
per case. The [activation](decisions/STAGE7_REPAIR_ACTIVATION_20261022.json)
binds the code, fixtures, prior audit and numerical checks. Stage 8 and held-out
scoring remain closed; the original 55-case campaign remains unchanged.

The [preflight record](checks/stage7-supplement-20261022/README.md) contains four
successful full-model density/gradient comparisons, a passing one-factor prior
plausibility audit, and 39 passing distinct tests. The masked fixture preserves
the prior anchors from the original unmasked counts. The matched fixture uses
the declared one-factor law and RNG seeds. The other two reduced fits retain
the original two-factor truth and are approximation checks.

All fits use four sequential chains, 2,000 warmup and 2,000 retained draws per
chain, target acceptance 0.99 and depth limit 15. Each case has a single reserved
directory, with no overwrite or sampler retry. Hosted reruns are refused.
The existing irreversible retained-failure rule stops an ineligible attempt
early and preserves its state and progress. Partial attempts are not recovery
evidence. Final numerical admission must be followed by exploration review.

The original production configuration is unchanged. A one-factor pilot cannot
be adopted solely because a small fixture converged; the supplement's declared
adoption conditions and any resulting amendment remain pending.
