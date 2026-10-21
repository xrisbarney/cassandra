# Stop attempts that have irreversibly failed a retained-draw gate

Recorded while the synthetic campaign is running, before applying this runner
change to subsequent attempts. It changes no model, prior, seed, requested
sampling budget, or acceptance threshold. Existing running processes retain
their original behavior; their complete failed attempts remain on disk.

An attempt with even one retained divergence, or a retained tree depth at the
declared maximum, cannot satisfy the frozen zero-tolerance gate. More draws
cannot undo that event. The runner now terminates such an attempt immediately,
saves the failure reason, chain/draw index, completed draw counts, timing and a
state checkpoint, then uses the next already permitted retry budget and seed.
After the final retry it records a failed fit. It does not compute R-hat/ESS
from the partial chain or use the checkpoint as posterior/recovery evidence.

Warmup events do not trigger this rule. Fits that complete without those events
still face the full four-chain R-hat, bulk/tail ESS, BFMI, numerical-warning,
finite-value and zero-divergence/depth checks. An admitted fit always has its
entire requested four-chain retained sample. Slow mixing or a low BFMI cannot
trigger this shortcut because these diagnoses require the completed chains.

Thus early rejection gives the same admission decision that finishing that
attempt would give: an already observed zero-tolerance violation cannot become
a pass. This is computational rejection, not an additional scientific threshold
or a mechanism for choosing favorable datasets. Distinguish partial failed
attempts from completed failed attempts in the final audit. Report recorded
divergences/depth events as lower bounds for an early-terminated attempt.
