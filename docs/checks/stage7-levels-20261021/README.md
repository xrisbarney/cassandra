# Anchored-level prior evidence

`audit.json` is the original completed 2,000-trajectory audit, preserved byte for
byte. The primary candidate passed the reference-effort and longitudinal gates.
Its reference median was 68 (trajectory-bootstrap 95% interval 65--72), q99.9
20,794 (17,356--25,649), and zero probability .00838 (.00746--.00919).

## Classification correction and declared repeat

The original classifier escalated a crossing confidence interval only when the
point estimate passed. The declaration requires escalation from either side.
This was corrected and regression-tested. The `dynamics_wide` point estimate
54,842 failed the ceiling, but its interval 47,487--63,216 crossed 50,000.
Its original "fail" label is therefore superseded by "unresolved".

The prescribed 8,000-trajectory repeat is in `wide-repeat.json`. Every stored
prior-variable array's first 2,000 draws matches the original exactly. q99.9 is
50,854, with interval 47,678--54,119: it remains **unresolved**, not admitted.
No further increase or threshold change was introduced. `level_wide` fails;
primary, `level_tight`, and `dynamics_tight` pass the pilot criteria.
Both wide candidates remain explicitly labelled posterior sensitivity stress
cases, never candidates admitted by the primary plausibility gate.

The R=3 + iid + kernel joint prior has reference median 144 and q99.9 147,120.
It is a diagnostic for an extension; the pilot gate does not validate it.
The e=3 highest-regime calculation is an envelope, not a joint-prior sample.

The two earlier audit-launch errors (nullable integers, then histogram inverse
shape) are retained under `artifacts/stage7/anchored-prior-v2` and `...-r1`.
The successful audit is `...-r2`; these were code errors before posterior fitting,
not prior-gate failures or changes to the declared probability laws.
