# Supplement preflight

All four full-size fixtures pass native versus NUMBA_OPT=3 log-density and
gradient comparisons at the initial point, a fixed perturbation and a
near-unit-persistence point. Maximum absolute gradient differences are below
4.7e-7, within the existing comparison tolerances. Regenerated arrays match
the sealed fixture arrays; fits consume the sealed arrays themselves.

The [one-factor prior check](prior.json) uses only January 2010--December 2018
counts. Its reference-effort median is 68 (95% trajectory-bootstrap interval
65--72), 99.9th percentile 25,650 (20,459--32,858), and zero probability 0.00911
(0.00814--0.00997). The predeclared gates pass. The longitudinal q99.9 is 50,454
and passes its separate effort-adjusted envelope gate. This is prior plausibility,
not posterior validation. Raw prior draws remain in the local audit artifact.

The fixtures and declared sampling budgets are checked separately from posterior
diagnostics. The masked case retains its original prior anchors; the matched
case uses a one-factor generating law and the declared count/future-state RNGs.
Reduced cases retain their two-factor generating truth. The [test record](tests.json)
contains 39 passing distinct tests: ten new guards and 29 regression tests.
The known BLAS performance warning is retained. The final guard rerun follows
a serialization fix that records nonfinite failed diagnostics as null, preserving
the failure rather than raising a JSON-writing error. No statistical gate changed.
No fit result is part of this preflight record.
