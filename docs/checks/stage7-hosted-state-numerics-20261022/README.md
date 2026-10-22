# Numerical checks at two hosted failure checkpoints

The two saved visited states from hosted first attempts of `dynamics_wide /
representative / 7101` and `primary / fallback_zero / 7103` pass complete
native/Numba density and gradient comparisons under NUMBA_OPT=3. Maximum
absolute gradient differences are approximately 6.00e-8 and 8.78e-8.

Selected first/last components of each variable block also pass the previously
declared finite-difference step grid: .003, .001, .0003, .0001, .00003 and
.00001. Two adjacent estimates must agree with each other and the analytic
gradient under the original 5e-5 absolute/relative tolerance. Every estimate,
including unsuccessful individual step sizes, is preserved in the JSON records.

These are saved **visited states**, which need not equal the rejected leapfrog
proposals that caused the divergences. Passing these local checks does not
explain or waive a divergence, establish global posterior exploration, or admit
either failed attempt. The [audit](audit.json) binds the inputs, numerical
source, compiler setting and complete outputs. No model law or tolerance changed.
