# Exact conditional Gaussian coordinates for severity

Declared 2026-10-21 before revised severity fits. Two v2 severity cases exhausted
all retries with retained divergences. Checkpoints included very small mark-noise
scales; checkpoints are not posterior samples or proof of identification failure.
The third case is still running. Keep all existing results and budgets.

Test a probability-preserving coordinate change, not a noise floor or prior
revision. Let the stacked stationary severity states have prior precision P0.
For a Gaussian proxy to the observed coarsened logit marks, let observation
matrix H, diagonal noise covariance R, and centred marks b define

    P = P0 + H' R^-1 H = L L'
    mean = P^-1 H' R^-1 b
    h = mean + L^-T u.

Missing marks contribute nothing to P or b. Use finite logits of the actual
coarsened endpoints for the proposal only. The model still evaluates the exact
censored mark likelihood. With a Flat base measure for u, add the normalized
stationary Gaussian state log density and log Jacobian `-sum(log(diag(L)))`.
All other priors are unchanged. This is an invertible transformation for every
positive noise scale and |phi|<1; including parameter-dependent normalizers and
the Jacobian preserves the joint target. For entirely interior Gaussian marks,
it standardizes their conditional Gaussian state geometry. With censoring or
additional regime structure it is a proposal-based change of coordinates, not
Gaussian replacement of the observation law.

Check the precision, full density ratio and gradients against an independent
NumPy/SciPy construction, including two factors, missing marks and endpoints.
Then try seed 7101 under the same three budgets. Choose the representation by
diagnostic admission only, never recovery accuracy. If admitted, run 7102 and
7103 under that representation and report both sets of fits. These are repeated
computational attempts on the same datasets, not three additional independent
replications. Keep the intensity coordinates and all frozen gates unchanged.

The [Stan reparameterization discussion](https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html)
motivates matching coordinates to data informativeness; the explicit transform
and its density identity above are this implementation's derivation.
