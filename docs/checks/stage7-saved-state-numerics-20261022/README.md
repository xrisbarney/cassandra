# Numerical review of saved states

These are states saved at retained draws that triggered a diagnostic failure.
They are not necessarily the leapfrog proposals responsible for the divergence.
None of these checks admits a failed fit or explains away its divergence.

The representative checkpoint passed native/Numba complete-density and gradient
comparisons and selected finite differences. At fallback seed 7102, the analytic
second-persistence gradient was 0.7077628284, while a float64 central difference
with step 1e-5 gave 0.7086043297. The original failed check is preserved in
[failed_check.json](failed_check.json).

The [step sweep](difference_steps.json) shows agreement at steps 0.003 and 0.001
within about 4e-6, followed by increasing error at smaller steps. To distinguish
cancellation from an incorrect gradient, an independent implementation rebuilt
the joint Gaussian coordinates with Python Decimal at 60-digit precision. It
then evaluated the persistence-dependent original Gaussian priors, NB likelihood,
coordinate Jacobians and constrained-parameter Jacobians. Constants independent
of persistence were omitted; no production model code was imported.

The [high-precision result](high_precision.json) converges to 0.707762827719 for
the second coefficient, differing from the float64 analytic derivative by
6.49e-10. The first coefficient differs by 2.34e-7. Both satisfy the original
5e-5 absolute/relative tolerance. The small-step float64 check suffered
cancellation; the tolerance was not widened. The sampler's divergence remains
a failure under its unchanged gate. This is a local numerical check, not a
claim about all points in the posterior or a replacement for mixing diagnostics.
