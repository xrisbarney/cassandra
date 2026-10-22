# Retained negative-boundary state check

The tightened-dynamics representative retry 1 visited second-factor persistence
-0.999992370958849. The [selection record](selection.json) identifies the actual
retained draw and posterior hash. Reconstructing every saved variable from its
unconstrained coordinates has maximum absolute error 3.51e-14.

The [complete-target audit](audit.json), using the hosted OPT3 compiler profile,
passes at this point. Native and Numba log densities differ by 2.73e-12 and
their full gradients by at most 1.06e-7. Thirteen selected coordinates, including
both persistence coefficients, pass the pre-existing two-adjacent-step rule
at absolute and relative tolerance 5e-5.

The complete step grid is retained. The negative-boundary persistence
coordinate's smaller steps (1e-4, 3e-5 and 1e-5) fail the tolerance; the three
larger steps pass. This illustrates small-step numerical instability and does
not justify reporting that every difference estimate agrees. No tolerance,
prior, sampler or posterior admission rule changed.

This is a local check at an actual retained state, not a rejected leapfrog
proposal and not proof of numerical accuracy at every possible state. It
neither explains nor waives the case's convergence failure. The
[inventory](artifact_inventory.json) preserves model-source and local point
hashes; the reproducible check is [check.py](check.py).
