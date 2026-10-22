# Final wider-dynamics fallback retry

The declared three-attempt schedule is exhausted for `dynamics_wide`, fallback
seed 7101. Retry 2 retained 2,000 draws from each of four chains. Maximum R-hat
is 1.5518, minimum bulk ESS 6.96, minimum tail ESS 20.80 and minimum BFMI 0.790.
There are no retained divergences or depth saturations. The diagnostic gate
fails; this case supplies no posterior recovery estimates.
Independent recomputation from the saved posterior reproduces every recorded
diagnostic within the audit's original 1e-10 comparison tolerance.

The [trace](chains.png), visually inspected, and [review](review.json) show
continued separation in the same regions documented before this retry ended.
The first chain never enters the lower-persistence marker; the other chains
spend 90.5%, 99.95% and 30.85% of their draws there. The third chain never enters
either high-persistence mean marker. Occasional transitions in the other
chains do not establish mixing, and these frequencies are not posterior region
weights. This is evidence under the wider-dynamics candidate's own target,
not an assumption that its posterior matches the primary prior.

The review records hashes of the posterior, attempt, run, data and plotting
source. Original attempts and the earlier region definition remain unchanged.
