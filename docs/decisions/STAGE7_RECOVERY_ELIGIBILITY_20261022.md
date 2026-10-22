# Numerical admission and unresolved posterior modes

The [kernel mode audit](../checks/stage7-kernel-mode-20261022/audit.json)
finds two loading orientations in the first kernel-present seed 7101 attempt.
Its numerically admitted third attempt visits only one. This finding uses
sampling diagnostics, not generating values or recovery performance.

Keep the immutable result's numerical admission, its diagnostics and all
posterior files. Add a separate, explicit recovery-eligibility flag to the
campaign audit and downstream summaries. Exclude this fit from validated
recovery tables and label its example figure as unresolved posterior modes.
Do not rewrite an accepted numerical result as a failed numerical result or
silently drop it from the experiment denominator. Record the exclusion's
evidence path and hash. The summary must distinguish numerical passes from
fits eligible for posterior recovery interpretation.

The optional kernel has already failed every absent-kernel replicate under
the prescribed budgets. A numerically successful present-kernel fit with an
unresolved alternative mode cannot rescue that component's validation. Keep
the kernel unvalidated for Stage 8; a later dedicated identification and
sampling investigation is required before admitting it. This conclusion
does not require inventing additional kernel fits or relaxing any gate.

The same distinction applies to the old proxy seed 7102 fit. Its documented
exact sign-chart repair is being evaluated separately. Replace proxy runs
uniformly only after verifying that repair, retaining their earlier numerical
admissions and mode-conditional recovery as historical artifacts. If a new
fit has an unresolved known mode, numerical admission alone cannot clear it.

Apply this evidence-based rule to other completed cases if their retained
traces reveal the same problem. An exclusion must name the actual fit and
supporting audit; it cannot be selected using truth coverage or forecast error.
