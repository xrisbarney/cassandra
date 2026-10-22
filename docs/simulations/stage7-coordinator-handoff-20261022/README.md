# Reconcile local scheduling with hosted case execution

The local rank-two coordinator ended with a nonzero process exit status. Its
raw progress/outcomes records contain two terminal diagnostic failures and
nine `execution_failed` entries. The [audit](audit.json) preserves and explains
each entry using the raw record, log, reservation and result hashes.

Eight entries correspond exactly to the predeclared hosted reservations.
Their local logs say `partial_run_preserved`: the coordinator refused to start
a duplicate fit in those directories. The ninth is the separately running
wider-dynamics fallback helper, which was likewise preserved and later
completed its prescribed schedule with a diagnostic failure.

These nine scheduling labels are not nine additional unsuccessful statistical
experiments. They are also not silently replaced in the raw log. The final
campaign audit must match the eight hosted imports to these reservations and
verify every canonical terminal result before declaring completion. At this
handoff snapshot the hosted final attempts remain in progress.
