# Fresh full-suite verification

All 222 collected tests pass across 29 files. Each file ran in a fresh process,
without deselection, under the pinned Python environment and compiler, with
NUMBA_OPT=0 and one numerical thread. The [audit](audit.json) records every
test-file result, log hash and source hash. All recorded source files were
unchanged from collection through completion.

The run began at clean revision `ba0dc2745fd1773db3e22f16d7a45a2ba70ccdc8`.
Its ending dirty-tree flag reflects concurrent report and review-document
updates; the independently compared source inventory remained identical.
The sum of test-file process times is 1,673.72 seconds under concurrent
sampling load. This is not a posterior fitting-time estimate.

Ten test-file logs retain a native PyTensor BLAS-link performance warning.
No test failed. This test record does not waive warnings or diagnostic failures
in posterior admission and does not establish posterior convergence.

Command: `python -u scripts/check_test_files.py --output artifacts/stage7-final-tests-20261022`.
The unabridged local logs and source result are retained in that directory.
