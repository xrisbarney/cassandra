# Compiler warning classification and floating-point settings

Recorded before the first joint representative attempt has completed, 2026-10-22.
Its gradient preflight emits NumbaWarning messages of the exact form:

    Cannot cache compiled function "..." as it uses dynamic globals (such as ctypes pointers and large global arrays)

The installed Numba `core/caching.py`, `CompileResultCacheImpl.check_cachable`,
emits this warning when the compiled library contains dynamic globals and
returns False to disable writing that compiled result to disk. It does not
change the calculation or indicate nonfinite values. PyTensor's installed
`link/numba/dispatch/basic.py` already suppresses this same warning for its
top-level generated graph functions; generated elementwise functions can still
emit it. Native/compiled complete-density and gradient comparisons are retained.

Classify only that exact message, with optional ANSI formatting and category
NumbaWarning, as a cache/performance warning. Keep its full text and source
location. Do not suppress other cache messages, object-mode warnings, numerical
warnings or model diagnostics under this rule. A warning remains unresolved
unless it matches this rule or the existing explicitly documented categories.
This is resolution of a nonnumerical warning, not relaxation of a numerical gate.

If an already running process records this exact warning as unresolved despite
passing every numerical gate, preserve the original record. A separate, explicit
review may resolve the warning after independently recomputing the diagnostics
from the saved four-chain posterior. Do not change its original files, discard
chains, change budgets, or choose among passing attempts by recovery performance.
Use the first eligible numerical pass. Any derived admission must retain the
original warning record, source hashes and review provenance. If no such warning
is captured inside the attempt, no retrospective review is necessary.

The installed PyTensor default is `numba__fastmath=True`; it enables selected
floating-point transformations in generated functions. The custom block
kernels use Numba's default `fastmath=False`. Previous statements that no new
fast-math option was added must not be read as claiming it was disabled in the
whole generated graph. Report this distinction, record actual compiler settings
in subsequent runs, and retain the native/compiled comparisons. No mathematical
prior or likelihood changes follow from this clarification. The existing runs
must not be retroactively described as using a different compiler profile.
