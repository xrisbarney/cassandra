# Primary campaign recorded timing

The [audit](audit.json) and [case table](cases.csv) cover the 43 canonical
primary cases and all their prescribed attempts, including failures. They
sum to 44.423 recorded component job-hours. This is neither campaign elapsed
time nor total research compute: jobs overlapped, and superseded engineering
trials, verification, artifact transfer and analysis are outside this table.

| Recorded component | Job-hours |
|---|---:|
| Graph construction, once per process | 0.0077 |
| Explicit preflight function compilation | 0.0115 |
| Preflight first call including deferred JIT | 0.0241 |
| Explicit sampler-step compilation | 0.4810 |
| Sampling calls | 43.4212 |
| Diagnostics | 0.3922 |
| Serialization | 0.0854 |

The [chain table](chains.csv) contains 282 chain-attempt records: 425,000 warmup
draws and 307,667 retained draws completed, including partial failed attempts.
Their recorded warmup and retained sampler times are 27.585 and 15.244 hours.
These are subdivisions of sampling calls, not additional costs to add. Sampling
calls can also contain deferred compilation and postprocessing; the explicit
compilation fields do not isolate all compilation work. Unmeasured interpreter
startup and outer-run overhead are not invented or treated as zero.

The representative case consumes 29,082.48 recorded component seconds across
three unsuccessful attempts. It is not a successful-fit benchmark. The hosted
fallback seed 7103 rebuilds the graph in three separate processes; the exporter
counts those builds separately. Host, compiler profile, concurrency, seed and
stopping behavior differ, so these timings do not establish an OPT3 speedup.

A production 60-development/24-evaluation-origin cost estimate still requires
a converged production-sized pilot. Multiplying failed synthetic runtimes by
84 would not supply that evidence. The final campaign timing audit adds the
12 sensitivity cases separately.
