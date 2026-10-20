# User-supplied evaluation: comparator and measurement extracts

Source: the LaTeX pasted in this conversation with the message beginning “start resolving. potential answers from a previous eval that failed.” Captured 2026-10-20. These are selected verbatim extracts, not a complete archive of the manuscript. The supplied text ended partway through an economic-validation figure.

The user subsequently instructed that only comparator identities and measurement families should carry forward. Accordingly, performance tables, numerical outcomes, and conclusions are deliberately not reproduced here. Citation keys identify references in the supplied manuscript; they do not establish source-code availability or faithful replication.

## Standard comparators: Experimental Setup

```latex
The proposed model is compared against:
(i) \textsc{RF}: a per-topic random forest (RF) on delay-embedded series, replicating \citet{kia2024};
(ii) \textsc{ARIMA}: a per-topic best-fit autoregressive integrated moving average (ARIMA) model;
(iii) \textsc{ETS}: an error-trend-seasonal (ETS) exponential smoothing state-space model;
(iv) \textsc{Na\"ive}: a seasonal random walk that forecasts each month as the value observed twelve months earlier;
(v) \textsc{BSTS-U}: a univariate Bayesian structural time series (BSTS) model (no factors, no regimes), isolating the value of the Bayesian treatment alone.
Point baselines are given predictive distributions via residual bootstrap so that probabilistic scores are comparable.
```

The word “replicating” and the blanket bootstrap prescription above belong to the supplied draft. The new design independently defines its adaptations and predictive laws.

## External comparator identities: External Benchmarks

The following short substrings are quoted from the list of external systems:

```latex
(i) \emph{Vuln4Cast} \citep{leverett2020}
(ii) the Bayesian encoder--decoder long short-term memory network (B-LSTM) of \citet{almahmoud2023}
(iii) the Bayesian state-space model (BSSM) count model of \citet{bakdash2018malware}
```

These substrings omit the surrounding implementation claims. Read the newly verified source/adaptation record in [BASELINE_SPECIFICATIONS.md](../BASELINE_SPECIFICATIONS.md) for the design actually adopted.

## Measurement names: Evaluation Metrics

These labels are extracted from the metric list, with the explanatory prose omitted:

```latex
\textbf{Continuous ranked probability score (CRPS):}
\textbf{Logarithmic score (LogS):}
\textbf{Point accuracy - mean absolute error (MAE) and root-mean-squared error (RMSE):}
\textbf{Interval coverage:}
\textbf{Probability integral transform (PIT) uniformity:}
\textbf{Diebold--Mariano (DM) test:}
```

The source also proposes an economic comparison using documented event-loss ranges and predictive intervals. Its values and interpretation are not evidence for the rerun. The new protocol separately defines the economic target and validation requirements.

## Traceability

The five standard entries and three external entries yield the eight requested comparators. The six measurement labels establish the requested metric families. The new [protocol](../EVALUATION_PROTOCOL.md), [baseline specifications](../BASELINE_SPECIFICATIONS.md), and [statistical analysis plan](../STATISTICAL_ANALYSIS_PLAN.md) define their calculation and use; they supersede procedural claims in the source excerpt.
