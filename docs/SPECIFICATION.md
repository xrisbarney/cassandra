# Stage 1: Equation-to-implementation specification

Source documents: `main.tex` (Sections 3 and 4) and `IMPLEMENTATION_PLAN.md` (Stage 1).
Status: Stage 1 design complete, with explicitly assigned later-stage evidence gates. Every decision below is tagged **[D]** and is the
default the implementation will follow; **[A]** marks an alternative that is to
be evaluated against the default on development data; **[OPEN]** marks a point
that a later stage must settle (the stage is named). The companion file
`MANUSCRIPT_CHANGES.md` lists the edits the manuscript needs so that it states
what is proposed for implementation. Factual and mathematical review updated
on 2026-10-20. Numerical checks validate specific identities and approximations;
they do not establish predictive performance or replace later implementation checks.

The [evaluation protocol](EVALUATION_PROTOCOL.md), [baseline specifications](BASELINE_SPECIFICATIONS.md), and [statistical analysis plan](STATISTICAL_ANALYSIS_PLAN.md) freeze initial model scope,
origin schedules, scoring, comparison, and failure rules. The
[carry-forward record](PRIOR_EVALUATION_REVIEW.md) limits reuse of the supplied
failed evaluation to baseline identities and measurement families. January
2024–December 2025 is the held-out forecast period for the rerun; design and
tuning use data through December 2023. No old results or conclusions guide selection.

---

## 0 Conventions

| Symbol | Meaning |
|---|---|
| $t = 1,\dots,T$ | calendar month. $T$ is the forecast origin of a given fit. |
| $k = 1,\dots,K$ | threat category. Category $K$ is the reserved "unknown" category (Stage 5). |
| $s = 1,\dots,S$ | economic sector (BEA summary classification, Stage 11). |
| $r = 1,\dots,R$ | regime label. |
| $j = 1,\dots,r_f$ and $j = 1,\dots,r_h$ | intensity-factor and severity-factor index. |
| $h = 1,\dots,H$ | forecast horizon in months, $H = 12$. |
| $m = 1,\dots,M$ | posterior draw. |

All logs are natural. $\operatorname{sigmoid}(x)=1/(1+e^{-x})$, $\operatorname{logit}$ its inverse. In $\mathcal N(\mu,v)$ and $\operatorname{LogNormal}(\mu,v)$, $v$ denotes variance (log-variance for LogNormal); software accepting a standard deviation must receive $\sqrt v$. $\operatorname{HalfNormal}(s)$ uses scale/standard deviation $s$. "Mask" means a binary availability indicator $m^{(\cdot)}_{k,t}$; a masked cell contributes nothing to the likelihood and is never imputed.

**Masking implementation [D].** Likelihood terms are never formed as mask × log-density, because $0 \times (-\infty)$ and $0 \times \text{NaN}$ are NaN. Either (i) observed cells are gathered by index before evaluating their densities, or (ii) missing observations are replaced by valid placeholders before any transformation and their finite log-density terms are selected with `where(mask, term, 0)`. Selection alone does not guarantee safe gradients: all evaluated branches, parameters, and denominators must be valid. In the heteroscedastic alternatives, masked score counts must also receive a positive computational placeholder. Option (i) is the default; fixed-shape implementations must verify finite values and gradients.

**Origin conditioning.** In a point-in-time experiment, every input at origin $T$ must have been available at $T$. Quantities indexed by $t < T$ inside an origin-$T$ fit (effort, within-fit filtered regime probabilities, latent states) are retrospective because their estimated parameters or latent draws use the window through $T$. Terminal values qualify as historical real-time forecasts only if the entire pipeline, including topic metadata, economic inputs, and model selection, respects the cutoff. The revised-data experiment below is explicitly labelled "revised historical data"; refitting it at each origin does not restore historical availability.

---

## 1 Observed data

| Object | Definition | Support | Availability mask |
|---|---|---|---|
| $N_{k,t}$ | number of CVE records with publication date in month $t$ hard-assigned to category $k$ by the frozen topic model | $\mathbb{Z}_{\ge 0}$ | $m^N_t$ from the Stage 4 coverage audit (months with incomplete NVD coverage are masked, not assumed complete) |
| $N^{\text{tot}}_t = \sum_k N_{k,t}$ | monthly CVE total, including the unknown category | $\mathbb{Z}_{\ge 0}$ | $m^N_t$ |
| $n^B_{k,t}$, $B_{k,t}$ | number of CVEs in $N_{k,t}$ with a selected CVSS base score, and their mean, under a declared score-vintage policy | $\mathbb{Z}_{\ge 0}$, $[0,10]$ | $m^B_{k,t} = 1$ only when the cohort is available and $n^B_{k,t}>0$; default revised-data vintage and point-in-time alternative are distinguished below |
| $n^E_{k,t}$, $E_{k,t}$ | number of CVEs in $N_{k,t}$ found in the selected EPSS snapshot dated no later than month-end, and their mean score | $\mathbb{Z}_{\ge 0}$, $[0,1]$ before clipping | $m^E_{k,t} = 1$ only when the cohort and acceptable snapshot are available and $n^E_{k,t}>0$; the selected archive begins in April 2021 |
| $V_{k,t}$ | number of CISA KEV catalogue additions dated in month $t$ whose CVE is assigned to category $k$ | $\mathbb{Z}_{\ge 0}$ | audit-derived $m^V_t$, no earlier than December 2021; November is excluded as the launch backlog, not treated as an ordinary monthly flow |
| $D_{s,t}$ | distinct incident–registrant pairs first disclosed under SEC 8-K Item 1.05 in month $t$, assigned to the registrant's sector; amendments linked to the original | $\mathbb{Z}_{\ge 0}$ | audit-derived $m^D_{s,t}$, no earlier than January 2024; December 2023 and June 2024 are excluded from the default full-month step model (§5.5) |
| $L_{s,t}$ | loss marks attached to disclosed incidents | $\mathbb{R}_{\ge 0}$ | **not a likelihood term** (see §6) |
| $e_t$ | reporting-effort proxy, computed at the origin (§2) | $\mathbb{R}$ | requires sufficient audited counts to fit the trend; missing months use prediction-only filtering |
| $M_{s,k,t}$ | exposure map | $[0,1]$ | exogenous input (Stage 11) |

Masks combine predeclared calendar exclusions with the Stage 4 coverage audit. Passing a date threshold does not establish that a cell is observed; the audit may mask additional cells.

**Source coverage and historical versions.** NIST reports a 2024 processing slowdown and a growing enrichment backlog in March 2025. Its April 2025 announcement deferred further enrichment for pre-2018 CVEs awaiting it. From April 15, 2026, enrichment prioritises KEV, federal-use, and critical-software CVEs; other submissions still enter NVD but are not scheduled for immediate enrichment. This does not mean every lower-priority record lacks CVSS/CPE data. Publication counts and enrichment coverage require separate audits; incomplete enrichment alone does not establish missing publications. [NIST announcements](https://www.nist.gov/itl/nvd).

NVD provides a CVE Change History API; the current record alone is insufficient to establish month-end metadata. Stage 4 must assess whether histories and archived records reconstruct the required fields. CNA-supplied scores also need timestamps or archived versions; their presence today does not prove publication-time availability. [NVD's API announcement](https://groups.google.com/a/list.nist.gov/g/nvd-news/c/8FGMfqB2PQ8).

**[D]** The initial revised-data experiment uses CVSS as retrieved, reports the vintage and $n^B/N$, and makes no point-in-time claim. **[A]** a separate as-of experiment uses only scores demonstrably available by the cutoff. Keep their panels and evaluation results separate.

**KEV launch.** Treat the November 2021 launch publication as a backlog rather than new exploitation that month. The design depends on that distinction, not an exact launch count. The previously cited approximate split (about 200 older entries plus 90 from 2021) is not needed as an implementation input; verify any exact historical count from a dated catalogue snapshot before reporting it. [CISA launch fact sheet](https://www.cisa.gov/sites/default/files/publications/Reducing_the_Significant_Risk_of_Known_Exploited_Vulnerabilities_20211103.pdf). Later additions are catalogue-entry dates, not dates on which exploitation first occurred.

**SEC dates and target.** The rule became effective September 5, 2023. Item 1.05 compliance began December 18, 2023 for registrants other than smaller reporting companies, and June 15, 2024 for smaller reporting companies. January and July are the first full months under those respective populations. [SEC release](https://www.sec.gov/rule-release/33-11216), [SEC compliance guide](https://www.sec.gov/resources-small-businesses/small-business-compliance-guides/cybersecurity-risk-management-strategy-governance-incident-disclosure). A shared attack disclosed by several registrants creates several incident–registrant observations; link the common event separately rather than accidentally removing affected firms during deduplication.

**[D] EPSS vs KEV.** The manuscript's single $E_{k,t}$ "mean EPSS or KEV inclusion counts" is split into two channels with different supports: $E$ (a probability) and $V$ (a count). $E$ is the primary exploitation channel; $V$ is an optional channel added in model version G and ablated separately.

**[D] Which CVEs $E$ summarises, and the cutoff.** Use the publication-month cohort and the last acceptable snapshot dated no later than month-end, at most seven calendar days old. This is a chosen operational tolerance, not an empirical result. CVEs absent from that snapshot are unavailable, not zero; $n^E$ is the number actually found. Record snapshot date, model version, retrieval hash, and age at month-end. Stage 4 must audit archive gaps; missing or too-old snapshots mask the cell. Verify actual publication timing where possible rather than treating an archive filename as proof of historical availability. Any tolerance change follows the protocol's amendment rule.

The [FIRST data page](https://www.first.org/epss/data.html) dates the available archive from 2021-04-14, v2 from 2022-02-04, v3 from 2023-03-07, v4 from 2025-03-17, and v5 from 2026-06-15, and documents score shifts at model changes. Thus the EPSS history through 2025 crosses v1–v4. **[D]** record the selected snapshot's version and use the corresponding step terms in §5.2. **[A]** restrict a separate diagnostic to snapshots produced by a single version. The [maintainer's archive](https://github.com/empiricalsec/epss_scores) is the bulk-data source. All 24 month-end filenames for 2024–2025 were verified on 2026-10-20; file contents and CVE-level completeness remain audit outputs.

**[D] Boundary values and scoring convention.** Preserve raw $B$ and $E$. Define coarsened analysis targets $B^c=\operatorname{clip}(B,0.05,9.95)$ and $E^c=\operatorname{clip}(E,10^{-4},1-10^{-4})$. Fit and score their induced censored distributions: endpoints contribute tail probability masses, and interior values contribute densities. Applying an ordinary Normal density to a clipped logit is not that likelihood. Apply the same coarsening to every model's predictions and label the scores as scores of the coarsened targets. Report clipped fractions and sensitivity to thresholds; the 1% review threshold is a design choice, not a validity guarantee. The endpoint-mass rule is illustrated in the [PyMC censored-distribution documentation](https://www.pymc.io/projects/docs/en/v5.16.0/api/distributions/censored.html).

**[D] Topic model.** TF–IDF + NMF with hard assignment per CVE, fitted on the initial training window and frozen (plan, Stage 5). The primary pilot uses nine learned topics plus unknown, total $K=10$; $K=20$ is a later resolution sensitivity experiment. The protocol defines assignment, initial window, and active/sparse reporting. Raw per-topic CRPS across different category systems is not a common-target comparison.

---

## 2 Reporting effort $e_t$

**Definition [D].** Let $y_t = \log(1 + N^{\text{tot}}_t)$. At forecast origin $T$, fit a Gaussian local-linear-trend model

$$y_t = \ell_t + \varepsilon^e_t,\quad \ell_t = \ell_{t-1} + \delta_{t-1} + \zeta^e_t,\quad \delta_t = \delta_{t-1} + \xi^e_t$$

by maximum likelihood on the available $y_{1:T}$ (statsmodels `UnobservedComponents`, local linear trend). Missing count months are missing observations in this auxiliary model, not zeros. Define the raw effort as the filtered level $\hat\ell_{t|t;\hat\theta_T}$, conditional on trend parameters fitted through $T$, and standardise it over the training window:

$$e_t = \big(\hat\ell_{t|t} - \bar\ell_T\big)/s_{\ell,T},\qquad t \le T,$$

with $\bar\ell_T, s_{\ell,T}$ the mean and standard deviation of $\hat\ell_{t|t}$ over $t = 1,\dots,T$. The hyperparameters of the trend model are re-estimated at every origin; no quantity from beyond $T$ enters.

The display suppresses the origin-dependent parameter subscript. If the fitted levels are constant, set $e_t=0$ rather than divide by zero. Insufficient coverage or a failed trend fit invalidates the origin; it is not silently replaced by a fitted effort series.

**What "one-sided" does and does not deliver.** The filtered level at $t$ conditions on $y_{1:t}$ *given the trend-model parameters*, but those parameters and the standardisation constants are estimated from $y_{1:T}$. So $e_t$ for $t < T$ is not a function of $y_{1:t}$ alone; it is the origin-$T$ effort series. This is sufficient for the forecast made at $T$ (nothing after $T$ enters) and is the only property claimed. The effort series is recomputed at every rolling origin, so the historical values used in different fits differ slightly.

**Forecast projection [D].** $e_{T+h} = e_T$ for all $h$ (hold-last, as the manuscript states).
**[A]** $e_{T+h}$ = standardised $h$-step forecast of $\ell$ from the trend model (carries drift and its own uncertainty). Compared in Stage 10.

**How $e_t$ enters [D].** As a fixed offset with coefficient one (Eq. obs-cve below). **[A]** a free coefficient $\delta_e \sim \mathcal{N}(1, 0.5^2)$.

**What is being assumed.** $e_t$ is a count-derived proxy treated as fixed in the main fit. Its coefficient and standardisation determine how much aggregate trend is removed; they do not force all genuine growth into effort. The remaining $\lambda$ can still contain aggregate trends. This is an empirical-Bayes working adjustment, not an independently identified reporting process or a fully generative joint model of $N$ and $e(N)$. Its estimation uncertainty is omitted in the default fit. Stage 7 must regenerate and refit this preprocessing inside each simulated dataset, and Stage 14 compares it with no adjustment and alternative projections.

---

## 3 Latent threat dynamics

### 3.1 Operational definition of the latent intensity [D]

$$\lambda_{k,t} \equiv \mathbb{E}[N_{k,t} \mid \text{latent state},e_t] \,/\, e^{e_t}$$

is the expected number of CVE publications per month in category $k$ at the reference value of the effort proxy ($e_t=0$, the training-window mean filtered log-volume level). Its unit is CVEs per month. This reference changes when the preprocessing window changes, so raw latent levels across origins are not automatically comparable. Predictive association with $E$, $V$, or $D$ can support usefulness, but cannot by itself establish that $\lambda$ measures true malicious activity or a causal effect.

### 3.2 Regime process

$$z_1 \sim \text{Uniform}\{1,\dots,R\},\qquad z_t \mid z_{t-1} = r' \sim \text{Categorical}(\bm{\Pi}_{r',\cdot}).$$

Regime affects **levels only** (Markov-switching mean). Transition dynamics of the factors do not depend on $z$.

**Regime-level parametrisation and label identification [D].**

$$\mu_{r,k} = m_k + c_r,\qquad c_1 = 0,\ c_r = c_{r-1} + \Delta_r,\ \Delta_r > 0 \ (r \ge 2),$$
$$\nu_{r,k} = n_k + d_r,\qquad d_1 = 0,\ d_r \in \mathbb{R}\ (r \ge 2).$$

Regimes are ordered by the common intensity shift $c_r$; this removes the default model's label permutation. Severity shifts $d_r$ are free. **[A]** category-specific deviations $\mu_{r,k}=m_k+c_r+\epsilon_{r,k}$, with $\epsilon_{1,k}=0$ and $\sum_k\epsilon_{r,k}=0$ for each $r>1$, and a Gaussian prior on the constrained deviations. These constraints separate category baselines, common shifts, and deviations; unconstrained deviations would reintroduce ambiguity. Evaluate the alternative only on development data.

### 3.3 Factors

Intensity factors $\bm{f}_t \in \mathbb{R}^{r_f}$ and severity factors $\bm{h}_t \in \mathbb{R}^{r_h}$:

$$\bm{f}_t = \bm{\Phi}\bm{f}_{t-1} + \bm{w}_t,\ \bm{w}_t \sim \mathcal{N}(\bm{0}, \bm{I}_{r_f});\qquad \bm{h}_t = \bm{\Phi}_h\bm{h}_{t-1} + \bm{w}^h_t,\ \bm{w}^h_t \sim \mathcal{N}(\bm{0}, \bm{I}_{r_h}).$$

**[D] Scale identification.** The innovation covariances are fixed to the identity ($\bm{Q} = \bm{I}$, $\bm{Q}_h = \bm{I}$). The manuscript's free $\bm{Q}$ together with a positive-diagonal lower-triangular $\bm{\Gamma}$ leaves the scale of each factor free (rescale column $j$ of $\bm{\Gamma}$ by $1/a$ and $f_{j,t}$ by $a$). With $\bm{Q} = \bm{I}$ the only admissible rotations are orthogonal, and the only orthogonal matrix that keeps a lower-triangular positive-diagonal block lower-triangular with positive diagonal is the identity. Loadings carry the scale.

**[D] Transition matrices.** $\bm{\Phi}=\operatorname{diag}(\phi_1,\dots,\phi_{r_f})$ with $\phi_j\in(-1,1)$; likewise $\bm{\Phi}_h$. Together with stationary initialisation, these define stationary independent AR(1) factors. **This is a substantive restriction, not only identification:** diagonal dynamics rule out cross-factor transitions, while the triangular block restricts which factors the leading categories can load on. The leading category loads on only the first factor, though it still has regime, kernel, and residual contributions. **[A]** full stable $\bm{\Phi}$, with its differentiable parameterisation and stationary initial covariance to be specified before comparison. Hard rejection at the stability boundary is not assumed to be a suitable NUTS parameterisation.

**Initial state.** $\bm{f}_0 \sim \mathcal{N}(\bm{0}, \operatorname{diag}(1/(1-\phi_j^2)))$ (stationary distribution); same for $\bm{h}_0$.

### 3.4 Log-intensity and logit-severity

$$\bm{\eta}_t = \bm{\mu}_{z_t} + \bm{\Gamma}\bm{f}_t + \bm{a}\,g_t + \bm{u}_t,\qquad u_{k,t} \sim \mathcal{N}(0, \tau_k^2),\qquad \lambda_{k,t} = \exp(\eta_{k,t}).$$
$$\bm{\varsigma}_t = \bm{\nu}_{z_t} + \bm{\Psi}\bm{h}_t,\qquad \sigma_{k,t} = \operatorname{sigmoid}(\varsigma_{k,t}) \in (0,1).$$

**[D] No idiosyncratic latent severity noise.** The severity process has no $v_{k,t}$ term. With latent iid Gaussian noise of variance $\omega_k^2$ and the default Gaussian observation variance $\kappa_k^2$ on the logit scale, integration identifies only $\omega_k^2+\kappa_k^2$. Censoring that same observation does not resolve the ambiguity. The default therefore assigns residual category-month variation to measurement noise. **[A]** investigate external or within-cohort evidence for measurement variance before adding latent severity noise. Raw CVSS dispersion is not directly the variance of the logit of its mean. For independent normalized scores with mean $p$ and variance $s^2$, a delta-method approximation is $s^2/[n^B p^2(1-p)^2]$, away from the boundaries. Dependence, score-selection bias, and small cohorts require further checks; identification is not guaranteed.

The analogous question for intensity: $u_{k,t}$ (latent Gaussian, additive on the log scale) and the NegBin overdispersion $\psi_k$ (§5.1) are both category-month extra variation. They are distinguishable in principle because a log-normal mixture of NegBins has a different shape from a NegBin, but the distinction is weak at low counts. Stage 7 checks recovery of $\tau_k$ and $\psi_k$ jointly; if they are not recovered, $u_{k,t}$ is dropped and $\psi_k$ carries all extra variation.

**[D] Severity scale.** $\sigma_{k,t}\in(0,1)$ is a normalized CVSS location: the median of the underlying uncensored logit-normal observation, not generally its mean. Using it in $\lambda\sigma$ defines a CVSS-weighted publication-rate index. Dividing CVSS by 10 does not turn it into a probability or an economically calibrated damage fraction. The manuscript's lognormal channel is alternative **[A]** in §5.3.

**Loading constraints [D].** Leading $r_f\times r_f$ block of $\bm{\Gamma}$ lower triangular with strictly positive diagonal; remaining rows free. Apply the analogous constraint to $\bm{\Psi}$. Order known categories by counts in the initial training window, freeze that order for the benchmark, and retain the unknown category at index $K$. Require $r_f,r_h\le K-1$ and sufficiently observed leading categories. Reordering by each origin's counts would change the restricted model and is not the default.

$u_{k,t}$ is sampled as a latent variable (non-centred: $u_{k,t} = \tau_k \tilde u_{k,t}$, $\tilde u \sim \mathcal{N}(0,1)$), so that given all continuous latents, $\bm{\eta}_t$ under regime $r$ is a deterministic function $\bm{\eta}^{(r)}_t = \bm{\mu}_r + \bm{\Gamma}\bm{f}_t + \bm{a} g_t + \bm{u}_t$, and $\bm{\varsigma}^{(r)}_t = \bm{\nu}_r + \bm{\Psi}\bm{h}_t$. This is what makes the regime marginalisation in §7 a pure emission-level recursion.

### 3.5 Temporal kernel

**[D] A causal, compact-support kernel with a fixed centre.** The width parameter is written $b$ (months), not $h$, to avoid collision with the forecast horizon. Fix an integer centre lag $c_0$ and maximum lag $D_{\max} = 2c_0$ (pilot: $c_0 = 30$, $D_{\max} = 60$) and let

$$g_t = \sum_{d=0}^{D_{\max}} w_d(b)\, \xi_{t-d},\qquad \xi_t \overset{iid}{\sim} \mathcal{N}(0,1),\qquad
\tilde w_d(b) = \operatorname{sigmoid}\!\big((b - |d - c_0|)/2\big),\qquad w_d = \tilde w_d \Big/ \sqrt{\textstyle\sum_{d'=0}^{D_{\max}} \tilde w_{d'}^2}.$$

This is a fixed integer shift of the two-sided kernel **after truncating that kernel to $[-c_0,c_0]$ and renormalising**. The shifted and truncated two-sided processes have the same finite-dimensional Gaussian distribution under iid innovations. Equality with the untruncated manuscript process is only approximate; a small covariance discrepancy does not by itself bound the observation-likelihood discrepancy.

The checks in [kernel_check2.py](checks/kernel_check2.py) compare a two-sided reference truncated at 300 months with $c_0=30$. At $b\in\{6,12,12.5,18\}$, lag-12 discrepancies are below $10^{-4}$, but this is not an all-lag or all-width bound: at $b=18$, the lag-34 discrepancy is about $2.3843\times10^{-4}$. At that width, the retained edge weight is $\operatorname{sigmoid}(-6)\approx0.002473$ and the largest discarded weight is $\operatorname{sigmoid}(-6.5)\approx0.001501$. Stage 7 must assess truncation across relevant lags and widths. A window centred at the learned $b$ instead gives lag-12 covariance about 0.550 versus 0.598 at $b=12$; that different construction is not used.

The half-width $b$ controls the decay shape, not the exact correlation cutoff. Every retained weight is positive, so covariance is nonzero at every lag through $D_{\max}$ and zero beyond it. Unit squared-weight normalisation gives $\operatorname{Var}(g_t)=1$ and the MA construction guarantees positive semidefiniteness. The shift makes future $g$ depend only on innovations through the forecast date; pre-sample innovations $\xi_{1-D_{\max}:0}$ receive the same iid prior. The pilot fixes $b_{\max}=18$ and $c_0=30$. The gap $c_0-b_{\max}=12$ is a numerical design choice, not a general error guarantee; larger widths require a new truncation assessment.

**[D] Scale and sign identification.** The manuscript's $\sigma_g$ is dropped: only the products $a_k \sigma_g$ are identified. The loadings $\bm{a}$ carry the scale, and the sign ambiguity $(\bm{a}, \bm{\xi}) \sim (-\bm{a}, -\bm{\xi})$ is removed by constraining $a_1 \ge 0$ (first category in the ordering of §3.4). The manuscript's interpretation "$\sigma_g$ measures how much local covariance adds" is replaced by the posterior of $\|\bm{a}\|$.

**Weak identification flagged.** $g_t$ is a common MA($D_{\max}$) component and $\bm{f}_t$ a common AR(1) component; they are distinguishable only through their autocorrelation shapes. Stage 7 (synthetic design "temporal-kernel dependence") must show that $b$ and $\|\bm{a}\|$ are recovered when present and that $\|\bm{a}\|$ shrinks when absent; if not, the kernel is reported as a non-identified extension.

---

## 4 Priors (all [D]; sensitivity in Stage 14)

| Parameter | Prior | Note |
|---|---|---|
| $m_k$ | $\mathcal{N}(\bar m, s_m^2)$, $\bar m \sim \mathcal{N}(0, 3^2)$, $s_m \sim \text{HalfNormal}(2)$ | hierarchical pooling of category levels |
| $\Delta_r$, $r \ge 2$ | $\text{HalfNormal}(1)$ | ordered regime shifts |
| $n_k$ | $\mathcal{N}(\bar n, s_n^2)$, $\bar n \sim \mathcal{N}(1, 1)$, $s_n \sim \text{HalfNormal}(1)$ | logit-severity levels; $\operatorname{sigmoid}(1) \approx 0.73$, i.e. CVSS 7.3 |
| $d_r$, $r \ge 2$ | $\mathcal{N}(0, 0.5^2)$ | |
| $\bm{\Pi}_{r,\cdot}$ | $\text{Dirichlet}(\bm{\alpha}_r)$, $\alpha_{r r'} = 1 + 9\cdot\mathbb{1}[r = r']$ | persistence prior; prior mean self-transition $10/(R+9)$ |
| $\phi_j$, $\phi^h_j$ | $\text{Uniform}(-1, 1)$ | |
| $\Gamma_{jj}$, $\Psi_{jj}$ (diagonal of leading block) | $\text{HalfNormal}(1)$ | |
| free entries of $\bm{\Gamma}$, $\bm{\Psi}$ | $\mathcal{N}(0, 1)$ | |
| $\tau_k$ | $\epsilon + \text{HalfNormal}(1)$, $\epsilon = 0.01$ | optional numerical floor on idiosyncratic intensity noise; it does not constrain regime transition probabilities |
| $\psi_k$ (NB dispersion) | $\text{LogNormal}(\log 10, 1)$ | |
| $\alpha_k$ | $\mathcal{N}(-3, 2^2)$ | proposed intercept prior; sigmoid(-3) is about 0.047, not an established empirical mean |
| $\beta_k$ | $\mathcal{N}(0, 1)$ | sign unconstrained; positive is the hypothesis |
| $\varsigma_k$ (EPSS noise sd) | $\text{HalfNormal}(1)$ | |
| $\kappa_k$ (severity noise sd) | $\text{HalfNormal}(0.5)$ | |
| $\alpha^V_k$, $\beta^V_k$, $\psi^V_k$ | $\mathcal{N}(-3, 2^2)$, $\mathcal{N}(0,1)$, $\text{LogNormal}(0, 1)$ | KEV channel (version G) |
| $\rho_s$ | $\text{LogNormal}(-5, 2^2)$ | **[OPEN, Stage 5]** re-centre after the scale of $\sum_k M\lambda$ is known |
| $\pi_s$ | $\text{HalfNormal}(0.5)$ | baseline incidents per sector-month |
| $\gamma_q$ (reporting-population step) | $\mathcal{N}(0, 0.5^2)$ | log-rate shift after smaller reporting companies joined (§5.5) |
| $\gamma_{v2}$, $\gamma_{v3}$, $\gamma_{v4}$ (EPSS version steps) | $\mathcal{N}(0, 1)$ | logit-scale shifts at the v2, v3, and v4 model changes (§5.2); instantiate only changes known at the fit's origin |
| $\phi^D_s$ (NB dispersion, alternative) | $\text{LogNormal}(0, 1)$ | only under [A] NegBin incidents |
| $b$ (kernel half-width, months) | $\text{LogNormal}(\log 12, 0.5)$ truncated to $(0, b_{\max}]$, $b_{\max} = 18$ | |
| $a_1$ | $\text{HalfNormal}(s_a)$; $a_k$, $k \ge 2$: $\mathcal{N}(0, s_a^2)$; $s_a \sim \text{HalfNormal}(0.5)$ | |

Sampled latents: $\tilde{\bm{u}}_{1:T}$, $\bm{f}_{0:T}$, $\bm{h}_{0:T}$, and $\xi_{1-D_{\max}:T}$. Their dimension is $KT+(r_f+r_h)(T+1)+T+D_{\max}$: 2,764 for $K=20,T=108,r_f=r_h=2,D_{\max}=60$. Static dimensions must be counted from the enabled channels and free constrained parameters; they include loadings, hyperparameters, and $R(R-1)$ transition degrees of freedom. All prior scales are proposed choices to check with prior predictive simulation, not claims of being uninformative.

---

## 5 Measurement model

The conditional working likelihood factors over channels given latent states, fixed covariates, and availability masks. Because mark existence depends on cohort size, this is not unconditional independence of counts and their observed marks. Each channel contributes only unmasked cells (§0); missingness is not modelled by this factorisation.

**Scoring on a common observable scale.** For an interior probability $x$, $\log p_X(x)=\log p_Y(\operatorname{logit}x)-\log(x(1-x))$. For interior severity, take $x=B/10$ and additionally subtract $\log 10$. At a clipping endpoint use the corresponding tail probability, not this density/Jacobian formula. CRPS uses coarsened predictive samples on the probability or 0–10 scale. All alternatives use the same target transformation and endpoint convention. These are scores of $E^c,B^c$; they are not exact scores of the raw boundary observations.

### 5.1 Vulnerability counts [D]
$$N_{k,t} \sim \text{NegBin}_2\big(\mu^N_{k,t}, \psi_k\big),\quad \mu^N_{k,t} = \exp(e_t + \eta_{k,t}),\quad \operatorname{Var} = \mu + \mu^2/\psi_k.$$
NB2 parametrisation (mean, concentration). Do not clip $\eta$ to hide implausible draws. Examine large rates using both $e_t+\eta$ and predictive counts; $\eta>20$ is a pilot diagnostic flag, not a universal mathematical failure threshold.

### 5.2 Exploitation probability [D]
Let $v(t)$ be the version of the selected snapshot, with cumulative indicators $I_j(t)=\mathbb{1}[v(t)\ge vj]$ for $j=2,3,4$ in the history through 2025. Let $\mathcal J_T$ contain only version changes known at origin $T$:
$$Z^E_{k,t}\sim\mathcal{N}\big(\alpha_k+\beta_k\eta_{k,t}+\textstyle\sum_{j\in\mathcal J_T}\gamma_{vj}I_j(t),\varsigma_k^2\big),\qquad E^c_{k,t}=\operatorname{clip}(\operatorname{sigmoid}Z^E_{k,t},10^{-4},1-10^{-4}).$$
Do not include v4 in a pre-announcement forecast merely because its release date is known today. A newly known change without observations has a prior-only effect, not an estimated adjustment. Apply §8's forecast-version rule and label any unsupported-version mark target ineligible before scoring; count forecasts remain eligible.
Evaluate the induced likelihood only for $m^E_{k,t}=1$. Version effects are proposed additive adjustments, not a guarantee of comparability. **[A]** variance $\varsigma_k^2/n^E_{k,t}$, or a v3-only experiment without version steps. Heteroscedastic forecasts require an explicit future-count or conditional-on-count protocol (§8); observed future $n^E$ must not silently enter an unconditional forecast.

### 5.3 Severity [D]
$$Z^B_{k,t}\sim\mathcal{N}(\varsigma_{k,t},\kappa_k^2),\qquad B^c_{k,t}=\operatorname{clip}(10\operatorname{sigmoid}Z^B_{k,t},0.05,9.95).$$
Evaluate its induced censored likelihood only for $m^B_{k,t}=1$.
**[A] (manuscript family)** draw $B^*_{k,t}\sim\operatorname{LogNormal}(\log\sigma^*_{k,t},\kappa_k^2)$ with $\log\sigma^*=\nu+\Psi h$, then apply the same coarsening to $B^*$. This family is unbounded before coarsening. Compare both families on $B^c$. For the economic layer use $\sigma=\sigma^*/10$ for this alternative and recalibrate damage parameters; unlike the default, this normalized index is not bounded by one. **[A]** variance $\kappa_k^2/n^B_{k,t}$, subject to the conditional-on-count evaluation and variance checks in §3.4 and §8.

### 5.4 Known-exploited additions [D, version G only]
$$V_{k,t} \sim \text{NegBin}_2\big(\exp(\alpha^V_k + \beta^V_k \eta_{k,t}),\ \psi^V_k\big),\qquad \text{masked by } m^V_t.$$

### 5.5 Incident disclosures [D]
$$D_{s,t} \sim \text{Poisson}\Big(e^{\gamma_q q_t}\big(\rho_s \sum_k M_{s,k,t}\lambda_{k,t} + \pi_s\big)\Big),\qquad \text{masked by } m^D_{s,t},$$
where $q_t=\mathbb{1}[t\ge\text{Jul 2024}]$ distinguishes full months before and after the smaller-company compliance change. Mask June 2024 in this default rather than treating its partial change as zero exposure. The common step is an approximation, not a measured population adjustment. **[A]** a sector-specific eligible-registrant exposure $R_{s,t}$, with rate $R_{s,t}(\tilde\rho_s\sum_kM_{s,k,t}\lambda_{k,t}+\tilde\pi_s)$ and coefficients/prior scales redefined per registrant. Equivalently, $\log R_{s,t}$ is an additive offset to the log of that rate; a log count must never replace a multiplicative rate factor directly. **[A]** NegBin observations with dispersion $\phi^D_s$.

### 5.6 Conditional independence
This is an assumption to assess in Stage 14. $E$ and $B$ use the current publication cohort; KEV additions can refer to older cohorts. Shared CVE information and selection processes can still produce residual dependence. Residual cross-correlations and channel-dropping comparisons can reveal problems but cannot prove conditional independence.

---

## 6 Loss marks and the economic layer: modular, not joint [D]

The posterior in §7 is $p(\text{latents}, \bm{\Theta} \mid N, B, E, [V], D)$. **$L_{s,t}$ is not a likelihood term.** Loss marks are reserved for the economic layer: (i) as part of the event ledger that calibrates $\phi_s$ (Stage 12) and (ii) as evaluation evidence (Stage 13). The reason: a loss likelihood would require a model of which disclosed incident produced which loss, and a dollar figure attached to a disclosure measures a different thing (ransom, remediation, claims) from the output disruption the economic layer models.

The economic layer is calibrated **conditionally on posterior draws of the technical model** (a cut model, two-stage): the technical posterior is fixed, and the damage parameters are fitted given it. Loss evidence does not feed back into $\lambda$ or $\sigma$. The manuscript's "single Bayesian hierarchical state-space model" is therefore two modules; see `MANUSCRIPT_CHANGES.md`.

**Cut distribution and its Monte Carlo approximation.** Let $\theta_E=(\beta,c,\bar d)$, $\mathcal{E}$ denote calibration events and the anchor, and $X$ denote the technical states needed to construct their shocks. Recover the regime path for each technical draw before constructing $\lambda,\sigma$. The cut joint is $p_T(X\mid\text{data})p(\theta_E\mid\mathcal{E},X)$, where the second factor is normalised separately for each $X$. Its marginal integral over $X$ is approximated by

$$p_{\text{cut}}(\theta_E \mid \mathcal{E}, \text{data}) \approx \frac{1}{M}\sum_{m=1}^{M} p\big(\theta_E \mid \mathcal{E}, \bm{\lambda}^{(m)}, \bm{\sigma}^{(m)}\big),\qquad
p\big(\theta_E \mid \mathcal{E}, \bm{\lambda}^{(m)}, \bm{\sigma}^{(m)}\big) \propto p(\theta_E)\, p\big(\mathcal{E} \mid \theta_E, \bm{\lambda}^{(m)}, \bm{\sigma}^{(m)}\big),$$

each conditional normalised separately. This differs from using the draw-averaged likelihood or the full feedback posterior. **[D]** select $J$ technical draws across converged chains without evidence-based reweighting (pilot $J=100$). Fit each three-parameter conditional using multiple NUTS chains, initially four chains with 200 retained draws each, increasing sampling until §7.4 diagnostics and Monte Carlo error targets are satisfied. Preserve pairs $(\theta_E,j)$ through future simulation; do not independently recombine economic and technical draws. Check sensitivity to both $J$ and conditional simulation error. Shared randomness reduces effective information, so the pooled draw count is not an effective sample size. [Cut-posterior formulation](https://arxiv.org/abs/2202.09968). Full feedback is a separate Stage 12 sensitivity experiment.

**Damage function** (exact form, fixed here; calibration is Stage 12). Shock load $S_{s,t}=\sum_kM_{s,k,t}\lambda_{k,t}\sigma_{k,t}$. Require $\beta>0,c>0,\bar d\in(0,1)$; Stage 12 sets priors. The operating point $\bar S_s$ is the median across training months of posterior-mean shock loads, computed once per fit and held fixed. This plug-in normalisation does not propagate its own uncertainty; compare a draw-specific alternative in Stage 12. If the median is zero but the training mean is positive, use that mean and record the fallback. A sector with zero training and forecast exposure has zero direct loss. Positive forecast exposure with no positive training reference is unsupported: supply an explicitly justified external reference or report the sector as uncalibrated, never silently assign zero loss.

$$\phi_s(S) = \bar d\ \frac{\operatorname{sigmoid}\big(\beta\,(S/\bar S_s - c)\big) - \operatorname{sigmoid}(-\beta c)}{1 - \operatorname{sigmoid}(-\beta c)},\qquad \phi_s(0) = 0,\ \phi_s(\infty) = \bar d,$$

three global parameters: steepness, centre in units of typical load, and maximum direct disruption fraction. Direct loss is $d_{s,t}=\phi_s(S_{s,t})x_{s,t}$. Dividing annual BEA gross output by 12 is a constant-month approximation, not observed monthly output; use vintages available at the origin for point-in-time experiments. Proposed propagation is $\bm\ell_t=\bm\Lambda_L\bm d_t$, $\ell_t^{\rm agg}=\bm1^\top\bm\ell_t$. Gross-output sums are not GDP/value-added losses; direct disruption bounds do not bound Leontief-amplified sector losses below output. **[OPEN, Stage 11]** the economic interpretation, compatible requirements table, feasibility checks, and no-amplification comparison. **[OPEN, Stage 12]** event/anchor likelihood, compatible loss definitions, and exposure uncertainty. Calibration and evaluation event sets must be disjoint for each reported holdout experiment.

---

## 7 Inference

### 7.1 Target
$$p\big(\bm{f}_{0:T}, \bm{h}_{0:T}, \tilde{\bm{u}}_{1:T}, \xi_{1-D_{\max}:T}, \bm{\Theta} \mid N, B, E, [V], D\big)$$
with $z_{1:T}$ integrated out. NUTS over all of the above; no discrete sampling.

### 7.2 Regime marginalisation: what enters the recursion
Given all continuous latents and $\bm{\Theta}$, the only regime-dependent quantities are the levels $\bm{\mu}_r$, $\bm{\nu}_r$. Define the **emission log-density under regime $r$** at month $t$:

Define $\mathcal O^C_t$ as the indices with mask one for channel $C$ at $t$. Suppressing fixed covariates and parameters,
$$\mathcal{L}_t(r)=\sum_{k\in\mathcal O^N_t}\log p(N_{k,t}\mid\eta^{(r)}_{k,t})+\sum_{k\in\mathcal O^E_t}\log p(E^c_{k,t}\mid\eta^{(r)}_{k,t})+\sum_{k\in\mathcal O^B_t}\log p(B^c_{k,t}\mid\varsigma^{(r)}_{k,t})+\Big[\sum_{k\in\mathcal O^V_t}\log p(V_{k,t}\mid\eta^{(r)}_{k,t})\Big]+\sum_{s\in\mathcal O^D_t}\log p(D_{s,t}\mid\bm\eta^{(r)}_t).$$

with $\eta^{(r)}_{k,t} = m_k + c_r + (\bm{\Gamma}\bm{f}_t)_k + a_k g_t + \tau_k\tilde u_{k,t}$ and $\varsigma^{(r)}_{k,t} = n_k + d_r + (\bm{\Psi}\bm{h}_t)_k$. Each sum runs over unmasked cells only (§0).

Forward recursion in log space:
$$\log\alpha_1(r) = -\log R + \mathcal{L}_1(r),\qquad
\log\alpha_t(r) = \operatorname{logsumexp}_{r'}\big[\log\alpha_{t-1}(r') + \log\Pi_{r'r}\big] + \mathcal{L}_t(r),\qquad
\log p(\text{data}\mid\cdot) = \operatorname{logsumexp}_r \log\alpha_T(r).$$

The densities of $\bm{f}$, $\bm{h}$, $\tilde{\bm{u}}$, $\bm{\xi}$ do **not** depend on $z$ and are added **once**, outside the recursion, as ordinary prior terms. Nothing is counted twice: the recursion contains only $\mathcal{L}_t(r)$ and $\bm{\Pi}$.

Verification (Stage 7): on $T \le 6$, $R \le 3$ examples the recursion must equal explicit enumeration over $R^T$ paths to machine precision, and its gradient must match finite differences. (A review check of the recursion as written agreed with explicit enumeration to about $4 \times 10^{-16}$.)

### 7.3 Regime recovery after sampling
For each draw $m$: within-fit filtered probabilities $p(z_t = r \mid y_{1:t}, \text{latents}^{(m)}, \bm{\Theta}^{(m)}) \propto \alpha_t(r)$ (normalised), smoothed probabilities by the backward pass, and sampled paths $z^{(m)}_{1:T}$ by forward-filtering backward-sampling. Reported probabilities are averages over draws.

**[D] What counts as a regime alert.** The within-fit filtered probability at $t<T$ uses the window through $T$ through its continuous draws, so is retrospective. Assemble the alert series from terminal probabilities at successive origins. Call it historical real-time only for verified point-in-time inputs and preprocessing; otherwise label it a revised-data rolling-origin diagnostic. Within-fit filtered and smoothed curves remain retrospective.

### 7.4 Diagnostics (acceptance criteria for a fit)
Use multiple chains; require $\hat R<1.01$ and bulk/tail ESS above 400 for reported quantities, examine all monitored latent-state diagnostics and their worst cases, and assess Monte Carlo error directly for tail summaries. Target zero post-warm-up divergences; documenting a divergence alone does not make a fit acceptable. Diagnose unresolved divergences before substantive reporting. Check prior and posterior predictions for all enabled channels. These are proposed diagnostic thresholds, not a proof of convergence.

---

## 8 Forecast simulation

One trajectory per posterior draw $m$. The trajectory is initialised **once** and then advanced sequentially; nothing in the horizon loop re-draws the starting state.

**Initialisation (once per draw $m$):**

- $z^{(m)}_T \sim \text{Categorical}(\text{normalised } \alpha^{(m)}_T)$, the terminal filtered distribution of that draw.
- $\bm{f}^{(m)}_T$, $\bm{h}^{(m)}_T$, and $\xi^{(m)}_{T-D_{\max}+1:T}$ are the draw's sampled terminal values.
- Hold $e_{T+h}=e_T$ and $M_{\cdot,\cdot,T+h}=M_{\cdot,\cdot,T}$ by default. Advance legally scheduled $q_{T+h}$ using only announcements known at origin $T$, rather than freezing it across a known compliance date. A newly active coefficient can remain prior-driven before observations identify it.
- Assume the latest EPSS version known at $T$ continues unless a change was already announced. Never inject a subsequently learned version-change date into an earlier forecast. Report an explicit scenario or omit the EPSS target at horizons where its measurement definition cannot be fixed. All other enabled channels can still be evaluated.

**Advance, for $h = 1,\dots,H$ in order:**

1. $z^{(m)}_{T+h} \sim \bm{\Pi}^{(m)}_{z^{(m)}_{T+h-1},\cdot}$.
2. $\bm{f}^{(m)}_{T+h} = \bm{\Phi}^{(m)}\bm{f}^{(m)}_{T+h-1} + \bm{w}$, $\bm{w} \sim \mathcal{N}(\bm{0}, \bm{I})$; same for $\bm{h}$.
3. $\xi_{T+h} \sim \mathcal{N}(0,1)$; $g^{(m)}_{T+h} = \sum_{d=0}^{D_{\max}} w_d(b^{(m)})\,\xi_{T+h-d}$, using the draw's sampled $\xi$ for indices $\le T$ and the trajectory's fresh draws for indices in $(T, T+h]$.
4. $u_{k,T+h} \sim \mathcal{N}(0, \tau_k^2)$ fresh.
5. $\eta^{(m)}_{T+h}$, $\varsigma^{(m)}_{T+h}$ from §3.4 at the regime $z^{(m)}_{T+h}$; $\lambda = \exp\eta$, $\sigma = \operatorname{sigmoid}\varsigma$.
6. **Observable scenarios.** Draw $N$ from §5.1. For positive cohorts, draw potential marks from §5.2–5.3 and apply the specified coarsening; empty cohorts have no marks. These scenarios do not model whether scores will actually be published for each CVE. Draw $[V]$ and $D$ from their likelihoods; reporting-transition months excluded by the observation protocol remain unscored.
7. **Latent forecasts**: $\lambda^{(m)}_{T+h}$, $\sigma^{(m)}_{T+h}$, carried into the economic layer.

**Conditional mark forecasts for evaluation [D].** Score every test cell with an actual available mark, independent of whether a finite set of simulated counts happened to include positives. For the default homoscedastic channels, construct $p(B^c,E^c\mid N>0,\mathcal F_T)$ by weighting predictive state draws by
$$w_m=\Pr(N>0\mid\mu_m,\psi_m)=1-\left(\frac{\psi_m}{\psi_m+\mu_m}\right)^{\psi_m},$$
then normalising those weights and mixing the corresponding mark distributions. Use stable arithmetic for small $w_m$ and report Monte Carlo precision; insufficient simulation is a forecast-computation failure, not a reason to remove a test cell. These are conditional mark scores on the observed subset, not validation of the unmodelled score-availability process. Count forecasts are scored separately on all audited count cells.

**[OPEN, Stage 8]** The heteroscedastic alternatives depend on future $n^B,n^E$. Either model their joint availability with counts, or declare a separate conditional task given realised cohort/score counts and apply the same conditioning to all competitors. Do not substitute realised future counts into an unconditional forecast. Until this is settled, the homoscedastic versions define the primary mark benchmark.

Dependence across categories, sectors and horizons is preserved within a draw; marginal forecasts are never resampled independently before aggregation.

Storage format: one table indexed by (origin $T$, horizon $h$, target type, category or sector, draw $m$).

---

## 9 Forecast targets and what is scored

| Target | Type | Scored over | Primary score |
|---|---|---|---|
| $N_{k,T+h}$ | observable count | all audited count cells at eligible origin–target pairs | CRPS, log predictive mass |
| $N^{\text{tot}}_{T+h}$ | observable count (aggregated within draw) | target months with complete count coverage | CRPS; log mass only with a validated aggregate probability calculation |
| $E^c_{k,T+h}$ | coarsened probability mark, conditional on positive cohort | actual available marks in target months with supported EPSS versions and snapshots; no filtering by simulated counts | CRPS and mixed mass/density log score (§5) |
| $B^c_{k,T+h}$ | coarsened severity mark, conditional on positive cohort | all actual available marks under the declared vintage policy; no filtering by simulated counts | CRPS and mixed mass/density log score |
| $D_{s,T+h}$ | observable count | audited full target months from January 2024, excluding June 2024 in the default; a very short series | log score; descriptive unless a sufficient separate validation window is available |
| $\lambda_{k,T+h}$, $\sigma_{k,T+h}$ | latent | synthetic data only | recovery RMSE, coverage |
| $\ell^{\text{agg}}_{T+h}$, VaR, ES, $\Pr(\ell > c)$ | economic | held-out events, sensitivity | Stage 13, separately qualified |
| $p(z_T = r \mid \text{fit at } T)$ across origins | regime alert | synthetic data; descriptive on real data | recovery of known transitions |

Horizons: $h\in\{1,3,6,12\}$ reported; all 12 simulated. Availability is a property of the target month $T+h$, not merely the origin. The evaluation protocol fixes origin–target schedules, scoring, and failure-policy principles; Stage 4 adds audited coverage masks. Distinguish forecasts before a channel has any training data from fitted-channel forecasts. All competitors use the same target eligibility rule. The held-out forecast period for this rerun is January 2024–December 2025; no evaluation scores guide model selection.

---

## 10 Model versions (plan, Stage 8) in terms of this specification

| Version | Components switched on | Switched off |
|---|---|---|
| A | §5.1 with $\eta_{k,t}=m_k+u_{k,t}$, $u_{k,t}=\phi^u_k u_{k,t-1}+\tau_k\tilde u_{k,t}$, $\phi^u_k\sim\operatorname{Uniform}(-1,1)$ and $u_{k,0}\sim\mathcal N(0,\tau_k^2/(1-(\phi^u_k)^2))$; $e_t=0$ | factors, regimes, kernel, other channels |
| B | + $\bm{\Gamma}\bm{f}_t$ (§3.3, §3.4); residual $u_{k,t}$ becomes **iid** as in §3.4 | regimes, kernel, $e_t$, other channels |
| B$'$ | as B but retaining A's AR(1) residual | control for the residual change: A→B changes two things, B$'$ isolates the factor contribution |
| C | + severity process and §5.3 | regimes, kernel, $e_t$ |
| D | + $e_t$ offset (§2) | regimes, kernel |
| E | + regimes (§3.2), marginalisation (§7.2) | kernel |
| F | + kernel (§3.5) | |
| G | + §5.2, §5.4, §5.5 channels | |

Each version is a configuration of the same model code; ablations in Stage 14 reuse the switches.

The initial factor pilot has $r_f=2$, $R=1$, and no kernel; severity adds $r_h=2$. Regime challengers use $R=2,3$ on development data. At $R=1$, independent severity and intensity processes do not acquire count–severity dependence merely by fitting them together. Setting $\Gamma=0$ is a no-factor ablation, not automatically independent topics. Tenfold prior inflation is prior-scale sensitivity, not removal of hierarchy. See the protocol for controlled comparisons.

---

## 11 Open items carried to later stages

| Item | Stage |
|---|---|
| Data-derived coverage masks for every channel (NVD completeness, EPSS month-end snapshots, KEV, SEC) | 4 |
| EPSS month-end snapshot content and version handling (v1/v2/v3/v4 through 2025) for $E_{k,t}$ | 4 |
| Incident deduplication and sector mapping for $D_{s,t}$; registrant-count offset versus the step covariate $q_t$ | 4, 5 |
| Prior centre for $\rho_s$ once the scale of $\sum_k M\lambda$ is known | 5 |
| CNA-supplied CVSS as a month-end severity score; reporting of $n^B/N$ by month | 4 |
| Empirical assessment of fixed pilot $K=10$, $r_f=r_h=2$, $R=1$; regime, rank, resolution and kernel sensitivity | 8 |
| Kernel identifiability versus AR factors; joint recovery of $\tau_k$ and $\psi_k$ | 7 |
| Heteroscedastic $E$ and $B$ noise (using $n^E$, $n^B$); whether it identifies a latent severity noise | 8 |
| Joint score-availability model versus explicitly conditional mark evaluation for heteroscedastic alternatives | 2, 8 |
| Diagonal versus full $\bm{\Phi}$; iid versus AR(1) residual (B versus B$'$) | 8 |
| Stable differentiable parameterisation and stationary covariance for full $\bm{\Phi}$ | 8, 9 |
| Audited cell eligibility under the fixed origin schedule and seven-day EPSS tolerance; historical availability of topic metadata and announced measurement changes | 4, 5 |
| Exposure map construction and Leontief interpretation | 11 |
| Event and anchor likelihood; damage-function identification from four events plus the anchor; which uncertainties propagate | 12 |
