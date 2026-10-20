# Monthly panel dictionary, version 1

2026-10-20. Stage 5 elaborates the frozen design without changing the Stage 4
dictionary or source snapshot. No forecast scores enter these decisions.

## Text map and counts

`configs/panel.json` pins the implementation settings before fitting. Train only
on non-rejected CVEs published January 2010–December 2018, sorted by identifier.
English descriptions use Unicode accent stripping, lowercase, English stop words,
letter/underscore tokens of length at least two, unigrams and bigrams, minimum
document frequency five, maximum fraction 0.95, and at most 20,000 features.
Use sublinear term frequencies, smoothed IDF and L2 normalization. These fixed
engineering settings were not chosen by forecasting results.

NMF uses nine components, NNDSVDa initialization, coordinate descent, Frobenius
loss, no regularization, seed 20261020, tolerance 1e-4 and a 1,000-iteration cap.
A convergence warning fails the build. BLAS/OpenMP work is limited to one thread
for reproducibility. Transform every cohort with the same fitted vocabulary,
IDF and components; never refit at later origins. Save vocabulary as JSON and
arrays as NPZ without executable pickle state. The [scikit-learn vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
and [NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
document the underlying algorithms and transform interfaces.

Topics are IDs 1–9 in component order; ID 10 is unknown. Assign the largest
nonnegative coefficient, breaking ties toward the lowest ID. Empty text, no
vocabulary terms, or all-zero coefficients map to unknown, with distinct reason
codes. `topic_max_share` is the largest coefficient divided by their sum; it is
a descriptive loading share, not a calibrated classification probability.
The training-ID register, top ten terms, three representative initial-period
CVEs and document-level NPMI support inspection. Coherence is descriptive only.

Each retained CVE contributes exactly one integer count in its publication month.
The topic panel has every month/category combination from 2010-01 to 2025-12.
An audited empty cell has N=0 and m_N=true; missing source coverage gives N=null
and m_N=false. Excluded rejected identifiers remain in a separate register.
All ten categories stay in the primary target. Active strata use at least 50
assignments and 24 positive months in the initial period alone.

## Severity, exploitation and KEV

Select CVSS using the version/source precedence fixed in the Stage 4 dictionary:
4.0, 3.1, 3.0, 2.0; NVD Primary, other Primary, NVD Secondary, other entries;
then lexical source and vector. Conflicting values at the winning identical
rank stop construction. Preserve selected version, source, vector and score.
`n_B` counts selected scores; B is their arithmetic mean, not severity-weighted
frequency. `B_coarsened` clips that mean to [0.05,9.95]. m_B requires source
coverage and n_B>0. Empty cohorts and positive cohorts without marks have
different reasons; neither receives an invented severity value. Version counts
are exported by month. Scores and descriptions retain the revised-data caveat.

EPSS joins the publication-month cohort to the pinned month-end snapshot, with
the seven-day tolerance and absent-CVE-as-missing rule. `n_E`, E and m_E mirror
the severity definitions, with E the available-score arithmetic mean and
`E_coarsened` clipped to [0.0001,0.9999]. Retain date, age, version and source hash.
The target eligibility table conservatively excludes version/category combinations
without observed training support by the origin, including the initial version
and version transitions. Future-announcement handling requires an origin-aware
forecast implementation; source presence alone does not establish support.

KEV V counts addition dates independently of publication cohorts. November 2021
is retained as `V_catalogued` for auditing but masked as launch backlog; V is null
before December 2021. In later months audited zeros remain observed. Join the
same retained CVE assignment; older/out-of-window or excluded identifiers map to
unknown with an explicit reason. No additions are silently lost. This uses the
current pinned catalogue, without reconstructing removals or old descriptions.

## SEC and sectors

`configs/sec_review.csv` records a disposition for every Stage 4 search hit.
All available Item 1.05 primary documents were inspected by the assistant; this
is not an independent human dual review. File hashes bind decisions to evidence.
Non-Item-1.05 hits are excluded by source item metadata. Link each amendment or
repeat through its narrative/original-report reference; a regular 8-K can be a
repeat too. Preserve one row per event and registrant, so a multi-registrant
filing retains all affected registrants. Do not infer shared events merely from
a shared suspected attacker. Dates are filing dates, not discovery dates.

The deduplicated register reports the first qualifying record **in this search
sample**, not a certified first disclosure in all EDGAR. Prior reports outside
the sample are flagged. Inclusion means an actual incident described under
Item 1.05, not an independent finding that it met a legal materiality threshold.
Foreign subsidiary incidents remain registrant disclosures; they do not establish
losses within the United States. Loss amounts are not calibration inputs here.

`configs/sec_sector_map.json` provides a provisional explicit mapping of the
reviewed registrants' SIC codes to codes in the pinned BEA summary table. SIC
describes a registrant's business in [SEC filings](https://www.sec.gov/search-filings/standard-industrial-classification-sic-code-list).
This analyst crosswalk is not an official concordance: software/publishing maps
to 511, communications to 513, REIT operations to ORE and broad miscellaneous
credit (including Coinbase's recorded SIC) to 521CI. These approximations require
Stage 11 review before economic use. Assign the registrant's sector, even when a
subsidiary is affected. Ambiguous/missing SIC array alignment maps to unknown.

The sector grid includes all sector codes from the pinned BEA requirements table
plus unknown. `reviewed_pairs_first_in_sample` is a descriptive sample count.
D remains null and m_D=false **for all cells**, because search recall, first
disclosures, and the eligible-registrant population are uncertified. Never use
the number of affected filers as the reporting-population denominator. That
denominator stays null. q=1 from July 2024 implements the specified approximate
population step only; it is not a measured offset. December 2023 and June 2024
retain explicit partial-month exclusions. Constructing this masked panel does
not clear the incident-channel gate.

## Files and eligibility

Local immutable panels live under `data/processed/<panel-id>`; compact summaries,
coverage plots, training topic descriptions and the panel manifest live under
`docs/panels/<panel-id>`. Build refuses an existing destination. The manifest
records upstream manifest hash, code/config/policy hashes, source revision,
software versions and hashes of every generated file. Code is committed before
the production build so the snapshot can cite a clean implementation revision.

`cell_eligibility.parquet` enumerates the fixed development/evaluation origins
and horizons for all channels and categories/sectors. It records origin, target,
source date, upstream manifest hash, vintage, observed flag, target eligibility
and exclusion reason. `available_at=unknown` preserves the point-in-time
limitation; a target-month date is not a verified release timestamp.
`target_eligible` describes observable targets after the origin-aware EPSS gate.
`eligible_for_scoring` stays false until Stage 6 execution checks. No fitting or
scoring is triggered by panel construction. The initial benchmark uses counts.

The original source manifest and frozen Stage 2 documents remain immutable.
