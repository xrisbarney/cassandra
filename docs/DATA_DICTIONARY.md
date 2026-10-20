# Ingestion data dictionary, version 1

2026-10-20. Source-dependent decisions within the Stage 2 protocol, made before
topic construction or forecast scoring. The audit manifest hashes this file.

## Scope and provenance

The primary track is revised historical data. Retain raw HTTP payloads, retrieval
timestamps, byte hashes, source identifiers, revision dates where supplied, and
explicit `unknown` historical publication/availability times. A feed timestamp,
an EPSS filename, and an HTTP Last-Modified header have different meanings;
none alone proves historical availability of every field.

`data/raw/<snapshot>/` contains original response bodies and download sidecars.
Successful downloads are reused only if their URL and SHA-256 match. NVD metadata
checksums are verified against decompressed JSON bytes. HTTP error bodies remain
separate and never become observations. `data/interim/<snapshot>/` contains
normalized Parquet files and coverage outputs. `docs/audits/<snapshot>/` contains
publishable manifests and aggregate coverage; raw responses stay out of Git.

BEA echoes API credentials inside its response's Request section. Preserve that
raw response locally, but never publish it. Export only Results to normalized
tables and redact credentials in request URLs; the source manifest marks these
raw files. The credential loader accepts only NVD_API_KEY, BEA_API_KEY and
SEC_USER_AGENT from the explicitly supplied file or environment.

## NVD records and CPE

- Download all standard identifier-year feeds from 2002 through the retrieval
  year, including the oldest feed's earlier identifiers. Probe 1999–2001 as well;
  their HTTP 404s are permitted only as absent optional early feeds. A failed
  2002-or-later feed prevents coverage certification.
- Retain CVEs whose **publication month** is January 2010–December 2025. Exact
  duplicate IDs may collapse only if normalized fields agree; conflicting
  duplicate records stop ingestion.
- Preserve every status. The primary Stage 5 cohort will exclude `Rejected`
  records, retain all other published records including deferred/unanalysed ones,
  and separately report excluded records. Empty text maps to unknown later;
  missing CVSS/CPE never removes a count observation. This revised-status policy
  cannot establish the count that would have been known at an old cutoff.
- `published`, `last_modified`, `publication_month`, `description_en`, `status`,
  `rejected`, source ID/hash and explicit metadata availability are record fields.
  English descriptions are deduplicated and joined deterministically.
- Preserve all CVSS entries as `cvss_json`, including family, version, source,
  primary/secondary type, score and vector. No monthly severity target is formed
  during Stage 4. Stage 5 selects one score using version order 4.0, 3.1, 3.0, 2.0;
  within version prefer NVD Primary, other Primary, NVD Secondary, other entries,
  then lexical source and vector order. Retain selected version/source and report
  version-specific coverage. Ties with conflicting values require explicit review.
  Mixing versions is a declared limitation, with separate version diagnostics.
- CPE match counts recurse through configuration nodes; vulnerable matches are
  counted separately from environmental conditions. The complete logical tree,
  range bounds and match criteria remain in the hashed raw CVE record. Counts are
  audit fields, not a flattened sector-exposure map.
- `feed_coverage_verified` certifies retrieval integrity of the pinned feed
  universe. It does not assert complete reporting of real vulnerabilities or
  reconstruct historical disclosure/enrichment timing.

## EPSS and KEV

EPSS uses a pinned Git archive commit and the latest available snapshot no later
than month-end and at most seven days old. Historical two-column CSVs (`cve,epss`)
are supported along with later percentile/header formats. Scores must lie in
[0,1]; duplicate CVEs are errors. A missing CVE is absent, never zero. Join to
the non-rejected publication-month cohort. Retain source header, score date,
snapshot age, version regime and hash. Version labels follow the dated FIRST
releases; archive score dates do not certify original release timestamps.

KEV is a separate current-catalogue snapshot. Preserve its version, declared
count, addition dates and all source fields. Duplicate IDs and count mismatches
are errors. November 2021 is excluded as launch backlog. Additions after that
are catalogue additions, not exploitation onset. Historical removals/revisions
are not reconstructed. CVEs outside the NVD publication cohort require a separate
text lookup or unknown assignment during Stage 5; they must not be silently lost.

## SEC

The initial adapter searches monthly from December 2023 through December 2025
for the phrase `material cybersecurity` under root form `8-K`. It checks the
returned query, total, pagination and dates. Search recall is not certified.
The observed service treats combined `8-K,8-K/A` as an amendment constraint and
ignores `filter_items`; the adapter avoids both assumptions.

Preserve document ID, accession, filing date, exact form, all registrant CIKs,
SIC codes and Item 1.05 metadata. Download the primary documents of Item 1.05
candidates; stop after three access failures and keep explicit retrieval flags.
Every row starts as `review_status=pending`,
`incident_id=null`, `observed_incident=false`. Distinct documents/accessions are
not distinct incidents. Stage 5 must retrieve qualifying filing text, adjudicate
incidents, link amendments/repeats, preserve incident–registrant pairs, map
sectors, and audit the reporting population. No incident zero is inferred here.
All D cells remain masked, including the predeclared December 2023/June 2024
transition exclusions. Pre-mandate text matches are outside this initial adapter.

## BEA and event evidence

Retrieve InputOutput summary tables 259 (use), 262 (domestic supply), and 61
(industry-by-industry total requirements) for 2010–2025. Preserve row/column codes,
labels, row types, years, source value strings, and notes in raw JSON. No sector
aggregation, A matrix, inversion, or monetary forecast is computed. The source
table descriptions identify different accounting objects; Stage 11 must confirm
units, price conventions, row/column compatibility, gross output and the shock
interpretation. Current tables do not establish vintages available at past origins.

The event register records source URL/publication date, geographic and accounting
scope, currency/price year, measurement period, amount bounds when actually
extracted, evidence type and exclusion reason. Numeric point figures are not
invented 90% intervals. Company expenses, ransom recoveries and national
gross-output losses are different targets. None of the initial four records is
eligible for the proposed national monthly economic calibration.

## Eligibility and finalization

Finalization is one-time per snapshot identifier; updates require a new snapshot.
The Stage 4 manifest freezes source hashes, the publication window, code hashes,
dictionary/event-register hashes, and the protocol hash. `source_eligibility.csv`
enumerates the prescribed development/evaluation origins, horizons and channels.
It distinguishes source availability from observed topic/sector cells. Category
is `unassigned_stage5`, observed/scoring flags are false until Stage 5 constructs
the panels and refines the cell manifest. This prevents preliminary source
coverage from being mistaken for permission to fit or score.

Point-in-time eligibility is not certified. Source manifests can be attached to
prepared runs only with a matching vintage and after all raw/derived hashes pass.
Model identities, fit status and forecast hashes are populated by the later
backtest runner; they are not fabricated in the ingestion manifest.

Sources: [NVD feeds](https://nvd.nist.gov/vuln/data-feeds),
[FIRST bulk and historical data](https://www.first.org/epss/data.html),
[SEC EDGAR access](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data),
[BEA API guide](https://apps.bea.gov/api/_pdf/bea_web_service_api_user_guide.pdf).
