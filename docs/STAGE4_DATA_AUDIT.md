# Stage 4: source ingestion and coverage audit

Completed 2026-10-20. The source-adapter, retrieval-integrity, coverage-report and
historical-limitations deliverables are complete. The primary revised-history
count source can proceed to Stage 5. Auxiliary incident and economic channels
retain the explicit gates below. No topic map, model, forecast, score or economic
calibration was fitted. The held-out forecast period remains January 2024–December
2025, and its outcomes have only been inspected for this source-coverage audit.

## Audited artifacts

Snapshot: `stage4-20261020`. Raw responses are local under
`data/raw/stage4-20261020`; normalized records are under
`data/interim/stage4-20261020`. The [data dictionary](DATA_DICTIONARY.md) defines
record/status handling, missingness, provenance and pending target construction.
The [data manifest](audits/stage4-20261020/data_manifest.json) freezes raw and
derived hashes, code/policy hashes, the protocol hash and the software environment.
The [source manifest](audits/stage4-20261020/source_manifest.json) gives individual
URLs, retrieval times, sizes, source versions/dates and unknown historical
availability. These are new measurements, not values carried from the failed run.

| Source | Observed result | What it supports |
|---|---|---|
| NVD | 25 annual feeds, 2002–2026, passed decompressed SHA-256/size and declared-count checks | Complete retrieval of this pinned annual-feed universe |
| Publication window | 285,067 unique CVEs published in 2010–2025; no duplicate IDs | Record inventory across all 192 months |
| Record status | 269,057 non-rejected; 16,010 rejected retained in the audit | Declared revised-status exclusion, not historical status reconstruction |
| English text | No missing English description among non-rejected records | Input text available for Stage 5 |
| CVSS | 908 non-rejected records have no score in any preserved CVSS family | Missing marks remain missing; deterministic selection is Stage 5 |
| CPE | 25,730 non-rejected records have no configuration matches | Exposure coverage is incomplete; counts remain usable |
| EPSS | 57/57 month-end snapshots, April 2021–December 2025, all age zero | Archive contents verified, including all 24 held-out months |
| EPSS cohort join | 152,957 matched CVE-month rows; 85,856 of 88,105 non-rejected held-out CVEs matched | Absent scores are missing, not zero |
| KEV | Current version 2026.10.20: 1,754 records; 1,193 additions dated December 2021–December 2025 | Revised catalogue-addition signal, distinct from exploitation onset |
| SEC | 105 candidate documents, 104 accessions, 19 amendment documents; 58 Item 1.05 documents retrieved | Candidate collection only; no incident counts certified |
| BEA | Summary use, domestic supply and industry-by-industry total-requirements tables cover 2010–2025 | Source inputs for the later economic audit; no A matrix constructed |
| Events | Four scope-reviewed primary-source entries; none eligible for national monthly loss calibration | Explicit exclusion evidence rather than assumed loss targets |

For NVD, 1999–2001 standalone feed probes returned HTTP 404; the oldest standard
feed is 2002. Earlier identifier-year feeds contribute records published within
the study window, demonstrating why filtering filenames to 2010–2025 would be
incorrect. Selection uses each record's publication timestamp. Integrity checks
verify our retrieval, not NVD's completeness relative to all real vulnerabilities.

| Partition | All records | Non-rejected | Rejected | Missing any CVSS | Missing CPE |
|---|---:|---:|---:|---:|---:|
| Initial, 2010–2018 | 76,901 | 71,287 | 5,614 | 0 | 0 |
| Development, 2019–2023 | 117,490 | 109,665 | 7,825 | 3 | 0 |
| Held-out, 2024–2025 | 90,676 | 88,105 | 2,571 | 905 | 25,730 |

Missing-score/CPE columns refer to non-rejected records. The monthly breakdowns
are [NVD](audits/stage4-20261020/nvd_monthly_coverage.csv),
[EPSS](audits/stage4-20261020/epss_monthly_coverage.csv), and
[SEC search coverage](audits/stage4-20261020/sec_search_coverage.csv).
These are data audit tables, not model performance results.

## Historical information limits

Every non-rejected record in this snapshot has a last-modified month later than
its publication month. A modification does not prove every field changed, but
the current record cannot establish the old field values. Three targeted NVD
history queries returned 15, 71 and 22 events for CVE-2010-3333, CVE-2021-44228 and
CVE-2024-3400. They expose useful change details, including some score and CPE
changes; they do not certify complete text/score/product snapshots for the entire
panel at every cutoff. The point-in-time track therefore remains uncertified.

EPSS content was fetched from commit
`133be40523ea21100539ee08265d1aa91ec3190b`. Early archives contain only CVE and
score columns; later formats add percentiles and metadata. The parser handles
both. Score dates and version steps are preserved, but an archive filename is not
proof of the exact original publication time. [FIRST archive documentation](https://www.first.org/epss/data.html).

The current KEV snapshot contains 291 entries dated November 2021. This is a
count in today's catalogue, **not** a newly verified launch-day count. November
remains excluded regardless. Historical catalogue removals are not reconstructed.
Twenty-one additions in the usable date window refer to CVEs outside the NVD
publication cohort; Stage 5 must obtain their text from the raw feeds or use the
reserved unknown category, never silently drop them.

## Auxiliary-channel gates

**SEC.** Monthly candidate searches used root form 8-K and the phrase
`material cybersecurity`. Observed EFTS behavior makes combined `8-K,8-K/A`
queries restrict the exact form to amendments; `filter_items` was ignored.
The adapter checks the returned query and pagination instead of assuming those
filters work. All 58 Item 1.05 documents downloaded successfully; a content check
found Item 1.05 and the Commission heading in each, with no detected access-block
page. Candidates still need semantic review, amendment/repeat linkage,
incident–registrant identification, sector mapping and a population/recall audit.
All D cells remain masked. A completed search is not a complete incident census.

**BEA.** Retrieved table 259 has 76,115 rows, table 262 has 25,983, and table 61
has 81,792; duplicate table/year/row/column coordinates are rejected. Table-family
unit conventions are recorded with confirmation pending. Current revisions,
price bases, classifications, units and gross-output identities must be reconciled
before Stage 11 uses these inputs. Supply/use and requirements tables represent
different objects; directly inverting a rectangular Use table is not implemented.
[BEA table guide](https://www.bea.gov/resources/guide-interactive-industry-input-output-accounts-tables).

**Event evidence.** The register separates company accounting impacts from
national output loss. The DOJ's Colonial Pipeline figure is recovered currency,
not production loss. UnitedHealth's annual company impact combines response
costs and reduced revenues. Maersk and Progress reporting likewise does not
directly supply a US monthly aggregate-output target. No old event range is used.
The DOJ and UnitedHealth documents were stored successfully; direct downloads
of the selected Maersk and Progress URLs returned HTTP 403, recorded in the
manifest. Their source pages were reviewed through the web reader, but no raw
download success is claimed. The [event register](../configs/event_evidence.json)
retains those access and scope limitations; Stage 12 still needs compatible
calibration/evaluation evidence.

## Execution and verification

Run from the repository root with the science environment installed:

```powershell
.\.venv\Scripts\ctm.exe audit-data --snapshot NEW_ID --credentials-file C:\path\to\.env --finalize
.\.venv\Scripts\ctm.exe verify-data docs/audits/stage4-20261020/data_manifest.json
.\.venv\Scripts\ctm.exe prepare-run configs/pilot.toml --output artifacts/runs/stage4-data-check --data-manifest docs/audits/stage4-20261020/data_manifest.json
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
```

The raw store is resumable before finalization and verifies cached bytes. A
finalized snapshot cannot be overwritten. A newly downloaded mutable source may
differ from this snapshot; it gets a new snapshot ID and audit, not the old hash.
BEA responses echo API credentials, so raw artifacts stay local and ignored by
Git. Exported request URLs are redacted and published files are checked against
the actual credential values before commit.

The test suite covers original scaffold behavior plus feed checksums,
publication-month selection, historical EPSS formats/cutoffs, missing versus zero
scores, KEV duplication, SEC candidate semantics, evidence-type exclusions,
cache corruption, data-manifest tampering/path traversal/vintage mismatch,
frozen-audit overwrite protection, and attachment to run provenance.

Observed verification: **37 tests passed**, Ruff clean, dependency lock current,
source and derived manifest hashes verified, and source/wheel distributions built.
The five Stage 2 frozen-file hashes still match; the data audit did not modify
the scientific protocol. The attached-run check records the data-manifest hash
and retains `scoring_ready=false`.

The 1,500 rows in the source-eligibility artifact enumerate development/evaluation
origin/horizon/channel combinations. Their source-availability flags can pass;
their category is still `unassigned_stage5`, and observed/scoring flags remain
false. Stage 5 must construct topic/sector cells and refine this manifest before
fitting. The primary technical data gate is clear; this is not permission to
score the held-out period before development and the other protocol gates pass.
