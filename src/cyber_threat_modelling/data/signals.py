"""Separate EPSS probability snapshots and KEV catalogue-addition records."""

import csv
import gzip
import json
from datetime import date, timedelta

from .store import DownloadError


def read_epss(path):
    """Preserve the source header; absent identifiers are never assigned score zero."""
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        first = stream.readline()
        metadata = first.strip() if first.startswith("#") else "unknown"
        if not first.startswith("#"):
            stream.seek(0)
        reader = csv.DictReader(stream)
        if not {"cve", "epss"}.issubset(reader.fieldnames or []):
            raise ValueError("Unexpected EPSS columns.")
        scores = {}
        for row in reader:
            score = float(row["epss"])
            percentile = float(row["percentile"]) if row.get("percentile") else None
            if not 0 <= score <= 1 or (percentile is not None and not 0 <= percentile <= 1):
                raise ValueError("EPSS probability/percentile outside [0,1].")
            if row["cve"] in scores:
                raise ValueError("Duplicate CVE in EPSS snapshot.")
            scores[row["cve"]] = score
    return metadata, scores


def select_snapshot(month_end, available_dates, tolerance=7):
    eligible = [
        d for d in available_dates if timedelta(0) <= month_end - d <= timedelta(days=tolerance)
    ]
    return max(eligible) if eligible else None


def epss_version(day):
    for start, version in [
        ("2025-03-17", "v4"),
        ("2023-03-07", "v3"),
        ("2022-02-04", "v2"),
        ("2021-04-14", "v1"),
    ]:
        if day.isoformat() >= start:
            return version
    return "unavailable"


def ingest_epss(store, output, nvd, progress=print):
    import pandas as pd

    root = "https://api.github.com/repos/empiricalsec/epss_scores"
    path, _ = store.fetch("epss/commit.json", root + "/commits/main")
    commit = json.loads(path.read_text(encoding="utf-8"))["sha"]
    listings = {}
    for year in range(2021, 2026):
        path, _ = store.fetch(f"epss/list-{year}.json", f"{root}/contents/{year}?ref={commit}")
        listings.update(
            {
                date.fromisoformat(item["name"][12:22]): item["path"]
                for item in json.loads(path.read_text(encoding="utf-8"))
                if item["name"].startswith("epss_scores-") and item["name"].endswith(".csv.gz")
            }
        )
    coverage, cohorts = [], []
    for month in pd.period_range("2021-04", "2025-12", freq="M"):
        end = month.end_time.date()
        chosen = select_snapshot(end, listings)
        block = nvd[(nvd.publication_month == str(month)) & ~nvd.rejected]
        row = {
            "month": str(month),
            "cohort_size": len(block),
            "snapshot_date": None,
            "matched_cves": 0,
            "snapshot_available": False,
            "available_at": "unknown",
        }
        if chosen:
            name = f"epss/{chosen}.csv.gz"
            try:
                path, record = store.fetch(
                    name,
                    f"https://raw.githubusercontent.com/empiricalsec/epss_scores/{commit}/{listings[chosen]}",
                )
                header, scores = read_epss(path)
            except DownloadError:
                row["reason"] = "download_failed"
                coverage.append(row)
                continue
            if header != "unknown" and "score_date:" in header and chosen.isoformat() not in header:
                raise ValueError("EPSS filename and score-date header disagree.")
            for identifier in block.cve_id:
                if identifier in scores:
                    cohorts.append(
                        {
                            "cve_id": identifier,
                            "month": str(month),
                            "epss": scores[identifier],
                            "snapshot_date": chosen.isoformat(),
                            "source_hash": record["sha256"],
                            "model_version": epss_version(chosen),
                        }
                    )
            matches = sum(identifier in scores for identifier in block.cve_id)
            row.update(
                snapshot_date=chosen.isoformat(),
                snapshot_age_days=(end - chosen).days,
                matched_cves=matches,
                snapshot_population=len(scores),
                snapshot_available=True,
                model_version=epss_version(chosen),
                header=header,
                source_hash=record["sha256"],
            )
            store.describe(
                name,
                source_date=chosen.isoformat(),
                source_version=header,
                archive_commit=commit,
                available_at="unknown",
            )
        coverage.append(row)
        progress(f"EPSS {month}: snapshot audited", flush=True)
    pd.DataFrame(coverage).to_csv(output / "epss_monthly_coverage.csv", index=False)
    pd.DataFrame(cohorts).to_parquet(output / "epss_cohort_scores.parquet", index=False)
    return {
        "archive_commit": commit,
        "months": coverage,
        "usable_snapshots": sum(r["snapshot_available"] for r in coverage),
        "matched_cohort_rows": len(cohorts),
        "historical_publication_time_verified": False,
    }


def parse_kev(data):
    entries = data["vulnerabilities"]
    if data["count"] != len(entries):
        raise ValueError("KEV declared count mismatch.")
    seen = set()
    for item in entries:
        if item["cveID"] in seen:
            raise ValueError("Duplicate CVE in KEV catalogue.")
        seen.add(item["cveID"])
        date.fromisoformat(item["dateAdded"])
    return entries


def ingest_kev(store, output, nvd):
    import pandas as pd

    name = "kev/catalog.json"
    path, record = store.fetch(
        name, "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
    )
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(parse_kev(data))
    frame["source_hash"] = record["sha256"]
    frame["addition_month"] = frame.dateAdded.str[:7]
    frame["in_nvd_publication_window"] = frame.cveID.isin(nvd.cve_id)
    frame.to_parquet(output / "kev_catalog.parquet", index=False)
    store.describe(name, source_date=data["dateReleased"], source_version=data["catalogVersion"])
    return {
        "catalog_version": data["catalogVersion"],
        "records": len(frame),
        "launch_month_entries_in_current_catalog": int((frame.addition_month == "2021-11").sum()),
        "window_additions": int(frame.addition_month.between("2021-12", "2025-12").sum()),
        "window_additions_missing_nvd_cohort": int(
            (
                frame.addition_month.between("2021-12", "2025-12")
                & ~frame.in_nvd_publication_window
            ).sum()
        ),
        "historical_catalog_removals_reconstructed": False,
    }
