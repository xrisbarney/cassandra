"""NVD 2.0 feed parsing. Publication timestamps, never identifier years, define cohorts."""

import gzip
import hashlib
import json
from collections import Counter
from datetime import date

from .store import DownloadError


def parse_meta(text):
    return dict(line.split(":", 1) for line in text.splitlines() if ":" in line)


def validate_feed(path, metadata):
    with gzip.open(path, "rb") as stream:
        digest = hashlib.sha256()
        size = 0
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    if digest.hexdigest().lower() != metadata["sha256"].lower() or size != int(metadata["size"]):
        raise ValueError("NVD uncompressed feed checksum/size mismatch.")


def cpe_matches(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "cpeMatch":
                yield from nested
            else:
                yield from cpe_matches(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from cpe_matches(nested)


def normalize_cve(cve, source_id, source_hash):
    identifier = cve["id"]
    published = cve["published"]
    date.fromisoformat(published[:10])
    descriptions = sorted(
        {
            entry["value"].strip()
            for entry in cve.get("descriptions", [])
            if entry.get("lang") == "en" and entry.get("value", "").strip()
        }
    )
    metrics = cve.get("metrics", {})
    scores = []
    for family, entries in metrics.items():
        for entry in entries:
            data = entry.get("cvssData", {})
            score = data.get("baseScore")
            if score is not None:
                if not 0 <= float(score) <= 10:
                    raise ValueError("CVSS score outside [0,10].")
                scores.append(
                    {
                        "family": family,
                        "version": data.get("version"),
                        "source": entry.get("source"),
                        "type": entry.get("type"),
                        "base_score": score,
                        "vector": data.get("vectorString"),
                    }
                )
    matches = list(cpe_matches(cve.get("configurations", [])))
    return {
        "cve_id": identifier,
        "published": published,
        "publication_month": published[:7],
        "last_modified": cve["lastModified"],
        "status": cve.get("vulnStatus", "unknown"),
        "rejected": cve.get("vulnStatus", "").lower() == "rejected",
        "description_en": "\n".join(descriptions),
        "cvss_count": len(scores),
        "cvss_json": json.dumps(scores, sort_keys=True),
        "cpe_match_count": len(matches),
        "vulnerable_cpe_count": sum(bool(m.get("vulnerable")) for m in matches),
        "source_id": source_id,
        "source_hash": source_hash,
        "metadata_available_at": "unknown",
        "vintage_track": "revised",
    }


def ingest_nvd(store, output, *, last_year, progress=print):
    import pandas as pd

    rows = {}
    feeds = []
    missing = []
    identical_duplicates = 0
    # The oldest standard annual feed includes older identifier years. Probe 1999–2001
    # too: an unavailable early feed is acceptable only with the 2002 feed verified.
    for year in range(1999, last_year + 1):
        base = f"https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-{year}"
        meta_name = f"nvd/{year}.meta"
        try:
            meta_path, _ = store.fetch(meta_name, base + ".meta")
        except DownloadError:
            if year < 2002 and store.records.get(meta_name, {}).get("http_status") == 404:
                continue
            missing.append(year)
            continue
        metadata = parse_meta(meta_path.read_text())
        name = f"nvd/{year}.json.gz"
        path, record = store.fetch(name, base + ".json.gz")
        validate_feed(path, metadata)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            feed = json.load(stream)
        if feed["totalResults"] != len(feed["vulnerabilities"]):
            raise ValueError("NVD feed count differs from totalResults.")
        store.describe(
            name,
            source_date=feed.get("timestamp", "unknown"),
            source_version=feed.get("version", "unknown"),
            uncompressed_sha256=metadata["sha256"].lower(),
            feed_last_modified=metadata["lastModifiedDate"],
        )
        retained = 0
        for item in feed["vulnerabilities"]:
            cve = item["cve"]
            if not "2010-01" <= cve["published"][:7] <= "2025-12":
                continue
            row = normalize_cve(cve, name, record["sha256"])
            old = rows.get(row["cve_id"])
            if old:
                comparable = {k: v for k, v in row.items() if k not in ("source_id", "source_hash")}
                prior = {k: v for k, v in old.items() if k not in ("source_id", "source_hash")}
                if comparable != prior:
                    raise ValueError("Conflicting duplicate CVE across pinned feeds.")
                identical_duplicates += 1
            else:
                rows[row["cve_id"]] = row
                retained += 1
        feeds.append(
            {
                "year": year,
                "feed_records": feed["totalResults"],
                "cohort_records": retained,
                "uncompressed_hash_verified": True,
            }
        )
        del feed
        progress(f"NVD {year}: feed verified; {retained} publication-window records", flush=True)
    frame = pd.DataFrame(rows.values()).sort_values("cve_id")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "nvd_records.parquet", index=False)
    summary = []
    for month in pd.period_range("2010-01", "2025-12", freq="M").astype(str):
        block = frame[frame.publication_month == month]
        accepted = block[~block.rejected]
        summary.append(
            {
                "month": month,
                "all_records": len(block),
                "rejected_records": int(block.rejected.sum()),
                "nonrejected_records": len(accepted),
                "missing_english": int((accepted.description_en == "").sum()),
                "missing_cvss": int((accepted.cvss_count == 0).sum()),
                "missing_cpe": int((accepted.cpe_match_count == 0).sum()),
                "modified_after_month_end": int((accepted.last_modified.str[:7] > month).sum()),
                "feed_coverage_verified": not missing,
            }
        )
    pd.DataFrame(summary).to_csv(output / "nvd_monthly_coverage.csv", index=False)
    return {
        "feeds": feeds,
        "missing_feed_years": missing,
        "unique_window_records": len(frame),
        "identical_duplicates": identical_duplicates,
        "statuses": dict(Counter(frame.status)),
        "all_feeds_verified": not missing,
        "monthly": summary,
    }, frame
