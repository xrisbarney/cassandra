"""Auditable SEC search candidates; incident adjudication remains an explicit gate."""

import json
from urllib.parse import urlencode

from .store import DownloadError


def sec_candidate(hit):
    source = hit["_source"]
    return {
        "document_id": hit["_id"],
        "accession": source["adsh"],
        "filing_date": source["file_date"],
        "form": source["form"],
        "ciks": source.get("ciks", []),
        "items": source.get("items", []),
        "sics": source.get("sics", []),
        "names": source.get("display_names", []),
        "item_105": "1.05" in source.get("items", []),
        "amendment": source["form"] == "8-K/A",
        "incident_id": None,
        "review_status": "pending",
        "observed_incident": False,
    }


def ingest_sec(store, output, user_agent, progress=print):
    import pandas as pd

    if not user_agent:
        return {
            "status": "unavailable",
            "reason": "SEC_USER_AGENT_not_provided",
            "incident_counts_eligible": False,
        }
    candidates, coverage = {}, []
    # Root 8-K includes amendments. Combining 8-K and 8-K/A constrains results to
    # amendments on the observed EFTS implementation; filter_items is ignored.
    for month in pd.period_range("2023-12", "2025-12", freq="M"):
        offset, expected = 0, None
        complete = True
        while True:
            params = {
                "dateRange": "custom",
                "startdt": str(month.start_time.date()),
                "enddt": str(month.end_time.date()),
                "forms": "8-K",
                "q": '"material cybersecurity"',
                "from": offset,
                "size": 100,
            }
            name = f"sec/search-{month}-{offset}.json"
            try:
                path, record = store.fetch(
                    name,
                    "https://efts.sec.gov/LATEST/search-index?" + urlencode(params),
                    headers={"User-Agent": user_agent},
                    interval=0.5,
                )
            except DownloadError:
                complete = False
                break
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("timed_out") or result.get("_shards", {}).get("failed", 0):
                raise ValueError("Incomplete SEC search response.")
            query = result["query"]["query"]["bool"]
            filters = query["filter"]
            if not any(f.get("terms", {}).get("root_forms") == ["8-K"] for f in filters):
                raise ValueError("SEC did not honor the expected root-form filter.")
            if any("form" in f.get("terms", {}) for f in filters):
                raise ValueError("SEC unexpectedly restricted the exact form.")
            total = result["hits"]["total"]
            if total["relation"] != "eq" or total["value"] > 10000:
                raise ValueError("SEC query requires smaller partitions.")
            if expected is not None and expected != total["value"]:
                raise ValueError("SEC search population changed during pagination.")
            expected = total["value"]
            hits = result["hits"]["hits"]
            for hit in hits:
                row = sec_candidate(hit)
                row["source_hash"] = record["sha256"]
                if row["document_id"] in candidates:
                    raise ValueError("Duplicate SEC search hit across pages.")
                if not row["filing_date"].startswith(str(month)):
                    raise ValueError("SEC returned a filing outside the query month.")
                candidates[row["document_id"]] = row
            offset += len(hits)
            if offset >= expected:
                break
            if not hits:
                raise ValueError("SEC pagination ended before its declared total.")
        coverage.append(
            {
                "month": str(month),
                "search_hits": offset,
                "declared_hits": expected,
                "query_complete": complete,
                "incident_coverage_verified": False,
            }
        )
        progress(f"SEC {month}: candidate search audited", flush=True)
    rows = list(candidates.values())
    retrieved_filings = 0
    access_failures = 0
    for row in rows:
        row["filing_retrieved"] = False
        row["filing_source_hash"] = None
        if not row["item_105"] or not row["ciks"] or access_failures >= 3:
            continue
        document = row["document_id"].split(":", 1)[1]
        if "/" in document or "\\" in document or document in (".", ".."):
            raise ValueError("Unexpected SEC primary-document name.")
        cik = int(row["ciks"][0])
        accession = row["accession"].replace("-", "")
        name = f"sec/filings/{row['accession']}-{document}"
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
        try:
            _, record = store.fetch(name, url, headers={"User-Agent": user_agent}, interval=0.5)
            row["filing_retrieved"] = True
            row["filing_source_hash"] = record["sha256"]
            store.describe(
                name,
                source_date=row["filing_date"],
                publication_date=row["filing_date"],
                source_version=row["accession"],
            )
            retrieved_filings += 1
        except DownloadError:
            access_failures += 1
    # Stop repeated blocked requests. Search coverage never substitutes for filing review.
    # Keep multi-registrant filing membership; accession is not a unique incident ID.
    pd.DataFrame(rows).to_parquet(output / "sec_candidates.parquet", index=False)
    pd.DataFrame(coverage).to_csv(output / "sec_search_coverage.csv", index=False)
    return {
        "status": "candidate_search_only",
        "query": '"material cybersecurity"',
        "documents": len(rows),
        "accessions": len({r["accession"] for r in rows}),
        "item_105_documents": sum(r["item_105"] for r in rows),
        "amendment_documents": sum(r["amendment"] for r in rows),
        "item_105_filings_retrieved": retrieved_filings,
        "filing_access_failures": access_failures,
        "months": coverage,
        "incident_counts_eligible": False,
        "reason": (
            "search_recall_not_established; "
            "filing_review_incident_linkage_population_sector_mapping_pending"
        ),
    }
