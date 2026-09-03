"CISA Known Exploited Vulnerabilities (KEV) catalog client."

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
_REQUEST_TIMEOUT = 60
_CACHE_FILENAME = "cisa_kev.json"


def fetch_kev(cache_dir: str) -> pd.DataFrame:
    "Download the CISA KEV catalog, cache locally, and return as a DataFrame."
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = Path(cache_dir) / _CACHE_FILENAME

    if cache_file.exists():
        logger.info("CISA KEV cache hit: %s", cache_file)
        with cache_file.open() as fh:
            payload = json.load(fh)
    else:
        logger.info("Downloading CISA KEV from %s", _KEV_URL)
        try:
            response = requests.get(_KEV_URL, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to download CISA KEV: %s", exc)
            return _empty_kev_df()

        payload = response.json()
        with cache_file.open("w") as fh:
            json.dump(payload, fh)
        logger.info("Cached CISA KEV to %s", cache_file)

    vulnerabilities = payload.get("vulnerabilities", [])
    records = []
    for item in vulnerabilities:
        records.append(
            {
                "cve_id": item.get("cveID", ""),
                "date_added": item.get("dateAdded", ""),
                "vendor_project": item.get("vendorProject", ""),
                "product": item.get("product", ""),
                "vulnerability_name": item.get("vulnerabilityName", ""),
                "short_description": item.get("shortDescription", ""),
                "required_action": item.get("requiredAction", ""),
                "due_date": item.get("dueDate", ""),
                "known_ransomware_campaign_use": item.get(
                    "knownRansomwareCampaignUse", ""
                ),
            }
        )

    if not records:
        return _empty_kev_df()

    df = pd.DataFrame(records)
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")
    df.drop_duplicates(subset=["cve_id"], inplace=True)
    logger.info("Loaded %d KEV entries", len(df))
    return df


def _empty_kev_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cve_id",
            "date_added",
            "vendor_project",
            "product",
            "vulnerability_name",
            "short_description",
            "required_action",
            "due_date",
            "known_ransomware_campaign_use",
        ]
    )


def merge_with_topics(
    kev_df: pd.DataFrame,
    cve_df: pd.DataFrame,
    topic_assignments: np.ndarray,
    K: int,
) -> np.ndarray:
    "Count KEV-listed CVEs per topic per month."
    if len(cve_df) != len(topic_assignments):
        raise ValueError(
            f"cve_df length {len(cve_df)} != topic_assignments length {len(topic_assignments)}"
        )

    kev_set: set[str] = set(kev_df["cve_id"].dropna().unique())

    work = cve_df[["published_date"]].copy()
    work["topic"] = topic_assignments
    work["month"] = pd.to_datetime(
        work["published_date"], utc=True
    ).dt.tz_localize(None).dt.to_period("M")
    work["in_kev"] = work.index.isin(kev_set).astype(int)

    all_months = pd.period_range(
        start=work["month"].min(), end=work["month"].max(), freq="M"
    )
    T = len(all_months)
    month_index = {m: i for i, m in enumerate(all_months)}

    KEV_kt = np.zeros((K, T), dtype=np.int64)

    grouped = work.groupby(["topic", "month"])["in_kev"].sum()
    for (k, m), count in grouped.items():
        t = month_index.get(m)
        if t is not None:
            KEV_kt[int(k), t] = int(count)

    return KEV_kt
