"""SEC EDGAR 8-K cybersecurity incident filing client.

Queries EDGAR Full-Text Search (EFTS) for 8-K filings disclosing cybersecurity
incidents under Item 1.05, enriches filings with NAICS sector codes, and
aggregates to a (S, T) monthly panel of incident disclosures.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_EFTS_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%22Item+1.05%22&forms=8-K&startdt={start}&enddt={end}"
)
_EDGAR_COMPANY_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_PAGE_SIZE = 40  # EFTS default page size cap
_REQUEST_TIMEOUT = 30
_INTER_REQUEST_DELAY = 0.11  # SEC fair-access: â‰¤10 req/s


# ---------------------------------------------------------------------------
# SIC -> NAICS 2-digit mapping by SIC division ranges. SIC codes have a fixed
# block structure, so range mapping covers essentially every filer (vastly more
# than a handful of hardcoded codes).
# ---------------------------------------------------------------------------
def _sic_to_naics2(sic: int | None) -> int:
    """Map a 4-digit SIC code to a 2-digit NAICS sector via SIC divisions."""
    if sic is None:
        return _DEFAULT_NAICS2
    if 100 <= sic <= 999:
        return 11            # Agriculture, forestry, fishing
    if 1000 <= sic <= 1499:
        return 21            # Mining
    if 1500 <= sic <= 1799:
        return 23            # Construction
    if 2000 <= sic <= 3999:
        return 31            # Manufacturing
    if 4000 <= sic <= 4799:
        return 48            # Transportation
    if 4800 <= sic <= 4899:
        return 51            # Communications -> Information
    if 4900 <= sic <= 4999:
        return 22            # Electric, gas & sanitary -> Utilities
    if 5000 <= sic <= 5199:
        return 42            # Wholesale Trade
    if 5200 <= sic <= 5999:
        return 44            # Retail Trade
    if 6000 <= sic <= 6799:
        return 52            # Finance, insurance & real estate
    if 7000 <= sic <= 8999:
        return 54            # Services -> Professional
    if 9100 <= sic <= 9999:
        return 92            # Public administration -> Government
    return _DEFAULT_NAICS2

_DEFAULT_NAICS2 = 99  # Unknown / unclassified


# 2-digit NAICS sector -> model sector index (0..10), matching the 11-sector
# BEA aggregation used elsewhere. Unmapped codes (e.g. 99 unclassified) are
# dropped from the incident panel rather than forced into a sector.
NAICS2_TO_SECTOR: dict[int, int] = {
    11: 0,                      # Agriculture
    21: 1,                      # Mining
    22: 2,                      # Utilities
    23: 3,                      # Construction
    31: 4, 32: 4, 33: 4,        # Manufacturing
    42: 5,                      # Wholesale Trade
    44: 6, 45: 6,               # Retail Trade
    48: 7, 49: 7,               # Transportation & Warehousing
    52: 8, 53: 8,               # Finance, Insurance & Real Estate
    51: 9, 54: 9, 55: 9, 56: 9, # Information + Professional/Admin services
    61: 9, 62: 9, 71: 9, 72: 9, 81: 9,  # Education, Health, Leisure, Other
    92: 10,                     # Government
}


def _cache_path(cache_dir: str, start_date: str, end_date: str) -> Path:
    # Cache key includes a schema version: bump it whenever the parsed record
    # schema changes, so stale caches from an older parser are not reused.
    key = hashlib.md5(f"sec8k-v2-{start_date}-{end_date}".encode()).hexdigest()
    return Path(cache_dir) / f"sec_8k_v2_{key}.json"


def _user_agent() -> str:
    agent = os.environ.get("SEC_USER_AGENT")
    if not agent:
        logger.warning(
            "SEC_USER_AGENT environment variable not set. "
            "SEC fair-access policy requires a valid User-Agent. "
            'Set it to e.g. "MyOrg myemail@example.com".'
        )
        return "cassandra_threatcast-research contact@example.com"
    return agent


def fetch_8k_cyber(start_date: str, end_date: str, cache_dir: str) -> pd.DataFrame:
    """Query SEC EDGAR EFTS for 8-K cybersecurity incident filings.

    Fetches all 8-K filings that mention "cybersecurity incident" and
    "Item 1.05" within [*start_date*, *end_date*].  Results are cached by
    date range.

    Parameters
    ----------
    start_date:
        ISO-8601 date string, e.g. ``"2023-01-01"``.
    end_date:
        ISO-8601 date string, e.g. ``"2023-12-31"``.
    cache_dir:
        Directory used to cache raw API responses.

    Returns
    -------
    pd.DataFrame
        Columns: cik, company_name, filed_date, accession_number, form_type.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir, start_date, end_date)

    if cache_file.exists():
        logger.info("SEC 8-K cache hit: %s", cache_file)
        with cache_file.open() as fh:
            records: list[dict] = json.load(fh)
    else:
        records, complete = _fetch_all_efts_hits(start_date, end_date)
        if not complete:
            # A partial/failed fetch must never be cached: caching it would
            # silently and permanently record "zero filings" for this date
            # range, even after the underlying API issue is resolved.
            raise RuntimeError(
                f"SEC EFTS fetch for {start_date}..{end_date} did not complete "
                f"({len(records)} partial record(s)) -- not caching. Retry once "
                "the SEC EDGAR full-text search API is healthy."
            )
        with cache_file.open("w") as fh:
            json.dump(records, fh)
        logger.info("Cached %d SEC 8-K records to %s", len(records), cache_file)

    if not records:
        return pd.DataFrame(
            columns=["cik", "company_name", "filed_date", "accession_number", "form_type"]
        )

    df = pd.DataFrame(records)
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    return df


def _fetch_all_efts_hits(start_date: str, end_date: str) -> tuple[list[dict], bool]:
    """Page through EFTS and return (records, complete).

    complete is False if a request failed partway through pagination -- the
    caller must not cache that as a genuine (possibly zero-result) fetch, or
    a transient error permanently poisons the cache for that date range.
    """
    ua = _user_agent()
    session = requests.Session()
    session.headers.update({"User-Agent": ua})

    base_url = _EFTS_URL.format(start=start_date, end=end_date)
    from_index = 0
    all_records: list[dict] = []
    complete = True

    while True:
        url = f"{base_url}&from={from_index}&size={_PAGE_SIZE}"
        logger.debug("SEC EFTS GET: %s", url)
        try:
            resp = session.get(url, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error("SEC EFTS request failed: %s", exc)
            complete = False
            break

        payload = resp.json()
        hits_obj = payload.get("hits", {})
        hits = hits_obj.get("hits", [])

        for hit in hits:
            src: dict[str, Any] = hit.get("_source", {})
            ciks = src.get("ciks") or [""]
            sics = src.get("sics") or []
            names = src.get("display_names") or [""]
            all_records.append(
                {
                    "cik": str(ciks[0]).zfill(10),
                    "company_name": names[0],
                    "filed_date": src.get("file_date", ""),
                    "accession_number": src.get("adsh", ""),
                    "form_type": src.get("form", "8-K"),
                    # SIC is returned inline by EFTS — capture it so enrich_with_naics
                    # can skip a per-company API call.
                    "sic": int(sics[0]) if sics and str(sics[0]).isdigit() else None,
                }
            )

        total = hits_obj.get("total", {}).get("value", 0) if isinstance(
            hits_obj.get("total"), dict
        ) else hits_obj.get("total", 0)
        from_index += len(hits)
        logger.debug("SEC EFTS: fetched %d / %d", from_index, total)

        if from_index >= total or not hits:
            break

        time.sleep(_INTER_REQUEST_DELAY)

    return all_records, complete


def enrich_with_naics(df: pd.DataFrame, cache_dir: str) -> pd.DataFrame:
    """Add a ``naics_2digit`` column derived from each filer's SIC code.

    Fetches SIC codes from SEC EDGAR company JSON submissions endpoint for
    each unique CIK in *df* and maps them to 2-digit NAICS sectors using a
    hardcoded lookup table.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`fetch_8k_cyber`.
    cache_dir:
        Directory used to cache per-CIK company JSON files.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an extra ``naics_2digit`` column (int).
    """
    if df.empty:
        df = df.copy()
        df["naics_2digit"] = pd.Series(dtype=int)
        return df

    ua = _user_agent()
    session = requests.Session()
    session.headers.update({"User-Agent": ua})
    sic_cache_dir = Path(cache_dir) / "sec_company"
    sic_cache_dir.mkdir(parents=True, exist_ok=True)

    cik_to_naics: dict[str, int] = {}
    unique_ciks = df["cik"].unique()

    # SIC codes returned inline by EFTS avoid a per-company API round-trip.
    inline_sic: dict[str, int] = {}
    if "sic" in df.columns:
        for _, row in df.iterrows():
            s = row.get("sic")
            if s is not None and not (isinstance(s, float) and np.isnan(s)):
                inline_sic[str(row["cik"])] = int(s)

    for cik_str in unique_ciks:
        if cik_str in inline_sic:
            cik_to_naics[cik_str] = _sic_to_naics2(inline_sic[cik_str])
            continue

        try:
            cik_int = int(cik_str)
        except (ValueError, TypeError):
            cik_to_naics[cik_str] = _DEFAULT_NAICS2
            continue

        cik_cache = sic_cache_dir / f"cik_{cik_int:010d}.json"
        if cik_cache.exists():
            with cik_cache.open() as fh:
                company_data = json.load(fh)
        else:
            url = _EDGAR_COMPANY_URL.format(cik=cik_int)
            try:
                resp = session.get(url, timeout=_REQUEST_TIMEOUT)
                resp.raise_for_status()
                company_data = resp.json()
                with cik_cache.open("w") as fh:
                    json.dump(company_data, fh)
            except requests.exceptions.RequestException as exc:
                logger.warning("Could not fetch company data for CIK %s: %s", cik_str, exc)
                cik_to_naics[cik_str] = _DEFAULT_NAICS2
                time.sleep(_INTER_REQUEST_DELAY)
                continue
            time.sleep(_INTER_REQUEST_DELAY)

        sic_raw = company_data.get("sic")
        try:
            sic = int(sic_raw) if sic_raw is not None else None
        except (ValueError, TypeError):
            sic = None

        naics2 = _sic_to_naics2(sic)
        cik_to_naics[cik_str] = naics2

    result = df.copy()
    result["naics_2digit"] = result["cik"].map(cik_to_naics).fillna(_DEFAULT_NAICS2).astype(int)
    return result


def aggregate_monthly(df: pd.DataFrame, sector_map: dict) -> np.ndarray:
    """Aggregate 8-K incident counts into a (S, T) monthly panel.

    Parameters
    ----------
    df:
        DataFrame with at least columns ``filed_date`` (datetime) and
        ``naics_2digit`` (int).  Typically output of :func:`enrich_with_naics`.
    sector_map:
        Mapping ``{naics_2digit: sector_index}`` where ``sector_index`` is an
        integer in ``0 .. S-1``.

    Returns
    -------
    D_st : np.ndarray, shape (S, T), dtype int64
        Monthly 8-K cyber incident disclosure counts per sector.
    """
    if df.empty:
        S = len(set(sector_map.values())) if sector_map else 0
        return np.zeros((S, 0), dtype=np.int64)

    S = len(set(sector_map.values()))

    work = df.copy()
    filed = pd.to_datetime(work["filed_date"], errors="coerce", utc=True).dt.tz_localize(None)
    work["month"] = filed.dt.to_period("M")
    work["sector"] = work["naics_2digit"].map(sector_map)
    # Drop rows with an unparseable filing date or an unmapped sector.
    work = work.dropna(subset=["month", "sector"])
    if work.empty:
        return np.zeros((S, 0), dtype=np.int64)
    work["sector"] = work["sector"].astype(int)

    all_months = pd.period_range(start=work["month"].min(), end=work["month"].max(), freq="M")
    T = len(all_months)
    month_index = {m: i for i, m in enumerate(all_months)}

    D_st = np.zeros((S, T), dtype=np.int64)

    grouped = work.groupby(["sector", "month"]).size()
    for (s, m), count in grouped.items():
        t = month_index.get(m)
        if t is not None:
            D_st[int(s), t] = int(count)

    return D_st
