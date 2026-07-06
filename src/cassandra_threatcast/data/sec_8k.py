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
    "?q=%22cybersecurity+incident%22+%22Item+1.05%22"
    "&dateRange=custom&startdt={start}&enddt={end}&forms=8-K"
)
_EDGAR_COMPANY_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_PAGE_SIZE = 40  # EFTS default page size cap
_REQUEST_TIMEOUT = 30
_INTER_REQUEST_DELAY = 0.11  # SEC fair-access: â‰¤10 req/s


# ---------------------------------------------------------------------------
# SIC â†’ NAICS 2-digit mapping (top 20 SIC codes relevant to cyber incidents)
# ---------------------------------------------------------------------------
_SIC_TO_NAICS2: dict[int, int] = {
    # Technology & Software
    7372: 51,  # Prepackaged Software â†’ Information
    7371: 51,  # Computer Programming, Data Processing
    7374: 51,  # Computer Processing and Data Preparation
    7379: 51,  # Services-Computer Related Services
    7375: 51,  # Computer Rental and Leasing
    # Telecommunications
    4813: 51,  # Telephone Communications
    4899: 51,  # Communications Services, NEC
    4812: 51,  # Radiotelephone Communications
    # Finance
    6020: 52,  # State commercial banks â†’ Finance & Insurance
    6022: 52,  # State commercial banks, Federal Reserve members
    6021: 52,  # National commercial banks
    6159: 52,  # Federal-Sponsored Credit Agencies
    6211: 52,  # Security Brokers, Dealers, Flotation Companies
    6282: 52,  # Investment Advice
    # Healthcare
    8011: 62,  # Offices and Clinics Of Doctors Of Medicine â†’ Health Care
    8049: 62,  # Offices of Other Health Practitioners
    8099: 62,  # Health Services, NEC
    # Retail
    5961: 44,  # Catalog, Mail-Order Houses â†’ Retail Trade
    5734: 44,  # Computer and Computer Software Stores
    # Manufacturing
    3577: 33,  # Computer Peripheral Equipment â†’ Manufacturing
}

_DEFAULT_NAICS2 = 99  # Unknown / unclassified


def _cache_path(cache_dir: str, start_date: str, end_date: str) -> Path:
    key = hashlib.md5(f"sec8k-{start_date}-{end_date}".encode()).hexdigest()
    return Path(cache_dir) / f"sec_8k_{key}.json"


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
        records = _fetch_all_efts_hits(start_date, end_date)
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


def _fetch_all_efts_hits(start_date: str, end_date: str) -> list[dict]:
    """Page through EFTS and return all matching filing records."""
    ua = _user_agent()
    session = requests.Session()
    session.headers.update({"User-Agent": ua})

    base_url = _EFTS_URL.format(start=start_date, end=end_date)
    from_index = 0
    all_records: list[dict] = []

    while True:
        url = f"{base_url}&from={from_index}&size={_PAGE_SIZE}"
        logger.debug("SEC EFTS GET: %s", url)
        try:
            resp = session.get(url, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error("SEC EFTS request failed: %s", exc)
            break

        payload = resp.json()
        hits_obj = payload.get("hits", {})
        hits = hits_obj.get("hits", [])

        for hit in hits:
            src: dict[str, Any] = hit.get("_source", {})
            all_records.append(
                {
                    "cik": str(src.get("entity_id", src.get("cik", ""))).zfill(10),
                    "company_name": src.get("display_names", [""])[0]
                    if src.get("display_names")
                    else src.get("entity_name", ""),
                    "filed_date": src.get("file_date", src.get("period_of_report", "")),
                    "accession_number": src.get("accession_no", ""),
                    "form_type": src.get("form_type", "8-K"),
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

    return all_records


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

    for cik_str in unique_ciks:
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

        naics2 = _SIC_TO_NAICS2.get(sic, _DEFAULT_NAICS2) if sic is not None else _DEFAULT_NAICS2
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

    work = df.copy()
    work["month"] = pd.to_datetime(work["filed_date"]).dt.to_period("M")
    work["sector"] = work["naics_2digit"].map(sector_map)
    work.dropna(subset=["sector"], inplace=True)
    work["sector"] = work["sector"].astype(int)

    S = len(set(sector_map.values()))
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
