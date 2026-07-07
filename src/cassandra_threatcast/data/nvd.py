"""National Vulnerability Database (NVD) CVE 2.0 API client.

Fetches CVE records, caches raw JSON by date range, and aggregates counts
and mean CVSS scores into (K, T) monthly panels.
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

_NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_RESULTS_PER_PAGE = 2000
_INITIAL_BACKOFF = 6.0  # NVD recommends ≥6 s between requests without API key
_MAX_RETRIES = 8
_REQUEST_TIMEOUT = 120  # seconds; NVD can be slow for full 2000-record pages


def _cache_path(cache_dir: str, start_date: str, end_date: str) -> Path:
    key = hashlib.md5(f"nvd-{start_date}-{end_date}".encode()).hexdigest()
    return Path(cache_dir) / f"nvd_{key}.json"


def fetch_cves(start_date: str, end_date: str, cache_dir: str) -> pd.DataFrame:
    """Fetch all CVEs published between *start_date* and *end_date* from NVD.

    Parameters
    ----------
    start_date:
        ISO-8601 date string, e.g. ``"2022-01-01"``.
    end_date:
        ISO-8601 date string, e.g. ``"2022-12-31"``.
    cache_dir:
        Directory used to cache raw API responses.

    Returns
    -------
    pd.DataFrame
        One row per CVE with columns: cve_id, published_date,
        last_modified_date, cvss_base_score, cvss_version, cvss_vector,
        cpe_list, description. CVEs with ``vulnStatus == "Rejected"`` are
        dropped: a rejected CVE is a formal MITRE/CNA determination that the
        ID does not correspond to a real vulnerability (duplicate, withdrawn,
        assigned in error, ...), and its description is boilerplate rejection
        text ("DO NOT USE THIS CANDIDATE NUMBER...") rather than vulnerability
        content -- counting it as an incident or feeding it into topic
        modeling only adds noise.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir, start_date, end_date)

    if cache_file.exists():
        logger.info("NVD cache hit: %s", cache_file)
        with cache_file.open() as fh:
            raw_items: list[dict] = json.load(fh)
    else:
        raw_items = _fetch_all_pages(start_date, end_date, cache_dir)
        with cache_file.open("w") as fh:
            json.dump(raw_items, fh)
        logger.info("Cached %d CVE items to %s", len(raw_items), cache_file)

    records = [_parse_cve_item(item) for item in raw_items]
    df = pd.DataFrame(records)
    if df.empty:
        return df

    n_before = len(df)
    df = df[df["vuln_status"] != "Rejected"].copy()
    n_rejected = n_before - len(df)
    if n_rejected:
        logger.info("Dropped %d rejected CVE record(s) (not real vulnerabilities)", n_rejected)

    df["published_date"] = pd.to_datetime(df["published_date"], utc=True)
    df["last_modified_date"] = pd.to_datetime(df["last_modified_date"], utc=True)
    df["cvss_base_score"] = pd.to_numeric(df["cvss_base_score"], errors="coerce")
    df.set_index("cve_id", inplace=True)
    return df


_NVD_MAX_WINDOW_DAYS = 119  # NVD API 2.0 hard limit is 120 days per request


def _date_windows(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split [start_date, end_date] into ≤119-day chunks required by NVD API."""
    from datetime import date, timedelta
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    windows: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + timedelta(days=_NVD_MAX_WINDOW_DAYS - 1), end)
        windows.append((cursor.isoformat(), window_end.isoformat()))
        cursor = window_end + timedelta(days=1)
    return windows


def _window_cache_path(cache_dir: str, win_start: str, win_end: str) -> Path:
    return Path(cache_dir) / f"nvd_window_{win_start}_{win_end}.json"


def _fetch_all_pages(start_date: str, end_date: str, cache_dir: str) -> list[dict]:
    """Page through the NVD API and return all raw CVE vulnerability items.

    Splits the full date range into ≤119-day windows to stay within the
    NVD API 2.0 limit of 120 days per pubStartDate/pubEndDate request.  Each
    window is cached to ``nvd_window_<start>_<end>.json`` as soon as it
    completes, so a crash or interruption loses at most one window and a
    re-run resumes from where it stopped.
    """
    api_key = os.environ.get("NVD_API_KEY")
    headers: dict[str, str] = {}
    if api_key:
        headers["apiKey"] = api_key
        inter_request_delay = 0.6  # 50 req/30s with key → ~0.6 s
    else:
        inter_request_delay = _INITIAL_BACKOFF

    session = requests.Session()
    session.headers.update(headers)

    windows = _date_windows(start_date, end_date)
    n_windows = len(windows)
    print(f"  NVD: {n_windows} windows x <= {_NVD_MAX_WINDOW_DAYS} days  "
          f"(API key: {'yes' if api_key else 'NO - slow mode, ~6 s/request'})")

    all_items: list[dict] = []

    for i, (win_start, win_end) in enumerate(windows, 1):
        win_cache = _window_cache_path(cache_dir, win_start, win_end)

        # Resume: skip windows already fetched in a prior (possibly crashed) run.
        if win_cache.exists():
            try:
                with win_cache.open() as fh:
                    window_items = json.load(fh)
            except json.JSONDecodeError:
                # The cache file itself was left truncated by an earlier crash
                # mid-write. Treat it like a missing window and refetch rather
                # than propagating the corruption forever.
                print(f"  [{i:>2}/{n_windows}] {win_start} -> {win_end}  "
                      f"(cache corrupt, refetching)", flush=True)
                win_cache.unlink()
            else:
                all_items.extend(window_items)
                print(f"  [{i:>2}/{n_windows}] {win_start} -> {win_end}  "
                      f"({len(window_items)} CVEs, cached)", flush=True)
                continue

        start_index = 0
        window_items: list[dict] = []
        print(f"  [{i:>2}/{n_windows}] {win_start} -> {win_end}", end="", flush=True)

        while True:
            params: dict[str, Any] = {
                "pubStartDate": f"{win_start}T00:00:00.000",
                "pubEndDate": f"{win_end}T23:59:59.999",
                "resultsPerPage": _RESULTS_PER_PAGE,
                "startIndex": start_index,
                # Server-side filter: excludes CVEs formally marked Rejected
                # (not real vulnerabilities) so we never fetch/cache/pay
                # pagination cost for them. fetch_cves() also filters
                # client-side below, since this only helps *new* fetches --
                # window files cached before this parameter was added can
                # still contain Rejected records.
                "noRejected": "",
            }

            data = _get_with_backoff(session, _NVD_BASE_URL, params, inter_request_delay)
            vulnerabilities = data.get("vulnerabilities", [])
            window_items.extend(vulnerabilities)

            total_results = data.get("totalResults", 0)
            start_index += len(vulnerabilities)

            if start_index >= total_results or not vulnerabilities:
                print(f"  ({total_results} CVEs)", flush=True)
                break

            print(f"  {start_index}/{total_results}", end="", flush=True)
            time.sleep(inter_request_delay)

        # Persist this window immediately (crash-safe / resumable). Written to
        # a temp file and atomically renamed so a crash mid-write can never
        # leave a truncated, corrupt cache file at the final path.
        tmp_cache = win_cache.with_suffix(win_cache.suffix + ".tmp")
        with tmp_cache.open("w") as fh:
            json.dump(window_items, fh)
        os.replace(tmp_cache, win_cache)
        all_items.extend(window_items)

    print(f"  NVD total: {len(all_items):,} CVEs fetched across all windows.")
    return all_items


def _get_with_backoff(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    base_delay: float,
) -> dict:
    """GET *url* with exponential backoff on 429 / 5xx responses."""
    backoff = base_delay
    for attempt in range(_MAX_RETRIES):
        try:
            response = session.get(url, params=params, timeout=_REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", backoff))
                logger.warning("NVD rate-limited; sleeping %.1f s", retry_after)
                time.sleep(retry_after)
                backoff *= 2
                continue
            if response.status_code >= 500:
                logger.warning(
                    "NVD server error %d (attempt %d/%d); retrying in %.1f s",
                    response.status_code,
                    attempt + 1,
                    _MAX_RETRIES,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            response.raise_for_status()
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.JSONDecodeError,
        ) as exc:
            # Transient network faults: dropped/incomplete responses, read
            # timeouts, or a truncated body that fails JSON parsing. Retry with
            # exponential backoff. (Genuine 4xx errors fall through
            # raise_for_status above and are not retried.)
            logger.warning(
                "NVD network error (attempt %d/%d): %s",
                attempt + 1, _MAX_RETRIES, exc,
            )
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError(
        f"NVD API request failed after {_MAX_RETRIES} retries for params={params}"
    )


def _parse_cve_item(item: dict) -> dict:
    """Parse one NVD CVE vulnerability item into a flat record dict.

    Parameters
    ----------
    item:
        A single element from the ``vulnerabilities`` list returned by NVD.

    Returns
    -------
    dict
        Keys: cve_id, published_date, last_modified_date, cvss_base_score,
        cvss_version, cvss_vector, cpe_list, description.
    """
    cve = item.get("cve", {})
    cve_id: str = cve.get("id", "")
    published_date: str = cve.get("published", "")
    last_modified_date: str = cve.get("lastModified", "")
    vuln_status: str = cve.get("vulnStatus", "")

    # --- CVSS score: prefer CVSSv3.1 > CVSSv3.0 > CVSSv2 ---
    cvss_base_score: float | None = None
    cvss_version: str | None = None
    cvss_vector: str | None = None

    metrics = cve.get("metrics", {})
    for metric_key, version_label in [
        ("cvssMetricV31", "3.1"),
        ("cvssMetricV30", "3.0"),
        ("cvssMetricV2", "2.0"),
    ]:
        entries = metrics.get(metric_key, [])
        if entries:
            primary = next(
                (e for e in entries if e.get("type") == "Primary"), entries[0]
            )
            cvss_data = primary.get("cvssData", {})
            cvss_base_score = cvss_data.get("baseScore")
            cvss_vector = cvss_data.get("vectorString")
            cvss_version = version_label
            break

    # --- CPE list ---
    cpe_list: list[str] = []
    for config in cve.get("configurations", []):
        for node in config.get("nodes", []):
            for cpe_match in node.get("cpeMatch", []):
                criteria = cpe_match.get("criteria", "")
                if criteria:
                    cpe_list.append(criteria)

    # --- Description (English preferred) ---
    description = ""
    for desc in cve.get("descriptions", []):
        if desc.get("lang") == "en":
            description = desc.get("value", "")
            break
    if not description and cve.get("descriptions"):
        description = cve["descriptions"][0].get("value", "")

    return {
        "cve_id": cve_id,
        "published_date": published_date,
        "last_modified_date": last_modified_date,
        "cvss_base_score": cvss_base_score,
        "cvss_version": cvss_version,
        "cvss_vector": cvss_vector,
        "cpe_list": cpe_list,
        "description": description,
        "vuln_status": vuln_status,
    }


def aggregate_monthly(
    df: pd.DataFrame,
    topic_assignments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate CVE counts and mean CVSS scores into (K, T) monthly panels.

    Parameters
    ----------
    df:
        DataFrame with cve_id as index and at least columns ``published_date``
        (datetime, tz-aware) and ``cvss_base_score`` (float).  Length == N.
    topic_assignments:
        Integer array of length N with values in ``0 .. K-1``, one per row of *df*.

    Returns
    -------
    N_kt : np.ndarray, shape (K, T)
        Monthly CVE counts per topic.
    B_kt : np.ndarray, shape (K, T)
        Monthly mean CVSS base score per topic (NaN where count is 0).
    """
    if len(df) != len(topic_assignments):
        raise ValueError(
            f"df length {len(df)} != topic_assignments length {len(topic_assignments)}"
        )

    K = int(topic_assignments.max()) + 1
    work = df[["published_date", "cvss_base_score"]].copy()
    work["topic"] = topic_assignments
    work["month"] = pd.to_datetime(
        work["published_date"], utc=True
    ).dt.tz_localize(None).dt.to_period("M")

    all_months = pd.period_range(
        start=work["month"].min(), end=work["month"].max(), freq="M"
    )
    T = len(all_months)
    month_index = {m: i for i, m in enumerate(all_months)}

    N_kt = np.zeros((K, T), dtype=np.int64)
    B_kt = np.full((K, T), np.nan, dtype=np.float64)

    grouped = work.groupby(["topic", "month"])
    for (k, m), grp in grouped:
        t = month_index.get(m)
        if t is None:
            continue
        N_kt[int(k), t] = len(grp)
        scores = grp["cvss_base_score"].dropna()
        if len(scores) > 0:
            B_kt[int(k), t] = scores.mean()

    return N_kt, B_kt
