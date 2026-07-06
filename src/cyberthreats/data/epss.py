"""Exploit Prediction Scoring System (EPSS) data client.

Downloads EPSS daily score files from cyentia.com, caches them locally,
and aggregates per-topic mean EPSS scores into a (K, T) monthly panel.
"""

from __future__ import annotations

import gzip
import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_EPSS_BASE_URL = "https://epss.cyentia.com/epss_scores-{date}.csv.gz"
_REQUEST_TIMEOUT = 60


def fetch_epss(date: str, cache_dir: str) -> pd.DataFrame:
    """Download EPSS scores for a single *date* (``YYYY-MM-DD`` format).

    Parameters
    ----------
    date:
        Date string in ``YYYY-MM-DD`` format.
    cache_dir:
        Directory used to cache downloaded CSV files.

    Returns
    -------
    pd.DataFrame
        Columns: cve_id, epss_score, percentile.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = Path(cache_dir) / f"epss_{date}.parquet"

    if cache_file.exists():
        logger.debug("EPSS cache hit: %s", cache_file)
        return pd.read_parquet(cache_file)

    url = _EPSS_BASE_URL.format(date=date)
    logger.info("Downloading EPSS scores from %s", url)

    try:
        response = requests.get(url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        logger.warning("EPSS HTTP error for date %s: %s", date, exc)
        return pd.DataFrame(columns=["cve_id", "epss_score", "percentile"])
    except requests.exceptions.RequestException as exc:
        logger.warning("EPSS request failed for date %s: %s", date, exc)
        return pd.DataFrame(columns=["cve_id", "epss_score", "percentile"])

    try:
        with gzip.open(io.BytesIO(response.content), "rt") as gz:
            raw = gz.read()
    except OSError as exc:
        logger.warning("EPSS gzip decompression failed for date %s: %s", date, exc)
        return pd.DataFrame(columns=["cve_id", "epss_score", "percentile"])

    # The CSV has a model-version comment line starting with '#' before the header
    lines = [line for line in raw.splitlines() if not line.startswith("#")]
    csv_text = "\n".join(lines)

    df = pd.read_csv(
        io.StringIO(csv_text),
        dtype={"cve": str, "epss": float, "percentile": float},
    )
    df.rename(
        columns={"cve": "cve_id", "epss": "epss_score"},
        inplace=True,
    )
    df = df[["cve_id", "epss_score", "percentile"]].copy()
    df.drop_duplicates(subset=["cve_id"], inplace=True)
    df.to_parquet(cache_file, index=False)
    logger.info("Cached %d EPSS records for %s", len(df), date)
    return df


def fetch_epss_range(start_date: str, end_date: str, cache_dir: str) -> pd.DataFrame:
    """Fetch EPSS scores for every day in [*start_date*, *end_date*].

    Parameters
    ----------
    start_date:
        ISO-8601 date string, e.g. ``"2022-01-01"``.
    end_date:
        ISO-8601 date string, e.g. ``"2022-12-31"``.
    cache_dir:
        Directory used to cache per-day CSV files.

    Returns
    -------
    pd.DataFrame
        Columns: cve_id, epss_score, percentile, date (datetime.date).
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    frames: list[pd.DataFrame] = []

    for ts in date_range:
        date_str = ts.strftime("%Y-%m-%d")
        day_df = fetch_epss(date_str, cache_dir)
        if day_df.empty:
            continue
        day_df = day_df.copy()
        day_df["date"] = ts.date()
        frames.append(day_df)

    if not frames:
        return pd.DataFrame(columns=["cve_id", "epss_score", "percentile", "date"])

    combined = pd.concat(frames, ignore_index=True)
    return combined


def aggregate_monthly(
    epss_df: pd.DataFrame,
    topic_assignments: pd.Series,
) -> np.ndarray:
    """Compute per-topic mean EPSS score for each calendar month.

    Parameters
    ----------
    epss_df:
        DataFrame with at least columns ``cve_id``, ``epss_score``, and
        ``date`` (date or datetime).  Typically the output of
        :func:`fetch_epss_range`.
    topic_assignments:
        Series indexed by ``cve_id`` with integer topic labels ``0 .. K-1``.

    Returns
    -------
    E_kt : np.ndarray, shape (K, T)
        Mean EPSS score per topic per month.  NaN where no CVEs exist.
    """
    if epss_df.empty or topic_assignments.empty:
        K = int(topic_assignments.max()) + 1 if not topic_assignments.empty else 0
        return np.full((K, 0), np.nan)

    work = epss_df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["month"] = work["date"].dt.to_period("M")

    # Map topic to each CVE
    work = work.join(topic_assignments.rename("topic"), on="cve_id", how="inner")
    work.dropna(subset=["topic"], inplace=True)
    work["topic"] = work["topic"].astype(int)

    K = int(topic_assignments.max()) + 1
    all_months = pd.period_range(
        start=work["month"].min(), end=work["month"].max(), freq="M"
    )
    T = len(all_months)
    month_index = {m: i for i, m in enumerate(all_months)}

    E_kt = np.full((K, T), np.nan, dtype=np.float64)

    grouped = work.groupby(["topic", "month"])["epss_score"].mean()
    for (k, m), mean_score in grouped.items():
        t = month_index.get(m)
        if t is not None:
            E_kt[int(k), t] = mean_score

    return E_kt
