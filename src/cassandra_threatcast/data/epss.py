"Exploit Prediction Scoring System (EPSS) data client."

from __future__ import annotations

import gzip
import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_EPSS_BASE_URL = "https://epss.empiricalsecurity.com/epss_scores-{date}.csv.gz"
_EPSS_START = pd.Timestamp("2021-04-14")  # first date EPSS daily scores exist
_REQUEST_TIMEOUT = 60


def fetch_epss(date: str, cache_dir: str) -> pd.DataFrame:
    "Download EPSS scores for a single *date* (``YYYY-MM-DD`` format)."
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = Path(cache_dir) / f"epss_{date}.csv"

    if cache_file.exists():
        logger.debug("EPSS cache hit: %s", cache_file)
        return pd.read_csv(cache_file, dtype={"cve_id": str})

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

    # The CSV has a model-version comment line starting with '#' before...
    lines = [line for line in raw.splitlines() if not line.startswith("#")]
    csv_text = "\n".join(lines)

    df = pd.read_csv(io.StringIO(csv_text))
    df.rename(
        columns={"cve": "cve_id", "epss": "epss_score"},
        inplace=True,
    )
    # EPSS v1 files (2021-04 .. early 2022) have no "percentile" column;...
    if "percentile" not in df.columns:
        df["percentile"] = np.nan
    df["cve_id"] = df["cve_id"].astype(str)
    df["epss_score"] = pd.to_numeric(df["epss_score"], errors="coerce")
    df["percentile"] = pd.to_numeric(df["percentile"], errors="coerce")
    df = df[["cve_id", "epss_score", "percentile"]].copy()
    df.drop_duplicates(subset=["cve_id"], inplace=True)
    df.to_csv(cache_file, index=False)
    logger.info("Cached %d EPSS records for %s", len(df), date)
    return df


def fetch_epss_range(start_date: str, end_date: str, cache_dir: str) -> pd.DataFrame:
    "Fetch EPSS scores at one representative snapshot per month in the range."
    empty = pd.DataFrame(columns=["cve_id", "epss_score", "percentile", "date"])
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if end < _EPSS_START:
        logger.warning(
            "EPSS scores start %s; requested range [%s, %s] is entirely earlier — "
            "no EPSS data available.",
            _EPSS_START.date(), start.date(), end.date(),
        )
        return empty

    eff_start = max(start, _EPSS_START)
    if eff_start > start:
        logger.info(
            "EPSS scores start %s; clamping EPSS fetch start from %s to %s.",
            _EPSS_START.date(), start.date(), eff_start.date(),
        )

    frames: list[pd.DataFrame] = []
    for period in pd.period_range(start=eff_start, end=end, freq="M"):
        # Mid-month snapshot (15th), clamped into [eff_start, end].
        target = period.to_timestamp() + pd.Timedelta(days=14)
        target = min(max(target, eff_start), end)
        date_str = target.strftime("%Y-%m-%d")

        day_df = fetch_epss(date_str, cache_dir)
        if day_df.empty:
            continue
        day_df = day_df.copy()
        day_df["date"] = target.date()
        frames.append(day_df)

    if not frames:
        return empty

    return pd.concat(frames, ignore_index=True)


def aggregate_monthly(
    epss_df: pd.DataFrame,
    topic_assignments: pd.Series,
) -> np.ndarray:
    "Compute per-topic mean EPSS score for each calendar month."
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
