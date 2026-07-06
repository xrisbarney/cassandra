"""Data pipeline orchestrator.

Builds the full threat-panel dictionary by calling NVD, EPSS, CISA KEV, and
SEC 8-K data modules, then saving / loading the result to / from disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_panel(
    start: str,
    end: str,
    cache_dir: str,
    topic_assignments: np.ndarray,
    sector_map: dict,
    K: int,
    S: int,
) -> dict:
    """Orchestrate all data sources and return a unified threat panel.

    Calls each data module in sequence.  If any individual source fails the
    corresponding array is filled with NaN / zeros and a warning is logged,
    so a partial panel is always returned.

    Parameters
    ----------
    start:
        Panel start date, ISO-8601 string, e.g. ``"2017-01-01"``.
    end:
        Panel end date, ISO-8601 string, e.g. ``"2023-12-31"``.
    cache_dir:
        Root cache directory; sub-modules will create sub-directories.
    topic_assignments:
        Integer array of length ``n_cves`` with values in ``0 .. K-1``.
        Passed to aggregation functions.  If not yet computed (e.g., first
        run before topic modelling), pass ``np.zeros(0, dtype=int)``.
    sector_map:
        ``{naics_2digit: sector_index}`` mapping for SEC 8-K aggregation.
    K:
        Number of threat topics.
    S:
        Number of economic sectors.

    Returns
    -------
    dict with keys:
        - ``N``  : np.ndarray (K, T) â€“ monthly CVE counts per topic
        - ``B``  : np.ndarray (K, T) â€“ monthly mean CVSS score per topic
        - ``E``  : np.ndarray (K, T) â€“ monthly mean EPSS score per topic
        - ``KEV``: np.ndarray (K, T) â€“ monthly KEV exploitation count per topic
        - ``D``  : np.ndarray (S, T) â€“ monthly 8-K disclosures per sector
        - ``dates``: list[pd.Period] â€“ monthly period labels (length T)
        - ``metadata``: dict â€“ source-level record counts and status flags
    """
    all_months = pd.period_range(start=start, end=end, freq="M")
    T = len(all_months)
    dates = list(all_months)
    metadata: dict = {"start": start, "end": end, "T": T, "K": K, "S": S}

    # ------------------------------------------------------------------
    # 1. NVD: CVE counts (N_kt) and mean CVSS scores (B_kt)
    # ------------------------------------------------------------------
    N_kt = np.zeros((K, T), dtype=np.int64)
    B_kt = np.full((K, T), np.nan, dtype=np.float64)
    cve_df: pd.DataFrame = pd.DataFrame()

    try:
        from cassandra_threatcast.data.nvd import fetch_cves, aggregate_monthly as nvd_agg

        cve_df = fetch_cves(start, end, cache_dir)
        metadata["nvd_cve_count"] = len(cve_df)
        metadata["nvd_status"] = "ok"

        if len(cve_df) > 0 and len(topic_assignments) == len(cve_df):
            N_raw, B_raw = nvd_agg(cve_df, topic_assignments)
            # Align to the panel's month range
            N_kt, B_kt = _align_kt(N_raw, B_raw, cve_df, all_months, K, T)
        elif len(topic_assignments) == 0:
            logger.warning("topic_assignments is empty; skipping NVD aggregation.")
            metadata["nvd_status"] = "no_topics"
        else:
            logger.warning(
                "topic_assignments length %d != cve_df length %d; skipping NVD aggregation.",
                len(topic_assignments),
                len(cve_df),
            )
            metadata["nvd_status"] = "length_mismatch"
    except Exception as exc:
        logger.warning("NVD module failed: %s", exc, exc_info=True)
        metadata["nvd_status"] = f"error: {exc}"
        metadata["nvd_cve_count"] = 0

    # ------------------------------------------------------------------
    # 2. EPSS: mean exploitation probability per topic (E_kt)
    # ------------------------------------------------------------------
    E_kt = np.full((K, T), np.nan, dtype=np.float64)

    try:
        from cassandra_threatcast.data.epss import fetch_epss_range, aggregate_monthly as epss_agg

        epss_df = fetch_epss_range(start, end, cache_dir)
        metadata["epss_record_count"] = len(epss_df)
        metadata["epss_status"] = "ok"

        if not epss_df.empty and len(topic_assignments) == len(cve_df) > 0:
            topic_series = pd.Series(
                topic_assignments, index=cve_df.index, name="topic"
            )
            E_raw = epss_agg(epss_df, topic_series)
            E_kt = _align_k_array(E_raw, epss_df, all_months, K, T, col="date")
        elif epss_df.empty:
            metadata["epss_status"] = "empty"
    except Exception as exc:
        logger.warning("EPSS module failed: %s", exc, exc_info=True)
        metadata["epss_status"] = f"error: {exc}"
        metadata["epss_record_count"] = 0

    # ------------------------------------------------------------------
    # 3. CISA KEV: exploitation flag counts per topic (KEV_kt)
    # ------------------------------------------------------------------
    KEV_kt = np.zeros((K, T), dtype=np.int64)

    try:
        from cassandra_threatcast.data.cisa_kev import fetch_kev, merge_with_topics

        kev_df = fetch_kev(cache_dir)
        metadata["kev_entry_count"] = len(kev_df)
        metadata["kev_status"] = "ok"

        if (
            not kev_df.empty
            and not cve_df.empty
            and len(topic_assignments) == len(cve_df)
        ):
            KEV_raw = merge_with_topics(kev_df, cve_df, topic_assignments, K)
            KEV_kt = _align_kt_int(KEV_raw, cve_df, all_months, K, T)
    except Exception as exc:
        logger.warning("CISA KEV module failed: %s", exc, exc_info=True)
        metadata["kev_status"] = f"error: {exc}"
        metadata["kev_entry_count"] = 0

    # ------------------------------------------------------------------
    # 4. SEC 8-K: incident disclosures per sector (D_st)
    # ------------------------------------------------------------------
    D_st = np.zeros((S, T), dtype=np.int64)

    try:
        from cassandra_threatcast.data.sec_8k import (
            fetch_8k_cyber,
            enrich_with_naics,
            aggregate_monthly as sec_agg,
        )

        sec_df = fetch_8k_cyber(start, end, cache_dir)
        metadata["sec_8k_filing_count"] = len(sec_df)
        metadata["sec_8k_status"] = "ok"

        if not sec_df.empty:
            sec_df = enrich_with_naics(sec_df, cache_dir)
            D_raw = sec_agg(sec_df, sector_map)
            D_st = _align_st(D_raw, sec_df, all_months, S, T)
    except Exception as exc:
        logger.warning("SEC 8-K module failed: %s", exc, exc_info=True)
        metadata["sec_8k_status"] = f"error: {exc}"
        metadata["sec_8k_filing_count"] = 0

    return {
        "N": N_kt,
        "B": B_kt,
        "E": E_kt,
        "KEV": KEV_kt,
        "D": D_st,
        "dates": dates,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Alignment helpers â€” map module-internal time axes to the panel's month range
# ---------------------------------------------------------------------------

def _align_kt(
    N_raw: np.ndarray,
    B_raw: np.ndarray,
    cve_df: pd.DataFrame,
    all_months: pd.PeriodIndex,
    K: int,
    T: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align (K, T_src) arrays produced by NVD aggregation to the panel's (K, T)."""
    src_months = pd.period_range(
        start=cve_df["published_date"].min().to_period("M"),
        end=cve_df["published_date"].max().to_period("M"),
        freq="M",
    )
    N_out = np.zeros((K, T), dtype=np.int64)
    B_out = np.full((K, T), np.nan, dtype=np.float64)

    for t_src, m in enumerate(src_months):
        matches = np.where(all_months == m)[0]
        if matches.size == 0:
            continue
        t_dst = int(matches[0])
        if t_src < N_raw.shape[1]:
            N_out[:, t_dst] = N_raw[:, t_src]
            B_out[:, t_dst] = B_raw[:, t_src]

    return N_out, B_out


def _align_kt_int(
    arr: np.ndarray,
    cve_df: pd.DataFrame,
    all_months: pd.PeriodIndex,
    K: int,
    T: int,
) -> np.ndarray:
    """Align a single (K, T_src) integer array to the panel's (K, T)."""
    src_months = pd.period_range(
        start=cve_df["published_date"].min().to_period("M"),
        end=cve_df["published_date"].max().to_period("M"),
        freq="M",
    )
    out = np.zeros((K, T), dtype=np.int64)
    for t_src, m in enumerate(src_months):
        matches = np.where(all_months == m)[0]
        if matches.size == 0:
            continue
        t_dst = int(matches[0])
        if t_src < arr.shape[1]:
            out[:, t_dst] = arr[:, t_src]
    return out


def _align_k_array(
    arr: np.ndarray,
    df: pd.DataFrame,
    all_months: pd.PeriodIndex,
    K: int,
    T: int,
    col: str = "date",
) -> np.ndarray:
    """Align a (K, T_src) float array using the date column of *df*."""
    df_dates = pd.to_datetime(df[col])
    src_min = df_dates.min().to_period("M")
    src_max = df_dates.max().to_period("M")
    src_months = pd.period_range(start=src_min, end=src_max, freq="M")

    out = np.full((K, T), np.nan, dtype=np.float64)
    for t_src, m in enumerate(src_months):
        matches = np.where(all_months == m)[0]
        if matches.size == 0:
            continue
        t_dst = int(matches[0])
        if t_src < arr.shape[1]:
            out[:, t_dst] = arr[:, t_src]
    return out


def _align_st(
    arr: np.ndarray,
    sec_df: pd.DataFrame,
    all_months: pd.PeriodIndex,
    S: int,
    T: int,
) -> np.ndarray:
    """Align a (S, T_src) int array using filed_date column of sec_df."""
    filed = pd.to_datetime(sec_df["filed_date"].dropna())
    if filed.empty:
        return np.zeros((S, T), dtype=np.int64)

    src_months = pd.period_range(
        start=filed.min().to_period("M"),
        end=filed.max().to_period("M"),
        freq="M",
    )
    out = np.zeros((S, T), dtype=np.int64)
    for t_src, m in enumerate(src_months):
        matches = np.where(all_months == m)[0]
        if matches.size == 0:
            continue
        t_dst = int(matches[0])
        if t_src < arr.shape[1]:
            out[:, t_dst] = arr[:, t_src]
    return out


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_panel(panel: dict, output_dir: str) -> None:
    """Persist a panel dict to *output_dir*.

    Arrays are saved as ``.npy`` files; the dates list and metadata dict are
    saved as ``panel_meta.json``.

    Parameters
    ----------
    panel:
        Dict returned by :func:`build_panel`.
    output_dir:
        Destination directory (will be created if absent).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    array_keys = ["N", "B", "E", "KEV", "D"]
    for key in array_keys:
        if key in panel and isinstance(panel[key], np.ndarray):
            np.save(str(out / f"{key}.npy"), panel[key])
            logger.info("Saved %s.npy  shape=%s", key, panel[key].shape)

    meta_payload = {
        "dates": [str(p) for p in panel.get("dates", [])],
        "metadata": panel.get("metadata", {}),
    }
    meta_file = out / "panel_meta.json"
    with meta_file.open("w") as fh:
        json.dump(meta_payload, fh, indent=2, default=str)
    logger.info("Saved panel metadata to %s", meta_file)


def load_panel(input_dir: str) -> dict:
    """Load a panel from *input_dir* previously written by :func:`save_panel`.

    Parameters
    ----------
    input_dir:
        Directory containing ``.npy`` files and ``panel_meta.json``.

    Returns
    -------
    dict
        Same structure as returned by :func:`build_panel`.
    """
    src = Path(input_dir)
    if not src.exists():
        raise FileNotFoundError(f"Panel directory not found: {input_dir}")

    panel: dict = {}
    for key in ["N", "B", "E", "KEV", "D"]:
        fpath = src / f"{key}.npy"
        if fpath.exists():
            panel[key] = np.load(str(fpath))
            logger.info("Loaded %s  shape=%s", fpath.name, panel[key].shape)
        else:
            logger.warning("Panel file not found: %s", fpath)

    meta_file = src / "panel_meta.json"
    if meta_file.exists():
        with meta_file.open() as fh:
            meta_payload = json.load(fh)
        panel["dates"] = [pd.Period(s, freq="M") for s in meta_payload.get("dates", [])]
        panel["metadata"] = meta_payload.get("metadata", {})
    else:
        panel["dates"] = []
        panel["metadata"] = {}
        logger.warning("panel_meta.json not found in %s", input_dir)

    return panel
