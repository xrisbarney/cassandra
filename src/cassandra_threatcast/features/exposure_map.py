"""
exposure_map.py
===============
Maps CPE vendor strings to BEA sector indices and builds the M_skt exposure tensor.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

# CPE vendor/product prefix → BEA sector index (0-based, 11 sectors)
CPE_SECTOR_MAP: dict[str, int] = {
    # ---- Information Technology (sector 7) ----
    "microsoft": 7,
    "apple": 7,
    "google": 7,
    "linux": 7,
    "canonical": 7,
    "redhat": 7,
    "debian": 7,
    "ubuntu": 7,
    "apache": 7,
    "nginx": 7,
    "mozilla": 7,
    "openssl": 7,
    "openssh": 7,
    "samba": 7,
    "wordpress": 7,
    "drupal": 7,
    "joomla": 7,
    "php": 7,
    "python": 7,
    "ruby": 7,
    "nodejs": 7,
    "openjdk": 7,
    "oracle_java": 7,
    "vmware": 7,
    "docker": 7,
    "kubernetes": 7,
    # ---- Telecom → Transportation & warehousing (sector 6) ----
    "cisco": 6,
    "juniper": 6,
    "fortinet": 6,
    "paloaltonetworks": 6,
    "checkpoint": 6,
    "f5": 6,
    "netgear": 6,
    "zyxel": 6,
    "dlink": 6,
    "tplink": 6,
    # ---- Finance & insurance (sector 8) ----
    "oracle": 8,
    "sap": 8,
    "ibm": 8,
    "salesforce": 8,
    "servicenow": 8,
    "intuit": 8,
    "finastra": 8,
    # ---- Professional & business services (sector 9) ----
    "adobe": 9,
    "atlassian": 9,
    "citrix": 9,
    "ivanti": 9,
    "progress": 9,
    "zoho": 9,
    "gitlab": 9,
    "github": 9,
    # ---- Manufacturing (sector 4) ----
    "siemens": 4,
    "ge": 4,
    "schneider": 4,
    "rockwellautomation": 4,
    "honeywell": 4,
    "beckhoff": 4,
    # ---- Health care (sector 10) ----
    "philips": 10,
    "medtronic": 10,
    "epic": 10,
    "cerner": 10,
    "changehealth": 10,
    # ---- Utilities (sector 2) ----
    "abb": 2,
    "emerson": 2,
    "yokogawa": 2,
}

# Default sector for unknown vendors
_DEFAULT_SECTOR = 7  # Information technology


def _extract_cpe_vendor(cpe_string: str) -> str:
    """Extract the vendor component from a CPE 2.3 URI or formatted string."""
    # cpe:2.3:a:vendor:product:... or cpe:/a:vendor:product:...
    parts = cpe_string.split(":")
    if len(parts) >= 4:
        return parts[3].lower().replace("-", "").replace("_", "")
    return ""


def _lookup_sector(vendor: str) -> int:
    """Look up sector index for a vendor string (normalised)."""
    vendor_clean = re.sub(r"[^a-z0-9]", "", vendor.lower())
    for prefix, sector in CPE_SECTOR_MAP.items():
        prefix_clean = re.sub(r"[^a-z0-9]", "", prefix.lower())
        if vendor_clean.startswith(prefix_clean) or prefix_clean in vendor_clean:
            return sector
    return _DEFAULT_SECTOR


def build_exposure_map(
    cve_df: pd.DataFrame,
    S: int,
    K: int,
    topic_assignments: np.ndarray,
    dates: list,
) -> np.ndarray:
    """
    Construct M_skt exposure tensor of shape (S, K, T).

    Parameters
    ----------
    cve_df : DataFrame with at least columns 'date' and 'cpe' (string or list of CPE strings).
    S : number of BEA sectors.
    K : number of threat topics.
    topic_assignments : (n_cves,) integer array of topic labels.
    dates : ordered list of T period labels (matched against cve_df['date']).

    Returns
    -------
    M_skt : np.ndarray of shape (S, K, T), column-normalised per time slice.
    """
    T = len(dates)
    date_index = {d: t for t, d in enumerate(dates)}

    M = np.zeros((S, K, T), dtype=np.float64)

    # Pre-resolve the per-row month index and topic as arrays (vectorized), then
    dates_arr = cve_df["date"].map(date_index).to_numpy()   # NaN where out of range
    cpe_arr = cve_df["cpe"].to_numpy()
    topics = np.asarray(topic_assignments)

    vendor_sector_cache: dict[str, int] = {}

    # Accumulate contributions as flat index lists for a single np.add.at call.
    idx_s: list[int] = []
    idx_k: list[int] = []
    idx_t: list[int] = []
    vals: list[float] = []

    n = len(cpe_arr)
    for i in range(n):
        t = dates_arr[i]
        if t != t:  # NaN -> date outside panel range
            continue
        t = int(t)
        k = int(topics[i])
        if k < 0 or k >= K:
            continue

        cpe_raw = cpe_arr[i]
        if isinstance(cpe_raw, str):
            cpe_list = [c.strip() for c in cpe_raw.split(";") if c.strip()]
        elif isinstance(cpe_raw, (list, tuple)):
            cpe_list = cpe_raw
        else:
            cpe_list = []

        sectors_hit: set[int] = set()
        for cpe_str in cpe_list:
            s = vendor_sector_cache.get(cpe_str)
            if s is None:
                vendor = _extract_cpe_vendor(cpe_str)
                s = _lookup_sector(vendor) if vendor else -1
                vendor_sector_cache[cpe_str] = s
            if 0 <= s < S:
                sectors_hit.add(s)

        if not sectors_hit:
            sectors_hit = set(range(S))  # unknown vendor -> spread uniformly

        weight = 1.0 / len(sectors_hit)
        for s in sectors_hit:
            idx_s.append(s); idx_k.append(k); idx_t.append(t); vals.append(weight)

    if vals:
        np.add.at(M, (np.array(idx_s), np.array(idx_k), np.array(idx_t)), np.array(vals))

    # Normalise each (t) slice so columns (over s) sum to 1
    for t in range(T):
        col_sums = M[:, :, t].sum(axis=0, keepdims=True)  # (1, K)
        nonzero = col_sums > 0
        M[:, :, t] = np.where(
            np.broadcast_to(nonzero, M[:, :, t].shape),
            M[:, :, t] / np.where(nonzero, col_sums, 1.0),
            1.0 / S,  # uniform for all-zero columns
        )

    return M


def summarize_exposure(
    M_skt: np.ndarray,
    sector_labels: list[str],
    topic_labels: list[str],
) -> pd.DataFrame:
    """
    Return tidy DataFrame with columns: sector, topic, mean_exposure, std_exposure.

    Parameters
    ----------
    M_skt : (S, K, T) exposure tensor.
    sector_labels : list of length S.
    topic_labels : list of length K.
    """
    S, K, T = M_skt.shape
    records = []
    for s in range(S):
        for k in range(K):
            series = M_skt[s, k, :]
            records.append(
                {
                    "sector": sector_labels[s] if s < len(sector_labels) else f"sector_{s}",
                    "topic": topic_labels[k] if k < len(topic_labels) else f"topic_{k}",
                    "mean_exposure": float(series.mean()),
                    "std_exposure": float(series.std()),
                }
            )
    return pd.DataFrame(records)
