"""BEA Input-Output table utilities.

Fetches Use tables from the BEA API (InputOutput dataset), aggregates to
the model's S-sector classification, and computes technical coefficients
and the Leontief inverse for systemic risk propagation analysis.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BEA_API_URL = "https://apps.bea.gov/api/data"
# Summary-level Use table (before redefinitions) — most recent stable year
_DEFAULT_TABLE_ID = "259"
_DEFAULT_YEAR = 2022

# ---------------------------------------------------------------------------
# 11-sector aggregation mapping
# BEA RowCode/ColCode 2-digit NAICS prefix → sector index 0..10
# ---------------------------------------------------------------------------
_DEFAULT_SECTOR_LABELS = [
    "Agriculture",        # 0
    "Mining",             # 1
    "Utilities",          # 2
    "Construction",       # 3
    "Manufacturing",      # 4
    "Wholesale Trade",    # 5
    "Retail Trade",       # 6
    "Transportation",     # 7
    "Finance & Real Estate",  # 8
    "Professional Services",  # 9
    "Government",         # 10
]

_CODE_TO_SECTOR: dict[str, int] = {
    "11": 0,              # Agriculture, forestry, fishing, hunting
    "21": 1,              # Mining
    "22": 2,              # Utilities
    "23": 3,              # Construction
    "31": 4, "32": 4, "33": 4,  # Manufacturing
    "42": 5,              # Wholesale Trade
    "44": 6, "45": 6,    # Retail Trade
    "48": 7, "49": 7,    # Transportation & Warehousing
    "51": 9,              # Information → Professional Services
    "52": 8, "53": 8,    # Finance & Insurance, Real Estate
    "54": 9, "55": 9, "56": 9,  # Professional & Business Services
    "61": 9,              # Educational Services
    "62": 9,              # Health Care
    "71": 9,              # Arts & Entertainment
    "72": 9,              # Accommodation & Food
    "81": 9,              # Other Services
    "92": 10,             # Government
}


def get_default_sector_labels() -> list[str]:
    return list(_DEFAULT_SECTOR_LABELS)


# ---------------------------------------------------------------------------
# BEA API fetch & cache
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: str, year: int) -> Path:
    return Path(cache_dir) / f"bea_use_table_{year}.json"


def _fetch_raw_api(year: int, cache_dir: str) -> list[dict]:
    """Fetch the BEA Summary Use table for *year* and return raw row list."""
    api_key = os.environ.get("BEA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "BEA_API_KEY environment variable is not set. "
            "Get a free key at https://apps.bea.gov/api/signup/"
        )

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir, year)

    if cache_file.exists():
        logger.info("BEA I-O cache hit: %s", cache_file)
        with cache_file.open() as fh:
            return json.load(fh)

    params = {
        "UserID": api_key,
        "method": "GetData",
        "DataSetName": "InputOutput",
        "TableID": _DEFAULT_TABLE_ID,
        "Year": str(year),
        "ResultFormat": "JSON",
    }

    logger.info("Fetching BEA Use table (TableID=%s, Year=%d) ...", _DEFAULT_TABLE_ID, year)
    resp = requests.get(_BEA_API_URL, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    # Surface any BEA API-level error messages
    if "Error" in payload.get("BEAAPI", {}):
        raise RuntimeError(f"BEA API error: {payload['BEAAPI']['Error']}")

    rows: list[dict] = payload["BEAAPI"]["Results"]["Data"]
    with cache_file.open("w") as fh:
        json.dump(rows, fh)
    logger.info("Cached %d BEA rows to %s", len(rows), cache_file)
    return rows


# ---------------------------------------------------------------------------
# Parsing & aggregation
# ---------------------------------------------------------------------------

def _parse_value(val: str) -> float:
    """Parse BEA data value string (may contain commas) to float."""
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _code_prefix(code: str) -> str:
    """Return first 2 characters of a BEA industry code."""
    return str(code).strip()[:2]


def _build_use_matrix(rows: list[dict], n_sectors: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (n_sectors × n_sectors) intermediate-use matrix and gross output vector.

    Returns
    -------
    U : np.ndarray, shape (n_sectors, n_sectors)
        Intermediate use: U[i, j] = commodity i used by industry j.
    x : np.ndarray, shape (n_sectors,)
        Gross output by sector (from the Total Industry Output row).
    """
    U = np.zeros((n_sectors, n_sectors), dtype=float)
    x = np.zeros(n_sectors, dtype=float)

    for row in rows:
        row_code = _code_prefix(row.get("RowCode", ""))
        col_code = _code_prefix(row.get("ColCode", ""))
        row_type = str(row.get("RowType", "")).strip()
        col_type = str(row.get("ColType", "")).strip()
        value = _parse_value(row.get("DataValue", 0))

        # Gross output row: ColType="I" (industry column), special RowCode "T017"
        # or RowDescr containing "Total industry output"
        row_descr = str(row.get("RowDescr", "")).lower()
        if "total industry output" in row_descr and col_type == "I":
            s = _CODE_TO_SECTOR.get(col_code)
            if s is not None and s < n_sectors:
                x[s] += value
            continue

        # Intermediate use: both row and column are industries
        if row_type != "I" or col_type != "I":
            continue

        s_row = _CODE_TO_SECTOR.get(row_code)
        s_col = _CODE_TO_SECTOR.get(col_code)
        if s_row is not None and s_col is not None and s_row < n_sectors and s_col < n_sectors:
            U[s_row, s_col] += value

    return U, x


# ---------------------------------------------------------------------------
# Public API (called by build_features.py)
# ---------------------------------------------------------------------------

def get_technical_coefficients(
    year: int = _DEFAULT_YEAR,
    n_sectors: int = 11,
    cache_dir: str = "data/cache",
) -> np.ndarray:
    """Fetch BEA Use table and return the direct-requirements matrix A.

    Parameters
    ----------
    year : int
        BEA I-O table year.  Defaults to 2022 (most recent Summary table).
    n_sectors : int
        Number of sectors in the model.
    cache_dir : str
        Directory used to cache raw API responses.

    Returns
    -------
    A : np.ndarray, shape (n_sectors, n_sectors)
        Technical coefficient matrix.  A[i, j] = intermediate input from
        sector i per dollar of output in sector j.
    """
    rows = _fetch_raw_api(year, cache_dir)
    U, x = _build_use_matrix(rows, n_sectors)

    x_safe = np.where(x == 0, np.nan, x)
    A = U / x_safe[np.newaxis, :]
    A = np.nan_to_num(A, nan=0.0)

    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    logger.info("A matrix: shape=%s  spectral_radius=%.4f", A.shape, rho)
    return A


def get_sector_output(
    year: int = _DEFAULT_YEAR,
    n_sectors: int = 11,
    cache_dir: str = "data/cache",
) -> np.ndarray:
    """Return gross output by sector (millions of dollars) from BEA Use table.

    Parameters
    ----------
    year : int
        BEA I-O table year.
    n_sectors : int
        Number of sectors in the model.
    cache_dir : str
        Directory used to cache raw API responses.

    Returns
    -------
    x_s : np.ndarray, shape (n_sectors,)
        Gross output in millions of current dollars.
    """
    rows = _fetch_raw_api(year, cache_dir)
    _, x = _build_use_matrix(rows, n_sectors)
    logger.info("Gross output: total = $%.1f trillion", x.sum() / 1e6)
    return x


# ---------------------------------------------------------------------------
# Core linear algebra (unchanged)
# ---------------------------------------------------------------------------

def leontief_inverse(A: np.ndarray) -> np.ndarray:
    """Compute the Leontief inverse L = (I - A)^{-1}.

    Raises
    ------
    ValueError
        If spectral radius of A ≥ 1 (non-productive economy).
    """
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    if rho >= 1.0:
        raise ValueError(
            f"Spectral radius of A is {rho:.6f} ≥ 1. "
            "Leontief inverse does not converge for a non-productive economy."
        )
    L = np.linalg.inv(np.eye(A.shape[0], dtype=A.dtype) - A)
    logger.info("Leontief inverse: max element = %.4f", float(L.max()))
    return L


# ---------------------------------------------------------------------------
# File-based loaders (kept as fallback if someone has local CSVs)
# ---------------------------------------------------------------------------

def load_use_table(path: str) -> tuple[np.ndarray, list[str]]:
    """Load a BEA Use table from a local CSV/Excel file."""
    fpath = Path(path)
    if not fpath.exists():
        raise FileNotFoundError(f"BEA Use table not found: {path}")

    raw = pd.read_excel(fpath, index_col=0) if fpath.suffix.lower() in {".xlsx", ".xls"} \
        else pd.read_csv(fpath, index_col=0)
    raw.index = raw.index.astype(str).str.strip()
    raw.columns = raw.columns.astype(str).str.strip()

    output_candidates = [i for i in raw.index
                         if any(k in i.lower() for k in ("total industry output", "gross output"))]
    gross_output_row = raw.loc[output_candidates[0]] if output_candidates else raw.sum(axis=0)
    intermediate = raw.drop(index=output_candidates) if output_candidates else raw

    common = [c for c in intermediate.columns if c in intermediate.index] or list(intermediate.columns)
    U = intermediate.loc[common, common].values.astype(float)
    x = gross_output_row[common].values.astype(float)
    x_safe = np.where(x == 0, np.nan, x)
    A = np.nan_to_num(U / x_safe[np.newaxis, :], nan=0.0)
    return A, common


def load_gross_output(path: str) -> tuple[np.ndarray, list[str]]:
    """Load BEA gross output from a local CSV/Excel file."""
    fpath = Path(path)
    if not fpath.exists():
        raise FileNotFoundError(f"BEA gross output file not found: {path}")

    raw = pd.read_excel(fpath, index_col=0) if fpath.suffix.lower() in {".xlsx", ".xls"} \
        else pd.read_csv(fpath, index_col=0)
    raw.index = raw.index.astype(str).str.strip()
    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError(f"No numeric columns in {path}")
    x_s = raw[numeric_cols[-1]].values.astype(float)
    return x_s, list(raw.index)
