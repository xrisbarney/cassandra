"""BEA Input-Output table utilities.

Fetches the Summary Use table (TableID 259, "Use of Commodities by Industries")
from the BEA API, aggregates the ~71 BEA summary industries into the model's
11-sector classification, and computes technical coefficients and the Leontief
inverse for systemic risk propagation.

The parsing logic here was validated against the live BEA API (2022 Summary
Use table): gross output is read from the ``T018`` ("Total industry output")
row, ``F*`` columns are final demand, and ``T*``/``V*`` rows/columns are
totals and value added — all excluded from the intermediate-use matrix.
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
_DEFAULT_TABLE_ID = "259"       # Use of Commodities by Industries – Summary
_DEFAULT_YEAR = 2022
_GROSS_OUTPUT_ROW = "T018"      # "Total industry output (basic prices)"

# 11-sector aggregation
_DEFAULT_SECTOR_LABELS = [
    "Agriculture",             # 0
    "Mining",                  # 1
    "Utilities",               # 2
    "Construction",            # 3
    "Manufacturing",           # 4
    "Wholesale Trade",         # 5
    "Retail Trade",            # 6
    "Transportation",          # 7
    "Finance",                 # 8  (includes insurance & real estate)
    "Professional Services",   # 9
    "Government",              # 10
]

# Explicit BEA summary industry/commodity code → sector index (validated
_CODE_TO_SECTOR: dict[str, int] = {
    # Agriculture
    "111CA": 0, "113FF": 0,
    # Mining
    "211": 1, "212": 1, "213": 1,
    # Utilities
    "22": 2,
    # Construction
    "23": 3,
    # Manufacturing
    "311FT": 4, "313TT": 4, "315AL": 4, "321": 4, "322": 4, "323": 4,
    "324": 4, "325": 4, "326": 4, "327": 4, "331": 4, "332": 4, "333": 4,
    "334": 4, "335": 4, "3361MV": 4, "3364OT": 4, "337": 4, "339": 4,
    # Wholesale Trade
    "42": 5,
    # Retail Trade
    "441": 6, "445": 6, "452": 6, "4A0": 6,
    # Transportation & Warehousing
    "481": 7, "482": 7, "483": 7, "484": 7, "485": 7, "486": 7,
    "487OS": 7, "493": 7,
    # Finance, Insurance & Real Estate
    "521CI": 8, "523": 8, "524": 8, "525": 8, "532RL": 8, "HS": 8, "ORE": 8,
    # Professional / Information / Education / Health / Leisure / Other
    "511": 9, "512": 9, "513": 9, "514": 9, "5411": 9, "5412OP": 9,
    "5415": 9, "55": 9, "561": 9, "562": 9, "61": 9, "621": 9, "622": 9,
    "623": 9, "624": 9, "711AS": 9, "713": 9, "721": 9, "722": 9, "81": 9,
    # Government
    "GFE": 10, "GFGD": 10, "GFGN": 10, "GSLE": 10, "GSLG": 10,
}


def get_default_sector_labels() -> list[str]:
    """Return the 11 model sector labels (always length 11)."""
    return list(_DEFAULT_SECTOR_LABELS)


# One-sentence description + example constituent industries per sector, for
_SECTOR_DESCRIPTIONS = [
    "Farming, forestry, fishing, and hunting -- crop and animal production, logging, commercial fishing.",
    "Extraction of oil, gas, coal, metal ores, and other minerals.",
    "Electric power, natural gas, water, and sewage/waste systems.",
    "Building construction, heavy and civil engineering, and specialty trade contractors.",
    "Production of physical goods: food, textiles, chemicals, machinery, electronics, vehicles, and more -- the broadest sector here.",
    "Businesses that sell goods in bulk to retailers, other businesses, or institutions, rather than directly to consumers.",
    "Businesses selling goods directly to consumers, in stores or online.",
    "Moving people and goods by air, rail, water, truck, and transit, plus postal/courier and warehousing.",
    "Banking, securities and investment firms, insurance carriers, and real estate/rental services.",
    "A broad catch-all: professional/scientific/technical services, information and media, education, health care, arts and entertainment, and other services.",
    "Federal, state, and local government agencies and government enterprises.",
]


def get_sector_descriptions() -> list[str]:
    """Return a one-sentence description per sector (always length 11,
    aligned index-for-index with get_default_sector_labels())."""
    return list(_SECTOR_DESCRIPTIONS)


# BEA API fetch & cache

def _cache_path(cache_dir: str, year: int) -> Path:
    return Path(cache_dir) / f"bea_use_table_{year}.json"


def _fetch_raw_api(year: int, cache_dir: str) -> list[dict]:
    """Fetch the BEA Summary Use table rows for *year* (cached by year)."""
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

    beaapi = payload.get("BEAAPI", {})
    if "Error" in beaapi:
        raise RuntimeError(f"BEA API error: {beaapi['Error']}")

    # Results may be a dict or a single-element list depending on the request.
    results = beaapi["Results"]
    if isinstance(results, list):
        rows = results[0]["Data"]
    else:
        rows = results["Data"]

    if not rows:
        raise RuntimeError(f"BEA API returned no data for year {year}, TableID {_DEFAULT_TABLE_ID}")

    with cache_file.open("w") as fh:
        json.dump(rows, fh)
    logger.info("Cached %d BEA rows to %s", len(rows), cache_file)
    return rows


def _parse_value(val: str) -> float:
    """Parse a BEA DataValue string (may contain commas) to float."""
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _build_use_and_output(rows: list[dict], n_sectors: int) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate raw BEA rows into a sector×sector use matrix and output vector.

    Only cells whose row (commodity) and column (industry) codes both map to
    a model sector are counted as intermediate use; ``F*`` (final demand),
    ``T*`` (totals) and ``V*`` (value added) codes are excluded automatically
    because they are absent from :data:`_CODE_TO_SECTOR`.

    Returns
    -------
    U : np.ndarray, shape (n_sectors, n_sectors)
        Intermediate use, aggregated to sectors.  U[s, t] = commodity from
        sector s used by industry sector t.
    g : np.ndarray, shape (n_sectors,)
        Gross output per sector, from the ``T018`` row.
    """
    U = np.zeros((n_sectors, n_sectors), dtype=float)
    g = np.zeros(n_sectors, dtype=float)

    for row in rows:
        row_code = str(row.get("RowCode", "")).strip()
        col_code = str(row.get("ColCode", "")).strip()
        value = _parse_value(row.get("DataValue", 0))

        # Gross output: the T018 row gives total output for each industry column.
        if row_code == _GROSS_OUTPUT_ROW:
            s_col = _CODE_TO_SECTOR.get(col_code)
            if s_col is not None and s_col < n_sectors:
                g[s_col] += value
            continue

        # Intermediate use: both axes must be real (mapped) sectors.
        s_row = _CODE_TO_SECTOR.get(row_code)
        s_col = _CODE_TO_SECTOR.get(col_code)
        if s_row is not None and s_col is not None and s_row < n_sectors and s_col < n_sectors:
            U[s_row, s_col] += value

    return U, g


# Public API (called by scripts/build_features.py)

def get_technical_coefficients(
    year: int = _DEFAULT_YEAR,
    n_sectors: int = 11,
    cache_dir: str = "data/cache",
) -> np.ndarray:
    """Fetch the BEA Use table and return the direct-requirements matrix A.

    A[i, j] = dollars of sector-i input per dollar of sector-j gross output.

    Parameters
    ----------
    year : int
        BEA I-O table year (default 2022, the most recent Summary table).
    n_sectors : int
        Number of model sectors (default 11).
    cache_dir : str
        Directory used to cache the raw API response.

    Returns
    -------
    A : np.ndarray, shape (n_sectors, n_sectors)
    """
    rows = _fetch_raw_api(year, cache_dir)
    U, g = _build_use_and_output(rows, n_sectors)

    if not np.any(U):
        raise RuntimeError(
            "BEA Use matrix is all zeros — the code→sector mapping likely does "
            "not match this table year's industry codes. Check _CODE_TO_SECTOR."
        )

    g_safe = np.where(g == 0, np.nan, g)
    A = np.nan_to_num(U / g_safe[np.newaxis, :], nan=0.0)

    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    logger.info(
        "BEA A matrix: shape=%s  nonzero=%d/%d  spectral_radius=%.4f",
        A.shape, int((A > 0).sum()), A.size, rho,
    )
    return A


def get_sector_output(
    year: int = _DEFAULT_YEAR,
    n_sectors: int = 11,
    cache_dir: str = "data/cache",
) -> np.ndarray:
    """Return gross output by sector (millions of current dollars).

    Parameters
    ----------
    year : int
        BEA I-O table year.
    n_sectors : int
        Number of model sectors.
    cache_dir : str
        Directory used to cache the raw API response.

    Returns
    -------
    x_s : np.ndarray, shape (n_sectors,)
    """
    rows = _fetch_raw_api(year, cache_dir)
    _, g = _build_use_and_output(rows, n_sectors)
    if not np.any(g):
        raise RuntimeError(
            f"No gross-output ({_GROSS_OUTPUT_ROW}) values parsed from BEA table."
        )
    logger.info("BEA gross output: total = $%.2f trillion", g.sum() / 1e6)
    return g


# Core linear algebra

def leontief_inverse(A: np.ndarray) -> np.ndarray:
    """Compute the Leontief inverse L = (I - A)^{-1}.

    Raises
    ------
    ValueError
        If the spectral radius of *A* is ≥ 1 (non-productive economy; the
        series I + A + A² + … diverges).
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


# File-based loaders (fallback for local CSV/Excel copies of BEA tables)

def load_use_table(path: str) -> tuple[np.ndarray, list[str]]:
    """Load a BEA Use table from a local CSV/Excel file and return (A, labels)."""
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
    """Load BEA gross output from a local CSV/Excel file and return (x_s, labels)."""
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
