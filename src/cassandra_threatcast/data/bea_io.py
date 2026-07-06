"""BEA Input-Output table utilities.

Loads BEA Use tables and gross output data, computes technical coefficients
and the Leontief inverse for systemic risk propagation analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# BEA Summary-level 11-sector aggregation (matches BEA GDP-by-industry)
_DEFAULT_SECTOR_LABELS = [
    "Agriculture",
    "Mining",
    "Utilities",
    "Construction",
    "Manufacturing",
    "Wholesale Trade",
    "Retail Trade",
    "Transportation",
    "Finance",
    "Professional Services",
    "Government",
]


def get_default_sector_labels() -> list[str]:
    """Return the 11 BEA summary-level sector labels.

    Returns
    -------
    list[str]
        Ordered list of sector names corresponding to BEA summary-level
        industry aggregation.  Length is always 11.
    """
    return list(_DEFAULT_SECTOR_LABELS)


def load_use_table(path: str) -> tuple[np.ndarray, list[str]]:
    """Load a BEA Use table and compute technical coefficients.

    Reads a CSV or Excel file containing the BEA Use (commodity × industry)
    table.  The file is expected to have industry sector labels either as
    column headers or as the first row.  Row labels (commodities / sectors)
    are expected in the first column.

    The function extracts the *intermediate-use* sub-matrix (industries as
    both rows and columns), then computes the direct-requirements (technical
    coefficient) matrix ``A`` by dividing each column by the sector's gross
    output (the last row labelled ``"Total Industry Output"`` or similar, or
    the column sum if absent).

    Parameters
    ----------
    path:
        Absolute path to the Use table CSV or Excel file.

    Returns
    -------
    A : np.ndarray, shape (S, S)
        Technical coefficient matrix.  ``A[i, j]`` is the dollar of commodity
        ``i`` required per dollar of output in industry ``j``.
    sector_labels : list[str]
        Ordered list of sector labels (length S).
    """
    fpath = Path(path)
    if not fpath.exists():
        raise FileNotFoundError(f"BEA Use table not found: {path}")

    if fpath.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(fpath, index_col=0, header=0)
    else:
        raw = pd.read_csv(fpath, index_col=0, header=0)

    raw.index = raw.index.astype(str).str.strip()
    raw.columns = raw.columns.astype(str).str.strip()

    # Identify gross-output row: look for a row labelled with "output" or "total"
    output_row_candidates = [
        idx for idx in raw.index
        if any(kw in idx.lower() for kw in ("total industry output", "gross output", "total output"))
    ]

    if output_row_candidates:
        gross_output_row = raw.loc[output_row_candidates[0]]
        intermediate = raw.drop(index=output_row_candidates)
    else:
        # Fall back: use column sums as proxy for gross output
        gross_output_row = raw.sum(axis=0)
        intermediate = raw

    # Keep only columns/rows that appear in both dimensions (square sub-matrix)
    common = [c for c in intermediate.columns if c in intermediate.index]
    if not common:
        # If no intersection, assume the table is already square and use all
        common_cols = list(intermediate.columns)
        common_rows = list(intermediate.index[: len(common_cols)])
        U = intermediate.loc[common_rows, common_cols].values.astype(float)
        sector_labels = common_cols
        x = gross_output_row[common_cols].values.astype(float)
    else:
        U = intermediate.loc[common, common].values.astype(float)
        sector_labels = common
        x = gross_output_row[common].values.astype(float)

    # Replace zeros in gross output with NaN to avoid division by zero
    x_safe = np.where(x == 0, np.nan, x)
    A = U / x_safe[np.newaxis, :]  # broadcast: divide each column by x_j
    A = np.nan_to_num(A, nan=0.0)  # treat undefined coefficients as 0

    logger.info(
        "Loaded BEA Use table: %d sectors, spectral radius of A = %.4f",
        len(sector_labels),
        float(np.max(np.abs(np.linalg.eigvals(A)))),
    )
    return A, sector_labels


def leontief_inverse(A: np.ndarray) -> np.ndarray:
    """Compute the Leontief inverse ``L = (I - A)^{-1}``.

    Parameters
    ----------
    A : np.ndarray, shape (S, S)
        Technical coefficient matrix.  Must satisfy spectral radius < 1 for
        the Leontief inverse to have an economic interpretation.

    Returns
    -------
    L : np.ndarray, shape (S, S)
        Leontief inverse (total-requirements) matrix.

    Raises
    ------
    ValueError
        If the spectral radius of *A* is ≥ 1, indicating a non-productive
        economy (the series expansion ``I + A + A² + …`` diverges).
    """
    eigenvalues = np.linalg.eigvals(A)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    if spectral_radius >= 1.0:
        raise ValueError(
            f"Spectral radius of A is {spectral_radius:.6f} ≥ 1.  "
            "The Leontief inverse does not converge for a non-productive economy."
        )

    S = A.shape[0]
    I = np.eye(S, dtype=A.dtype)
    L = np.linalg.inv(I - A)
    logger.info("Leontief inverse computed; max element = %.4f", float(L.max()))
    return L


def load_gross_output(path: str) -> tuple[np.ndarray, list[str]]:
    """Load BEA GDP-by-industry gross output data.

    Reads a CSV or Excel file with industry labels in the first column and
    at least one numeric column representing gross output in millions of
    current dollars.  If multiple year columns are present the most recent
    one is used.

    Parameters
    ----------
    path:
        Absolute path to the gross output CSV or Excel file.

    Returns
    -------
    x_s : np.ndarray, shape (S,)
        Gross output in millions of current dollars, one value per sector.
    sector_labels : list[str]
        Ordered sector labels corresponding to *x_s*.
    """
    fpath = Path(path)
    if not fpath.exists():
        raise FileNotFoundError(f"BEA gross output file not found: {path}")

    if fpath.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(fpath, index_col=0, header=0)
    else:
        raw = pd.read_csv(fpath, index_col=0, header=0)

    raw.index = raw.index.astype(str).str.strip()

    # Drop rows that look like headers or totals
    drop_patterns = ("total", "all industries", "private industries", "government")
    mask = ~raw.index.str.lower().str.startswith(drop_patterns)
    raw = raw[mask]

    # Select the rightmost numeric column (most recent year)
    numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        # Try coercing all columns
        for col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise ValueError(f"No numeric columns found in gross output file: {path}")

    latest_col = numeric_cols[-1]
    x_s = raw[latest_col].values.astype(float)
    sector_labels = list(raw.index)

    logger.info("Loaded gross output for %d sectors (column: %s)", len(sector_labels), latest_col)
    return x_s, sector_labels
