"Optional curated incident-loss marks used by paper Steps 1, 4, and 34."
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_monthly_loss_marks(path: str | Path, months, S: int) -> np.ndarray:
    "Aggregate a curated CSV to sector/month loss marks in US dollars."
    out = np.full((S, len(months)), np.nan)
    source = Path(path)
    if not source.exists():
        return out
    frame = pd.read_csv(source)
    required = {"date", "sector_idx", "loss_usd"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    frame["month"] = pd.to_datetime(frame["date"]).dt.to_period("M")
    month_index = {month: idx for idx, month in enumerate(months)}
    grouped = frame.groupby(["sector_idx", "month"], observed=True)["loss_usd"].sum()
    for (sector, month), value in grouped.items():
        if 0 <= int(sector) < S and month in month_index:
            out[int(sector), month_index[month]] = float(value)
    return out
