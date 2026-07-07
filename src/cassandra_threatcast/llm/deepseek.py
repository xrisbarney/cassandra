"""
deepseek.py
===========
Thin client for DeepSeek's OpenAI-compatible chat completions API, used to
turn the forecast's numeric outputs (predictive quantiles, loss distribution)
into a short, plain-English summary for non-technical readers.

Purely additive and optional: if DEEPSEEK_API_KEY is not set, or the request
fails for any reason, explain_forecast() returns None and callers fall back
to showing the raw tables/figures with no summary -- nothing else in the
pipeline depends on this succeeding.
"""
from __future__ import annotations

import os
import pandas as pd
import requests

_API_URL = "https://api.deepseek.com/chat/completions"
_MODEL = "deepseek-chat"

_SYSTEM_PROMPT = (
    "You explain cyber-threat forecasting model output to a non-technical "
    "executive audience (risk officers, not data scientists). You are given "
    "a fixed set of numbers extracted from the model's output. Write a short "
    "(150-200 word) plain-English summary of what the forecast says, which "
    "threat categories and economic sectors carry the most risk, and how "
    "confident the range of outcomes is. Use ONLY the numbers provided -- "
    "never invent statistics, dates, or figures that are not given to you. "
    "Do not use markdown headers; a few short paragraphs is fine."
)


def _summarize_topics(q_df: pd.DataFrame, top_n: int = 3) -> str:
    """Rank topics by their final-horizon forecast level and by growth over
    the horizon, and describe the top few of each in plain text."""
    by_topic = q_df.groupby("topic_label")
    rows = []
    for label, g in by_topic:
        g = g.sort_values("date")
        first_mean = float(g["mean"].iloc[0])
        last_mean = float(g["mean"].iloc[-1])
        growth = last_mean - first_mean
        rows.append({"label": label, "first": first_mean, "last": last_mean, "growth": growth})
    summary_df = pd.DataFrame(rows)

    top_level = summary_df.sort_values("last", ascending=False).head(top_n)
    top_growth = summary_df.sort_values("growth", ascending=False).head(top_n)

    lines = ["Highest projected activity by threat category (final forecast month):"]
    for _, r in top_level.iterrows():
        lines.append(f"  - {r['label']}: ~{r['last']:.1f} events/month")
    lines.append("Fastest-rising threat categories over the forecast horizon:")
    for _, r in top_growth.iterrows():
        lines.append(f"  - {r['label']}: {r['first']:.1f} -> {r['last']:.1f} events/month")
    return "\n".join(lines)


def _summarize_losses(loss_df: pd.DataFrame, top_n: int = 3) -> str:
    """Describe the aggregate loss distribution and the riskiest sectors."""
    var_col = next((c for c in loss_df.columns if c.startswith("VaR")), None)
    es_col = next((c for c in loss_df.columns if c.startswith("ES")), None)

    lines = []
    agg = loss_df[loss_df["scope"] == "aggregate"]
    if not agg.empty:
        row = agg.iloc[0]
        lines.append(
            f"Aggregate projected economic loss: mean ${row['mean']:,.0f}, "
            f"median ${row['q50']:,.0f}, "
            f"90% range ${row['q05']:,.0f} to ${row['q95']:,.0f}"
            + (f", {var_col}=${row[var_col]:,.0f}" if var_col else "")
            + (f", {es_col}=${row[es_col]:,.0f}" if es_col else "") + "."
        )

    sectors = loss_df[loss_df["scope"] != "aggregate"]
    if not sectors.empty and var_col:
        top_sectors = sectors.sort_values(var_col, ascending=False).head(top_n)
        lines.append("Sectors with the highest projected loss exposure:")
        for _, r in top_sectors.iterrows():
            lines.append(f"  - {r['scope']}: mean ${r['mean']:,.0f}, {var_col}=${r[var_col]:,.0f}")
    return "\n".join(lines)


def _call_deepseek(prompt: str, timeout: float = 30.0) -> str | None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    try:
        resp = requests.post(
            _API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": _MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 400,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001 -- optional feature, never fatal
        print(f"      (DeepSeek summary unavailable: {exc})")
        return None


def explain_forecast(q_df: pd.DataFrame, loss_df: pd.DataFrame) -> str | None:
    """
    Build a plain-English summary of a forecast run from its two output
    tables (forecast_quantiles.csv and loss_distribution.csv, already loaded
    as DataFrames). Returns None if DEEPSEEK_API_KEY is not set or the API
    call fails -- this is always an optional enhancement, never required.
    """
    prompt = (
        "Here is the cyber-threat forecast data:\n\n"
        + _summarize_topics(q_df)
        + "\n\n"
        + _summarize_losses(loss_df)
        + "\n\nWrite the summary now."
    )
    return _call_deepseek(prompt)
