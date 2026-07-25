"""
deepseek.py
===========
Thin client for DeepSeek's OpenAI-compatible chat completions API. Two uses:

1. explain_forecast() -- turn the forecast's numeric outputs (predictive
   quantiles, loss distribution) into a short plain-English summary.
2. label_topics() -- turn a fitted topic model's raw top-TF-IDF-word clusters
   into short, analyst-recognizable category names (e.g. "SQL Injection"
   instead of "Sql / Injection / Php").

Both are purely additive and optional: if DEEPSEEK_API_KEY is not set, or a
request fails for any reason, these return None and callers fall back to
their non-LLM behavior -- nothing else in the pipeline depends on either
succeeding.
"""
from __future__ import annotations

import json
import os
import re
import pandas as pd
import requests

_API_URL = "https://api.deepseek.com/chat/completions"
_MODEL = "deepseek-chat"

_SYSTEM_PROMPT = (
    "You explain cyber-threat forecasting model output to a non-technical "
    "executive audience (risk officers, not data scientists). You are given "
    "a fixed set of numbers and category tags extracted from the model's "
    "output. Write a short (150-200 word) plain-English summary of what the "
    "forecast says, which threat categories and economic sectors carry the "
    "most risk, and how confident the range of outcomes is.\n\n"
    "The category tags you are given (e.g. 'Sql / Injection / Php') are raw "
    "machine-learning labels, NOT natural phrases -- never quote them "
    "verbatim or wrap them in bold/asterisks. Instead, paraphrase each one "
    "into the plain security term it's pointing at (e.g. that tag means "
    "'SQL injection'). If a tag is too garbled to confidently interpret, "
    "describe it generically (e.g. 'one threat category') rather than "
    "repeating the raw tag text.\n\n"
    "Use ONLY the numbers provided -- never invent statistics, dates, or "
    "figures that are not given to you. State clearly that loss figures are "
    "totals accumulated over the full forecast horizon, not a monthly rate, "
    "and that activity figures are monthly counts. "
    "Write plain prose with no markdown at all: no headers, no bold/italic "
    "asterisks or underscores, no bullet points."
)

_TOPIC_LABEL_SYSTEM_PROMPT = (
    "You are given the top TF-IDF keywords for several unsupervised clusters "
    "of real-world software vulnerability (CVE) descriptions. For EACH "
    "cluster, return a short (2-5 word) canonical cybersecurity threat-"
    "category name that a security analyst would immediately recognize "
    "(e.g. 'SQL Injection', 'Cross-Site Scripting', 'Privilege Escalation', "
    "'Denial of Service', 'Information Disclosure', 'Remote Code "
    "Execution', 'Cross-Site Request Forgery').\n\n"
    "If a cluster's keywords are too generic, mixed, or boilerplate "
    "(report-writing filler, not a real vulnerability type) to identify a "
    "specific category, name it 'Miscellaneous Vulnerabilities' rather than "
    "guessing. Give different clusters different names when their keywords "
    "are genuinely different; only reuse a name if two clusters truly "
    "describe the same category.\n\n"
    "Respond with ONLY a JSON object mapping each cluster's index (as a "
    "string) to its plain-text name, e.g. "
    '{"0": "SQL Injection", "1": "Cross-Site Scripting"}. No other text, '
    "no markdown formatting inside the names."
)


def _strip_markdown(text: str) -> str:
    """Remove stray markdown the LLM might emit despite instructions not to.

    st.info() in the dashboard renders markdown, so a leftover/unmatched '**'
    or '_' from the model reads as randomly-inconsistent bold/italic text
    mid-sentence -- this makes the output safe to render as plain prose
    regardless of what the model actually returned.
    """
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text.strip()


def _summarize_topics(q_df: pd.DataFrame, top_n: int = 3) -> str:
    """Rank topics by their final-horizon forecast level and by growth over
    the horizon, and describe the top few of each in plain text."""
    by_topic = q_df.groupby("topic_label")
    rows = []
    for label, g in by_topic:
        g = g.sort_values("date")
        first_mean = float(g["mean_cves_per_month"].iloc[0])
        last_mean = float(g["mean_cves_per_month"].iloc[-1])
        growth = last_mean - first_mean
        rows.append({"label": label, "first": first_mean, "last": last_mean, "growth": growth})
    summary_df = pd.DataFrame(rows)

    top_level = summary_df.sort_values("last", ascending=False).head(top_n)
    top_growth = summary_df.sort_values("growth", ascending=False).head(top_n)

    lines = [
        "Threat category tags below are raw ML labels (words joined by '/'), "
        "not natural phrases -- paraphrase them, don't quote them.",
        "Highest projected activity by threat category tag (final forecast month, CVEs/month):",
    ]
    for _, r in top_level.iterrows():
        lines.append(f"  - tag[{r['label']}]: ~{r['last']:.1f} CVEs/month")
    lines.append("Fastest-rising threat category tags over the forecast horizon (CVEs/month):")
    for _, r in top_growth.iterrows():
        lines.append(f"  - tag[{r['label']}]: {r['first']:.1f} -> {r['last']:.1f} CVEs/month")
    return "\n".join(lines)


def _summarize_losses(loss_df: pd.DataFrame, top_n: int = 3) -> str:
    """Describe the aggregate loss distribution and the riskiest sectors."""
    var_col = next((c for c in loss_df.columns if c.startswith("VaR")), None)
    es_col = next((c for c in loss_df.columns if c.startswith("ES")), None)
    horizon = int(loss_df["horizon_months"].iloc[0]) if "horizon_months" in loss_df.columns else None

    lines = []
    if horizon:
        lines.append(f"All loss figures below are USD totals accumulated over {horizon} months.")

    agg = loss_df[loss_df["scope"] == "aggregate"]
    if not agg.empty:
        row = agg.iloc[0]
        lines.append(
            f"Aggregate projected economic loss: mean ${row['mean_usd_total']:,.0f}, "
            f"median ${row['q50_usd_total']:,.0f}, "
            f"90% range ${row['q05_usd_total']:,.0f} to ${row['q95_usd_total']:,.0f}"
            + (f", {var_col}=${row[var_col]:,.0f}" if var_col else "")
            + (f", {es_col}=${row[es_col]:,.0f}" if es_col else "") + "."
        )

    sectors = loss_df[loss_df["scope"] != "aggregate"]
    if not sectors.empty and var_col:
        top_sectors = sectors.sort_values(var_col, ascending=False).head(top_n)
        lines.append("Sectors with the highest projected loss exposure:")
        for _, r in top_sectors.iterrows():
            lines.append(f"  - {r['scope']}: mean ${r['mean_usd_total']:,.0f}, {var_col}=${r[var_col]:,.0f}")
    return "\n".join(lines)


def _call_deepseek(
    system_prompt: str,
    user_prompt: str,
    timeout: float = 30.0,
    json_mode: bool = False,
    max_tokens: int = 400,
) -> str | None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2 if json_mode else 0.3,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        resp = requests.post(
            _API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001 -- optional feature, never fatal
        print(f"      (DeepSeek request unavailable: {exc})")
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
    result = _call_deepseek(_SYSTEM_PROMPT, prompt)
    return _strip_markdown(result) if result else None


def label_topics(topic_top_words: list[list[str]]) -> list[str | None] | None:
    """
    Given each fitted topic's top TF-IDF words (see WikiTopicMapper.top_words),
    ask DeepSeek for a short, analyst-recognizable category name per topic.

    Returns a list the same length as topic_top_words, with None in any slot
    the model didn't return a name for (caller should fall back to the
    word-based label for that slot). Returns None entirely if
    DEEPSEEK_API_KEY is not set or the request fails.
    """
    prompt = "\n".join(
        f"{k}: {', '.join(words)}" for k, words in enumerate(topic_top_words)
    )
    content = _call_deepseek(
        _TOPIC_LABEL_SYSTEM_PROMPT, prompt, json_mode=True, max_tokens=300,
    )
    if not content:
        return None
    try:
        mapping = json.loads(content)
    except json.JSONDecodeError as exc:
        print(f"      (DeepSeek topic labels: invalid JSON response: {exc})")
        return None
    return [
        _strip_markdown(str(mapping[str(k)])).strip() if str(k) in mapping else None
        for k in range(len(topic_top_words))
    ]


_BACKTEST_SYSTEM_PROMPT = (
    "You explain, to a non-technical risk-officer audience, how accurate a "
    "past cyber-threat forecast turned out to be now that the real data for "
    "that period is in. You are given: per-topic actual-vs-predicted CVE "
    "counts, how often the actual value fell inside the model's stated 90% "
    "uncertainty range ('coverage' -- close to 90% is good; much lower means "
    "the model was overconfident, much higher means it was underconfident), "
    "and standard error metrics (CRPS/MAE/RMSE, lower is better, no fixed "
    "'good' threshold -- only usable by comparing across topics/horizons). "
    "Category tags (e.g. 'Sql / Injection / Php') are raw ML labels -- "
    "paraphrase them into plain security terms, never quote them verbatim. "
    "Write 150-200 words of plain prose (no markdown, no bold/italic "
    "asterisks or underscores, no bullet points): say plainly whether the "
    "forecast under- or over-predicted overall, which categories it got most "
    "and least right, and what the coverage number implies about how much to "
    "trust its stated uncertainty ranges going forward. Use ONLY the numbers "
    "given -- never invent a statistic you were not given."
)


def explain_backtest(detail_df: pd.DataFrame, scores_df: pd.DataFrame, coverage: float) -> str | None:
    """
    Build a plain-English "how did the forecast do" summary from a backtest
    comparison (scripts/backtest.py): per-topic actual-vs-predicted detail,
    aggregate CRPS/MAE/RMSE by horizon, and the empirical 90%-interval
    coverage rate. Returns None if DEEPSEEK_API_KEY is not set or the API
    call fails.
    """
    by_topic = detail_df.groupby("topic_label").agg(
        actual_total=("actual_cves", "sum"),
        predicted_total=("predicted_mean_cves", "sum"),
    )
    by_topic["error"] = by_topic["predicted_total"] - by_topic["actual_total"]
    over = by_topic.sort_values("error", ascending=False).head(3)
    under = by_topic.sort_values("error").head(3)

    lines = [
        f"Empirical 90% interval coverage across all topic-months: {coverage:.1%} "
        "(target is ~90%; well below means the model was overconfident, well "
        "above means it was underconfident).",
        "",
        "Error metrics by horizon (lower is better; compare across rows, no fixed threshold):",
        scores_df.to_string(index=False),
        "",
        "Categories the forecast OVER-predicted most (predicted total - actual total, summed over the period):",
    ]
    for label, row in over.iterrows():
        lines.append(f"  - tag[{label}]: predicted {row['predicted_total']:.0f} vs actual {row['actual_total']:.0f}")
    lines.append("Categories the forecast UNDER-predicted most:")
    for label, row in under.iterrows():
        lines.append(f"  - tag[{label}]: predicted {row['predicted_total']:.0f} vs actual {row['actual_total']:.0f}")

    result = _call_deepseek(_BACKTEST_SYSTEM_PROMPT, "\n".join(lines) + "\n\nWrite the summary now.")
    return _strip_markdown(result) if result else None
