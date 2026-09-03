"""
CASSANDRA — interactive dashboard.

A friendly, click-through interface to the cyber-threat forecasting pipeline.
Run it with:

    streamlit run app.py

Each step is a button. You do not need to touch the command line.
"""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from cassandra_threatcast.data.bea_io import get_default_sector_labels, get_sector_descriptions
from cassandra_threatcast.features.topic_map import load_topic_labels

ROOT = Path(__file__).parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
PY = sys.executable
RUN_LOG = RESULTS / "run_all.log"
RUN_PID = RESULTS / "run_all.pid"

SECTOR_LABELS = get_default_sector_labels()
SECTOR_DESCRIPTIONS = dict(zip(SECTOR_LABELS, get_sector_descriptions()))

st.set_page_config(page_title="CASSANDRA — Cyber Threat Forecasting", page_icon="🔮", layout="wide")


def sector_glossary_expander() -> None:
    """Reference table of all 11 sectors and what they cover, for anyone who
    wants to browse definitions instead of hovering over each name."""
    with st.expander("ℹ️ What do these sectors mean?"):
        for name in SECTOR_LABELS:
            st.markdown(f"**{name}** — {SECTOR_DESCRIPTIONS[name]}")


def _sector_table_html(df: pd.DataFrame, name_col: str = "scope") -> str:
    """Render a small DataFrame as an HTML table with a native hover tooltip
    (title attribute) on any cell whose value is a known sector name."""
    header = "".join(
        f"<th style='padding:4px 10px;text-align:left;border-bottom:1px solid rgba(128,128,128,0.4)'>{html.escape(str(c))}</th>"
        for c in df.columns
    )
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            val = row[c]
            if c == name_col and val in SECTOR_DESCRIPTIONS:
                cells.append(
                    "<td style='padding:4px 10px;'>"
                    f"<span title=\"{html.escape(SECTOR_DESCRIPTIONS[val])}\" "
                    "style='border-bottom:1px dotted currentColor;cursor:help;'>"
                    f"{html.escape(str(val))}</span></td>"
                )
            elif isinstance(val, float):
                cells.append(f"<td style='padding:4px 10px;text-align:right;'>{val:,.0f}</td>")
            else:
                weight = "font-weight:600;" if c == name_col else ""
                cells.append(f"<td style='padding:4px 10px;{weight}'>{html.escape(str(val))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<div style='overflow-x:auto;'><table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


# Helpers
def artifact_exists(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


def panel_has_data() -> bool:
    f = PROCESSED / "N.npy"
    if not f.exists():
        return False
    try:
        return float(np.nansum(np.load(f))) > 0
    except Exception:
        return False


def run_step(cmd: list[str], title: str) -> bool:
    """Run a pipeline script, streaming its output into the page. Returns success."""
    log_area = st.empty()
    lines: list[str] = []
    with st.status(f"Running: {title} …", expanded=True) as status:
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.append(line.rstrip())
                log_area.code("\n".join(lines[-25:]))   # show last 25 lines
            proc.wait()
        except Exception as exc:  # noqa: BLE001
            status.update(label=f"{title} — could not start ({exc})", state="error")
            return False

        if proc.returncode == 0:
            status.update(label=f"{title} — done ✅", state="complete")
            return True
        status.update(label=f"{title} — something went wrong ❌", state="error")
        return False


def status_badge(done: bool, ready: bool = True) -> str:
    if done:
        return "🟢 Done"
    if not ready:
        return "⚪ Waiting for the previous step"
    return "🔵 Ready to run"


# ---- Unattended "run everything" support ----------------------------------
def launch_full_run(fresh: bool, quick: bool, enhanced: bool, start: str, end: str) -> None:
    """Start the whole pipeline as a detached background process (survives closing
    this browser window). Output is streamed to results/run_all.log."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "scripts/run_all.py", "--start", start, "--end", end]
    if fresh:
        cmd.append("--fresh")
    if quick:
        cmd.append("--quick")
    if enhanced:
        cmd.append("--enhanced")
    logf = open(RUN_LOG, "w", encoding="utf-8")  # noqa: SIM115 (child keeps writing)
    child_env = {**os.environ, "PYTHONUTF8": "1"}  # belt-and-suspenders; run_all.py also sets this
    if os.name == "nt":
        flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
                                env=child_env, creationflags=flags)
    else:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
                                env=child_env, start_new_session=True)
    RUN_PID.write_text(str(proc.pid))


def full_run_state() -> tuple[str, str]:
    """Return (state, log_text): state is none/running/done/failed."""
    if not RUN_LOG.exists():
        return "none", ""
    # encoding="utf-8" is required here: the log is written in UTF-8 (see
    text = RUN_LOG.read_text(encoding="utf-8", errors="replace")
    if "ALL DONE" in text:
        return "done", text
    if "FAILED" in text:
        return "failed", text
    return "running", text


def stop_full_run() -> None:
    if not RUN_PID.exists():
        return
    pid = RUN_PID.read_text().strip()
    if not pid:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", pid], capture_output=True)
    else:
        import signal
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass


@st.fragment(run_every="5s")
def live_full_run_log() -> None:
    """Auto-refreshing view of the unattended run's progress."""
    state, text = full_run_state()
    if state == "none":
        return
    done = len(re.findall(r"DONE \d/5", text))
    st.progress(min(done / 5.0, 1.0), text=f"{done} of 5 steps complete")
    if state == "done":
        st.success("✅ All finished! Open the **📊 Your data** and **🔮 The forecast** tabs.")
    elif state == "failed":
        st.error("❌ A step failed — the log below shows what happened.")
    else:
        st.info("⏳ Running… you can safely close this window; it keeps running in the background.")
    lines = text.splitlines()
    st.code("\n".join(lines[-25:]) or "(starting…)")


# Header
st.title("🔮 CASSANDRA")
st.caption("Forecasting cyber threats and their economic impact — a guided, step-by-step tool.")

st.markdown(
    "This tool builds a forecast in five steps, plus an optional sixth once the "
    "forecast window is in the past. Do them **in order, top to bottom**. "
    "Each step shows a green light once it has finished. You can watch the progress "
    "log as it runs — you don't need to understand it."
)

# Settings (sidebar)
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Sensible defaults are already chosen. You can leave these as they are.")
    start = st.text_input("Start month", value="2010-01", help="Earliest month of data to use (YYYY-MM).")
    end = st.text_input("End month", value="2024-12", help="Latest month of data to use (YYYY-MM).")

    run_mode = st.radio(
        "Training effort",
        ["Quick preview (minutes)", "Full run (slow, most accurate)"],
        help="Quick preview uses fewer computations so you can see results fast. "
             "Use the full run for final numbers.",
    )
    enhanced = st.toggle(
        "Use the enhanced model",
        value=False,
        help="An experimental variant with heavier-tailed assumptions. Leave off to use "
             "the standard model from the paper.",
    )
    st.divider()
    st.caption("Data and results are saved on this computer under `data/` and `results/`.")

quick = run_mode.startswith("Quick")

# Current status
done_ingest = panel_has_data()
done_features = artifact_exists(PROCESSED / "M_skt.npy", PROCESSED / "e_t.npy", PROCESSED / "Lambda_L.npy")
done_train = artifact_exists(RESULTS / "idata.nc")
done_eval = artifact_exists(RESULTS / "evaluation" / "scores.csv")
done_forecast = artifact_exists(RESULTS / "forecasts" / "forecast_quantiles.csv")
done_backtest = artifact_exists(RESULTS / "backtest" / "threat_comparison.csv")

tab_run, tab_data, tab_forecast, tab_kernel = st.tabs(
    ["▶️ Run the steps", "📊 Your data", "🔮 The forecast", "🧪 Kernel lab"])

# TAB 1 — Run the steps
with tab_run:
    # Run-everything (unattended)
    with st.container(border=True):
        st.subheader("🌙 Run everything, unattended")
        st.write(
            "Runs all five steps in order, then leaves the results ready for you. "
            "Perfect for leaving overnight — **you can close this window and it keeps "
            "running** in the background. Come back and reopen the dashboard to check on it."
        )
        fresh = st.checkbox(
            "Start completely fresh — delete all downloaded data and results first",
            value=False,
            help="Re-downloads everything from scratch. This is the slowest option: the "
                 "vulnerability download alone can take 30–60 minutes.",
        )
        if fresh:
            st.warning(
                "This will permanently delete everything under `data/` and `results/` and "
                "re-download from the internet. Only the raw code and your keys are kept."
            )
        c_go, c_stop = st.columns([2, 1])
        with c_go:
            if st.button("🚀 Run the whole pipeline now", type="primary", key="runall"):
                launch_full_run(fresh, quick, enhanced, start, end)
                st.rerun()
        with c_stop:
            if st.button("⏹ Stop", key="stopall"):
                stop_full_run()
                st.toast("Asked the pipeline to stop.")
        live_full_run_log()

    st.divider()
    st.caption("Prefer to go step by step? Use the buttons below (one at a time, top to bottom).")

    # Step 1
    with st.container(border=True):
        st.subheader(f"Step 1 — Collect the data   {status_badge(done_ingest)}")
        st.write(
            "Downloads public cyber-threat information: software vulnerabilities, how likely "
            "each is to be exploited, and company breach disclosures. Organises it by month. "
            "**The first time, this can take a while** (it downloads a lot); afterwards it's cached and fast."
        )
        if st.button("Collect the data", key="b1", type="primary"):
            ok = run_step([PY, "scripts/ingest_data.py", "--start", start, "--end", end,
                           "--cache-dir", "data/cache/"], "Collecting data")
            if ok:
                st.rerun()

    # Step 2
    with st.container(border=True):
        st.subheader(f"Step 2 — Prepare the inputs   {status_badge(done_features, ready=done_ingest)}")
        st.write(
            "Turns the raw data into the tables the model needs, links each piece of software "
            "to the economic sectors it affects, and pulls in the official economic linkage tables."
        )
        if st.button("Prepare the inputs", key="b2", disabled=not done_ingest):
            if run_step([PY, "scripts/build_features.py"], "Preparing inputs"):
                st.rerun()
        if not done_ingest:
            st.info("Finish Step 1 first.")

    # Step 3
    with st.container(border=True):
        st.subheader(f"Step 3 — Train the model   {status_badge(done_train, ready=done_features)}")
        st.write(
            "The model studies the historical patterns — how threats rise and fall, cluster together, "
            "and ripple through the economy. **This is the slow step.** Use *Quick preview* in the "
            "sidebar the first time to see it work end-to-end."
        )
        cmd3 = [PY, "scripts/train.py"]
        if quick:
            cmd3 += ["--num-warmup", "150", "--num-samples", "150", "--num-chains", "1"]
        if enhanced:
            cmd3 += ["--enhanced-mode"]
        if st.button("Train the model", key="b3", disabled=not done_features):
            if run_step(cmd3, "Training the model"):
                st.rerun()
        if not done_features:
            st.info("Finish Step 2 first.")

    # Step 4
    with st.container(border=True):
        st.subheader(f"Step 4 — Check the accuracy   {status_badge(done_eval, ready=done_train)}")
        st.write(
            "Tests how well the model predicts months it was not shown, and compares it against "
            "simpler forecasting methods. This tells you how much to trust the forecast."
        )
        if st.button("Check the accuracy", key="b4", disabled=not done_train):
            if run_step([PY, "scripts/evaluate.py"], "Checking accuracy"):
                st.rerun()
        if not done_train:
            st.info("Finish Step 3 first.")

    # Step 5
    with st.container(border=True):
        st.subheader(f"Step 5 — Forecast the future   {status_badge(done_forecast, ready=done_train)}")
        st.write(
            "Produces the 12-month-ahead outlook: expected threat activity by category and the "
            "range of possible economic losses. Results appear in the **🔮 The forecast** tab."
        )
        cmd5 = [PY, "scripts/forecast.py", "--horizon", "12"]
        if enhanced:
            cmd5 += ["--enhanced-mode"]
        if st.button("Create the forecast", key="b5", disabled=not done_train, type="primary"):
            if run_step(cmd5, "Creating the forecast"):
                st.rerun()
        if not done_train:
            st.info("Finish Step 3 first.")

    # Step 6 (optional — only meaningful once the forecast window is in the past)
    with st.container(border=True):
        st.subheader(f"Step 6 — Check the model against reality   {status_badge(done_backtest, ready=done_train)}")
        st.write(
            "Replays history the way the model actually works: each month it predicts "
            "the next month **before** seeing it, is scored against what really "
            "happened, then updates its beliefs with that month's actual data — from "
            "the first month of the panel onward. See the **🔮 The forecast** tab "
            "for the comparison."
        )
        if st.button("Run the sequential backtest", key="b6", disabled=not done_train):
            if run_step([PY, "scripts/backtest.py"], "Running the sequential backtest"):
                st.rerun()
        if not done_train:
            st.info("Finish Step 3 first.")

# TAB 2 — Your data
with tab_data:
    if not done_ingest:
        st.info("No data yet. Run **Step 1 — Collect the data** first.")
    else:
        from cassandra_threatcast.viz.interactive import (
            activity_line, topic_heatmap, sector_bar,
        )

        N = np.load(PROCESSED / "N.npy")            # (K, T)
        meta = json.loads((PROCESSED / "panel_meta.json").read_text(encoding="utf-8"))
        dates = meta.get("dates") or meta.get("metadata", {}).get("dates", [])

        c1, c2, c3 = st.columns(3)
        c1.metric("Vulnerabilities collected", f"{int(np.nansum(N)):,}")
        c2.metric("Threat categories", f"{N.shape[0]}")
        c3.metric("Months of history", f"{N.shape[1]}")

        chart_dates = dates[: N.shape[1]] if dates else [str(i) for i in range(N.shape[1])]

        st.markdown("#### Monthly vulnerability activity")
        st.caption("Total new software vulnerabilities recorded each month. Hover for values; drag to zoom.")
        st.plotly_chart(activity_line(np.nansum(N, axis=0), chart_dates),
                        use_container_width=True)

        st.markdown("#### Activity by threat category")
        st.caption("Darker means more vulnerabilities that month for that category. Hover any cell.")
        topic_labels = load_topic_labels(str(PROCESSED), N.shape[0])
        st.plotly_chart(topic_heatmap(N, topic_labels, chart_dates),
                        use_container_width=True)

        D = PROCESSED / "D.npy"
        if D.exists() and np.nansum(np.load(D)) > 0:
            st.markdown("#### Disclosed company incidents by sector")
            st.caption("Hover a bar for the exact count, or a sector name below for what it covers.")
            Dd = np.load(D)
            sector_names_chart = SECTOR_LABELS[: Dd.shape[0]]
            st.plotly_chart(sector_bar(np.nansum(Dd, axis=1), sector_names_chart),
                            use_container_width=True)
            sector_glossary_expander()

# TAB 3 — The forecast
with tab_forecast:
    if not done_forecast:
        st.info("No forecast yet. Finish **Step 5 — Forecast the future** to see results here.")
    else:
        forecast_dir = RESULTS / "forecasts"

        summary_path = forecast_dir / "summary.txt"
        if summary_path.exists():
            st.markdown("#### What this means")
            st.info(summary_path.read_text(encoding="utf-8"))
        else:
            st.caption(
                "Tip: set `DEEPSEEK_API_KEY` in your `.env` file and re-run "
                "**Step 5 — Forecast the future** to get an AI-generated "
                "plain-English summary here."
            )

        st.markdown("#### Threat forecast (next months)")
        st.caption(
            "Predicted new vulnerabilities **per month**, by category, with an "
            "uncertainty range. `topic_label` is an automatically-derived "
            "category name (AI-assisted where available) — categories the "
            "model couldn't confidently name are labeled 'Miscellaneous "
            "Vulnerabilities' rather than guessing."
        )
        q = pd.read_csv(forecast_dir / "forecast_quantiles.csv")
        st.dataframe(q, use_container_width=True, height=260)

        loss_path = forecast_dir / "loss_distribution.csv"
        if loss_path.exists():
            st.markdown("#### Possible economic losses")
            loss_preview = pd.read_csv(loss_path)
            horizon_note = ""
            if "horizon_months" in loss_preview.columns and len(loss_preview):
                horizon_note = f" over the next {int(loss_preview['horizon_months'].iloc[0])} months"
            st.caption(
                f"All figures are in **US dollars, totaled{horizon_note}** "
                "(not a monthly rate) — the model's range of plausible "
                "economy-wide losses from cyber incidents. Hover a sector "
                "name for what it covers."
            )
            st.markdown(_sector_table_html(loss_preview), unsafe_allow_html=True)
            sector_glossary_expander()

        sys_path = forecast_dir / "systemic_probs.csv"
        if sys_path.exists():
            st.markdown("#### Systemic-event probabilities")
            st.caption(
                "The model's probability that total economy-wide losses over "
                "the forecast horizon exceed each threshold."
            )
            sp = pd.read_csv(sys_path)
            sp["threshold"] = sp["threshold_usd"].map(lambda v: f"${v/1e9:,.0f}B")
            sp["probability"] = sp["prob_exceed"].map(lambda p: f"{p:.1%}")
            st.dataframe(sp[["threshold", "probability"]],
                         use_container_width=True, hide_index=True)

        expo_path = forecast_dir / "sector_exposure.csv"
        if expo_path.exists():
            st.markdown("#### Ranked sector exposure (next 12 months)")
            st.caption(
                "Exposure index = expected fraction of sector output "
                "disrupted; loss contribution = the sector's share of total "
                "expected losses; dominant topics = the threat categories "
                "driving that sector's exposure."
            )
            st.dataframe(pd.read_csv(expo_path),
                         use_container_width=True, hide_index=True)

        fig_dir = forecast_dir / "figures"
        if fig_dir.exists():
            # Per-topic fan charts are intentionally NOT shown here: the
            fig_jsons = [f for f in sorted(fig_dir.glob("*.json"))
                         if not f.stem.startswith("fan_chart_topic_")]
            if fig_jsons:
                import plotly.io as pio
                st.markdown("#### Charts")
                st.caption(
                    "The predictive loss distribution and the regime "
                    "early-warning signal. Per-topic predicted-vs-actual "
                    "charts are in the backtest section below. Hover for "
                    "values; drag to zoom."
                )
                for fj in fig_jsons:
                    st.plotly_chart(
                        pio.from_json(fj.read_text(encoding="utf-8")),
                        use_container_width=True,
                    )
            else:
                imgs = sorted(fig_dir.glob("*.png"))
                if imgs:
                    st.markdown("#### Charts")
                    for img in imgs:
                        st.image(str(img), caption=img.stem.replace("_", " "))

        # --- Model vs. baselines (rolling-origin evaluation) -------------
        eval_scores_path = RESULTS / "evaluation" / "scores.csv"
        if eval_scores_path.exists():
            st.divider()
            st.markdown("### 🏁 How does it compare to the baselines?")
            st.caption(
                "Rolling-origin evaluation against five baselines: random "
                "forest (the published standard), ARIMA, exponential "
                "smoothing, seasonal naïve, and univariate Bayesian "
                "structural time series. Lower is better on every metric."
            )
            from cassandra_threatcast.viz.interactive import baseline_lines
            eval_scores = pd.read_csv(eval_scores_path)
            metric_options = [m for m in ["CRPS", "LogS", "MAE", "RMSE"]
                              if m in eval_scores["metric"].unique()]
            metric_pick = st.selectbox("Metric", metric_options, index=0,
                                       key="baseline_metric")
            st.plotly_chart(baseline_lines(eval_scores, metric_pick),
                            use_container_width=True)

            dm_path = RESULTS / "evaluation" / "dm_test.csv"
            if dm_path.exists():
                st.caption(
                    "Diebold–Mariano tests on CRPS differentials vs. the "
                    "proposed model — 'worse' means the baseline is "
                    "significantly less accurate."
                )
                st.dataframe(pd.read_csv(dm_path),
                             use_container_width=True, hide_index=True)

        # --- Backtest: how did this forecast actually do? ---------------
        backtest_dir = RESULTS / "backtest"
        if done_backtest:
            st.divider()
            st.markdown("### 📏 How did the forecast do?")
            st.caption(
                "Sequential 1-step-ahead check: each month the model predicted the "
                "next month **before** seeing it, was scored, then updated its "
                "beliefs with the actual data — from the first month of the panel "
                "onward."
            )

            bt_summary_path = backtest_dir / "backtest_summary.txt"
            if bt_summary_path.exists():
                st.info(bt_summary_path.read_text(encoding="utf-8"))

            threat_cmp = pd.read_csv(backtest_dir / "threat_comparison.csv")
            if "period" in threat_cmp.columns:
                cov_src = threat_cmp[threat_cmp["period"] == "test"]
            else:
                cov_src = threat_cmp
            coverage = cov_src["within_90pct_interval"].mean()
            st.metric(
                "90% interval coverage (test window)",
                f"{coverage:.0%}",
                help="Share of topic-months where the actual count fell inside the "
                     "model's stated 90% range. Close to 90% means the model's "
                     "uncertainty range was well-calibrated; much lower means it "
                     "was overconfident, much higher means underconfident.",
            )
            st.markdown("#### Threat forecast: predicted vs. actual, by month")
            st.dataframe(threat_cmp, use_container_width=True, height=260)

            econ_bt_path = RESULTS / "calibration" / "economic_backtest.csv"
            if econ_bt_path.exists():
                st.markdown("#### Economic layer vs. documented events")
                st.caption(
                    "For each reference event, the model's pre-event 90% "
                    "predictive loss interval against the documented loss "
                    "range from public post-incident analyses."
                )
                eb = pd.read_csv(econ_bt_path)
                for col in ["documented_low_usd", "documented_high_usd",
                            "predictive_q05_usd", "predictive_q50_usd",
                            "predictive_q95_usd"]:
                    if col in eb.columns:
                        eb[col.replace("_usd", "")] = eb[col].map(
                            lambda v: f"${v/1e9:,.2f}B")
                        eb = eb.drop(columns=[col])
                st.dataframe(eb, use_container_width=True, hide_index=True)

            incident_cmp_path = backtest_dir / "incident_comparison.csv"
            if incident_cmp_path.exists():
                incident_cmp = pd.read_csv(incident_cmp_path)
                st.markdown("#### Disclosed incidents forecast: predicted vs. actual, by sector")
                st.caption(
                    "The dollar loss forecast has no routinely-collected actual-loss "
                    "series to check against, so this — disclosed incident counts, "
                    "one of the loss model's real inputs — is the closest available "
                    "check on it. Pick sectors below; blue is the model's "
                    "1-month-ahead prediction, black is what happened."
                )
                from cassandra_threatcast.viz.interactive import incident_sector_chart
                all_sectors = sorted(incident_cmp["sector"].unique())
                # Default to the sectors where something actually happened.
                active_sectors = sorted(
                    incident_cmp.loc[incident_cmp["actual_incidents"] > 0, "sector"].unique())
                chosen = st.multiselect(
                    "Sectors", all_sectors,
                    default=active_sectors or all_sectors[:3],
                    key="incident_sectors")
                for sec in chosen:
                    st.plotly_chart(
                        incident_sector_chart(incident_cmp[incident_cmp["sector"] == sec], sec),
                        use_container_width=True)
                with st.expander("Full table (all sectors, all months)"):
                    st.markdown(_sector_table_html(incident_cmp, name_col="sector"), unsafe_allow_html=True)
                sector_glossary_expander()

            bt_fig_dir = backtest_dir / "figures"
            if bt_fig_dir.exists():
                fig_jsons = sorted(bt_fig_dir.glob("backtest_topic_*.json"))
                if fig_jsons:
                    import plotly.io as pio
                    st.markdown("#### Predicted vs. actual, by category")
                    st.caption(
                        "Blue: the model's 1-month-ahead prediction (median line, "
                        "50%/90% credible bands). Black: what actually happened. "
                        "Drag the slider below each chart to zoom; hover for values."
                    )
                    for fj in fig_jsons:
                        st.plotly_chart(
                            pio.from_json(fj.read_text(encoding="utf-8")),
                            use_container_width=True,
                        )
                else:
                    bt_imgs = sorted(bt_fig_dir.glob("*.png"))
                    if bt_imgs:
                        st.markdown("#### Forecast vs. actual, by category")
                        for img in bt_imgs:
                            st.image(str(img), caption=img.stem.replace("_", " "))
        elif done_forecast:
            st.divider()
            st.caption(
                "Run **Step 6 — Check the model against reality** to replay history "
                "with sequential 1-step-ahead predictions and see how the model did here."
            )

# TAB 4 — Kernel lab (test theory: learned moving-window covariance)
with tab_kernel:
    st.markdown("### 🧪 Learned temporal kernel — a test theory")
    st.caption(
        "**Experimental — nothing here touches the main model or the other "
        "tabs.** The main model assumes nearby months predict each other via "
        "its AR factor dynamics. This experiment makes that assumption "
        "explicit and learnable: a square window of learnable width slides "
        "over the series, and the model learns (1) how far apart two months "
        "can be and still co-move — the window half-width h, (2) how much "
        "that local covariance matters relative to the historical trends — "
        "the amplitude σ_g, and (3) which threat topics feel it — the "
        "loadings a_k. If local covariance adds nothing beyond the existing "
        "dynamics, σ_g shrinks toward zero and the kernel self-ablates."
    )

    kernel_params_path = RESULTS / "kernel" / "kernel_params.json"
    kernel_row_exists = False
    ablation_path = RESULTS / "ablation" / "ablation.csv"
    if ablation_path.exists():
        _ab = pd.read_csv(ablation_path)
        kernel_row_exists = bool((_ab["variant"] == "kernel").any())

    if not kernel_params_path.exists():
        if kernel_row_exists:
            st.info(
                "The kernel variant is trained. Generate this tab's report with:\n\n"
                "`python scripts/kernel_report.py`"
            )
        else:
            st.info(
                "The kernel variant hasn't been trained yet. Run:\n\n"
                "`python scripts/ablation.py --variants kernel`\n\n"
                "then `python scripts/kernel_report.py`. The main model is "
                "unaffected either way — the kernel is opt-in via the "
                "`kernel:` block in configs/default.yaml."
            )
    else:
        kp = json.loads(kernel_params_path.read_text(encoding="utf-8"))
        hw, sg = kp["halfwidth_months"], kp["sigma_g"]
        comp = kp.get("comparison", {})

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Learned half-width",
            f"{hw['mean']:.0f} months",
            help=f"90% CI {hw['q05']:.0f}–{hw['q95']:.0f} months. How far "
                 "apart two months can be and still co-move — the width of "
                 "the moving square window, learned from the data.",
        )
        c2.metric(
            "Amplitude σ_g",
            f"{sg['mean']:.3f}",
            help=f"90% CI {sg['q05']:.3f}–{sg['q95']:.3f} (log-intensity "
                 "units). How much local covariance matters beyond the AR "
                 "factors, regimes, and effort trend. Near zero = the "
                 "existing dynamics already capture it.",
        )
        if comp.get("kernel") and comp.get("full"):
            delta = 100.0 * (comp["kernel"]["CRPS_h1"] - comp["full"]["CRPS_h1"]) / comp["full"]["CRPS_h1"]
            c3.metric(
                "CRPS h=1 vs. full model",
                f"{comp['kernel']['CRPS_h1']:.1f}",
                delta=f"{delta:+.1f}% vs {comp['full']['CRPS_h1']:.1f}",
                delta_color="inverse",
                help="Sequential 1-step-ahead CRPS on the 2023–2024 test "
                     "window, settings-matched against the full model "
                     "without the kernel. Negative = kernel improves accuracy.",
            )

        if comp.get("kernel") and comp.get("full"):
            comp_df = pd.DataFrame([
                {"model": "Full (no kernel)", **comp["full"]},
                {"model": "Full + kernel", **comp["kernel"]},
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        kfig_dir = RESULTS / "kernel" / "figures"
        if kfig_dir.exists():
            import plotly.io as pio
            for stem, blurb in [
                ("kernel_covariance",
                 "The learned covariance between months as a function of how "
                 "far apart they are — the quantified version of 'close "
                 "points affect each other'."),
                ("kernel_g_t",
                 "The kernel factor through time: local co-movement the "
                 "window picks up beyond the global trends."),
                ("kernel_loadings",
                 "Per-topic loadings: which threat categories participate in "
                 "the local covariance."),
            ]:
                fj = kfig_dir / f"{stem}.json"
                if fj.exists():
                    st.plotly_chart(pio.from_json(fj.read_text(encoding="utf-8")),
                                    use_container_width=True)
                    st.caption(blurb)
