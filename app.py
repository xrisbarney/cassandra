"""
CASSANDRA — interactive dashboard.

A friendly, click-through interface to the cyber-threat forecasting pipeline.
Run it with:

    streamlit run app.py

Each step is a button. You do not need to touch the command line.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
PY = sys.executable

st.set_page_config(page_title="CASSANDRA — Cyber Threat Forecasting", page_icon="🔮", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
                text=True, bufsize=1,
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


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔮 CASSANDRA")
st.caption("Forecasting cyber threats and their economic impact — a guided, step-by-step tool.")

st.markdown(
    "This tool builds a forecast in five steps. Do them **in order, top to bottom**. "
    "Each step shows a green light once it has finished. You can watch the progress "
    "log as it runs — you don't need to understand it."
)

# ---------------------------------------------------------------------------
# Settings (sidebar)
# ---------------------------------------------------------------------------
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
done_eval = artifact_exists(RESULTS / "scores.csv")
done_forecast = artifact_exists(RESULTS / "forecast_quantiles.csv")

tab_run, tab_data, tab_forecast = st.tabs(["▶️ Run the steps", "📊 Your data", "🔮 The forecast"])

# ---------------------------------------------------------------------------
# TAB 1 — Run the steps
# ---------------------------------------------------------------------------
with tab_run:
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

# ---------------------------------------------------------------------------
# TAB 2 — Your data
# ---------------------------------------------------------------------------
with tab_data:
    if not done_ingest:
        st.info("No data yet. Run **Step 1 — Collect the data** first.")
    else:
        import matplotlib.pyplot as plt

        N = np.load(PROCESSED / "N.npy")            # (K, T)
        meta = json.loads((PROCESSED / "panel_meta.json").read_text())
        dates = meta.get("dates") or meta.get("metadata", {}).get("dates", [])

        c1, c2, c3 = st.columns(3)
        c1.metric("Vulnerabilities collected", f"{int(np.nansum(N)):,}")
        c2.metric("Threat categories", f"{N.shape[0]}")
        c3.metric("Months of history", f"{N.shape[1]}")

        st.markdown("#### Monthly vulnerability activity")
        st.caption("Total new software vulnerabilities recorded each month.")
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(np.nansum(N, axis=0), color="#7c3aed")
        ax.fill_between(range(N.shape[1]), np.nansum(N, axis=0), alpha=0.2, color="#7c3aed")
        ax.set_xlabel("Month"); ax.set_ylabel("New vulnerabilities")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

        st.markdown("#### Activity by threat category")
        st.caption("Darker means more vulnerabilities that month for that category.")
        fig2, ax2 = plt.subplots(figsize=(9, 3))
        im = ax2.imshow(N, aspect="auto", cmap="magma", interpolation="nearest")
        ax2.set_xlabel("Month"); ax2.set_ylabel("Threat category")
        fig2.colorbar(im, ax=ax2, label="Vulnerabilities")
        st.pyplot(fig2)

        D = PROCESSED / "D.npy"
        if D.exists() and np.nansum(np.load(D)) > 0:
            st.markdown("#### Disclosed company incidents by sector")
            Dd = np.load(D)
            fig3, ax3 = plt.subplots(figsize=(9, 2.6))
            ax3.bar(range(Dd.shape[0]), np.nansum(Dd, axis=1), color="#0d9488")
            ax3.set_xlabel("Economic sector"); ax3.set_ylabel("Incidents")
            ax3.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig3)

# ---------------------------------------------------------------------------
# TAB 3 — The forecast
# ---------------------------------------------------------------------------
with tab_forecast:
    if not done_forecast:
        st.info("No forecast yet. Finish **Step 5 — Forecast the future** to see results here.")
    else:
        import pandas as pd

        st.markdown("#### Threat forecast (next months)")
        st.caption("Predicted vulnerability activity per category, with an uncertainty range.")
        q = pd.read_csv(RESULTS / "forecast_quantiles.csv")
        st.dataframe(q, use_container_width=True, height=260)

        loss_path = RESULTS / "loss_distribution.csv"
        if loss_path.exists():
            st.markdown("#### Possible economic losses")
            st.caption("The model's range of plausible economy-wide losses from cyber incidents.")
            st.dataframe(pd.read_csv(loss_path), use_container_width=True, height=260)

        fig_dir = RESULTS / "figures"
        if fig_dir.exists():
            imgs = sorted(fig_dir.glob("*.png"))
            if imgs:
                st.markdown("#### Charts")
                for img in imgs:
                    st.image(str(img), caption=img.stem.replace("_", " "))
