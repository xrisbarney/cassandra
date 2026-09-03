#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

"""
Sparse-topic population run (paper §4.1 "Evaluation set" + Appendix).

The paper's headline tables use the ACTIVE-TOPIC set (topics with >= 50 CVE
assignments and >= 24 non-zero months in the initial training window) and
additionally report Full and BSTS-U on the COMPLETE topic population in an
appendix -- the hierarchical-shrinkage claim is exercised on the long tail
of sparse topics.  This script produces that population and both results:

  1. Fits a large-K topic model (MiniBatchNMF over TF-IDF, K_full topics)
     on the cached CVE corpus and builds the (K_full, T) monthly count panel.
  2. Derives the active-topic set from the initial training window.
  3. Trains the Full model on the complete population via VI -- the paper's
     sanctioned scalable alternative "when K is in the thousands" (§3).
     The E/B channels have no per-topic data at this granularity and enter
     as fully missing (NaN-masked); the exposure map is uniform.
  4. Scores Full (sequential one-step-ahead CRPS over the test window) on
     the full population, the active set, and the sparse complement, and
     runs BSTS-U per series with rolling 6-month refits for the appendix
     comparison.

Outputs (results/full_population/):
  topic_panel.npz         N_full (K_full, T), active mask, topic labels
  idata.pkl               VI posterior for the population model
  appendix_scores.csv     CRPS by model x topic-set
  meta.json               settings, counts, runtime notes

Usage:
    python scripts/full_population.py                     # K_full=1024
    python scripts/full_population.py --k-full 512 --skip-bsts
    python scripts/full_population.py --smoke             # tiny wiring check
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import yaml

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cassandra_threatcast.data import nvd
from cassandra_threatcast.evaluation.scoring import crps_ensemble
from cassandra_threatcast.evaluation.sequential import sequential_one_step_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sparse-topic population build + Full/BSTS-U appendix comparison.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/processed/")
    parser.add_argument("--cache-dir", default="data/cache/")
    parser.add_argument("--output-dir", default="results/full_population/")
    parser.add_argument("--k-full", type=int, default=1024,
                        help="Size of the full topic population. Default 1024.")
    parser.add_argument("--active-min-cves", type=int, default=50)
    parser.add_argument("--active-min-months", type=int, default=24)
    parser.add_argument("--test-start", default="2023-01")
    parser.add_argument("--test-end", default="2024-12")
    parser.add_argument("--vi-steps", type=int, default=4000)
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Posterior draws from the VI guide.")
    parser.add_argument("--bsts-warmup", type=int, default=150)
    parser.add_argument("--bsts-samples", type=int, default=150)
    parser.add_argument("--bsts-origin-step", type=int, default=6,
                        help="Months between BSTS-U refit origins in the test window.")
    parser.add_argument("--skip-bsts", action="store_true")
    parser.add_argument("--skip-train", action="store_true",
                        help="Reuse an existing results/full_population/idata.pkl.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="K_full=32, 300 VI steps, BSTS on 8 series -- wiring check only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    if args.smoke:
        args.k_full, args.vi_steps, args.num_samples = 32, 300, 50

    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    S = config["model"]["S"]

    # 1. Corpus -> large-K topic panel
    with open(os.path.join(args.data_dir, "panel_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    date_strs = meta.get("dates") or meta.get("metadata", {}).get("dates", [])
    periods = pd.PeriodIndex(date_strs, freq="M")
    T = len(periods)
    start_dt = str(periods[0].to_timestamp().date())
    end_dt = str((periods[-1].to_timestamp() + pd.offsets.MonthEnd(0)).date())

    panel_path = os.path.join(args.output_dir, "topic_panel.npz")
    if os.path.exists(panel_path):
        print(f"[1/4] Loading existing topic panel ({panel_path}) ...")
        npz = np.load(panel_path, allow_pickle=True)
        N_full, active, labels = npz["N_full"], npz["active"], list(npz["labels"])
        K_full = N_full.shape[0]
    else:
        print(f"[1/4] Building K={args.k_full} topic panel from the cached CVE corpus ...")
        cve_df = nvd.fetch_cves(start_dt, end_dt, args.cache_dir)
        print(f"      {len(cve_df):,} CVEs in cache window {start_dt}..{end_dt}")
        descriptions = cve_df["description"].fillna("").tolist()

        from cassandra_threatcast.features.topic_map import WikiTopicMapper
        mapper = WikiTopicMapper(n_topics=args.k_full)
        t0 = time.time()
        mapper.fit(descriptions)
        print(f"      Topic model fitted in {(time.time()-t0)/60:.1f} min")
        assign = mapper.assign_hard(descriptions)
        labels = mapper.get_topic_labels(n_display=3)

        months = pd.PeriodIndex(pd.to_datetime(cve_df["published_date"]), freq="M")
        month_idx = periods.get_indexer(months)
        valid = month_idx >= 0
        K_full = args.k_full
        N_full = np.zeros((K_full, T), dtype=np.int64)
        np.add.at(N_full, (assign[valid], month_idx[valid]), 1)

        # Active-topic set from the INITIAL TRAINING WINDOW only.
        train_mask = np.asarray(periods < pd.Period(args.test_start, freq="M"))
        N_train = N_full[:, train_mask]
        active = ((N_train.sum(axis=1) >= args.active_min_cves)
                  & ((N_train > 0).sum(axis=1) >= args.active_min_months))
        np.savez_compressed(panel_path, N_full=N_full, active=active,
                            labels=np.array(labels, dtype=object))
        print(f"      Panel saved -> {panel_path}")

    n_active = int(active.sum())
    print(f"      Population: {K_full} topics | active set: {n_active} "
          f"(>= {args.active_min_cves} CVEs & >= {args.active_min_months} "
          f"non-zero training months) | sparse tail: {K_full - n_active}")

    test_mask = np.asarray((periods >= pd.Period(args.test_start, freq="M"))
                           & (periods <= pd.Period(args.test_end, freq="M")))

    # 2. Train the Full model on the population via VI
    idata_path = os.path.join(args.output_dir, "idata.pkl")
    if args.skip_train and os.path.exists(idata_path):
        print("[2/4] Reusing existing population posterior ...")
        with open(idata_path, "rb") as fh:
            idata = pickle.load(fh)
    else:
        print(f"[2/4] Training Full model on K={K_full} via VI "
              f"({args.vi_steps} steps) -- the paper's scalable alternative ...")
        from cassandra_threatcast.model import full as full_module
        from cassandra_threatcast.inference import vi as vi_module
        import arviz as az

        e_t = np.load(os.path.join(args.data_dir, "e_t.npy"))
        D_st = np.load(os.path.join(args.data_dir, "D.npy")).astype(float)
        pop_config = json.loads(json.dumps(config))   # deep copy
        pop_config["model"]["K"] = int(K_full)

        data = {
            "N": N_full.astype(float),
            "E": None,          # no per-topic EPSS at this granularity (NaN-masked)
            "B": None,          # no per-topic severity marks (NaN-masked)
            "D": D_st,
            "M_skt": np.ones((S, K_full, T)) / S,
            "e_t": e_t,
        }
        t0 = time.time()
        # num_particles=1: at population scale each extra ELBO particle
        guide, params, losses = vi_module.train_vi(
            full_module.full_model, data, pop_config,
            num_steps=args.vi_steps, seed=args.seed, num_particles=1)
        print(f"      VI done in {(time.time()-t0)/60:.1f} min "
              f"(final ELBO {-losses[-1]:.1f})")
        samples = vi_module.vi_predictive_samples(
            guide, params, full_module.full_model, data,
            config=pop_config, n_samples=args.num_samples, seed=args.seed)
        posterior = {k: np.asarray(v)[np.newaxis, ...] for k, v in samples.items()}
        # arviz >= 1.0: from_dict takes {"group": {...}} rather than kwargs.
        idata = az.from_dict({"posterior": posterior})
        with open(idata_path, "wb") as fh:
            pickle.dump(idata, fh)
        print(f"      Posterior -> {idata_path}")

    # 3. Score Full (sequential one-step-ahead) on the three topic sets
    print("[3/4] Scoring Full model (sequential 1-step-ahead) ...")
    post = {k: np.asarray(v) for k, v in idata.posterior.items()}
    # Drop zero-size sites (numpyro.factor sites come back empty from
    post = {k: v.reshape((-1,) + v.shape[2:]) for k, v in post.items() if v.size > 0}
    e_t = np.load(os.path.join(args.data_dir, "e_t.npy"))
    M_uniform = np.ones((S, K_full, T)) / S

    out = sequential_one_step_predict(post, e_t, M_uniform, tail_months=0,
                                      enhanced=False, student_t_df=4.0,
                                      seed=args.seed)
    N_pred = out["N_pred"]                               # (n, K_full, T)
    obs = N_full.astype(float)

    def _crps_subset(mask_topics):
        if not np.any(mask_topics):
            return float("nan")
        o = obs[mask_topics][:, test_mask]
        s = np.moveaxis(N_pred[:, mask_topics][:, :, test_mask], 0, -1)
        return float(crps_ensemble(o, s.reshape(o.shape + (N_pred.shape[0],))).mean())

    rows = [
        {"model": "Full (VI)", "topic_set": "full_population",
         "n_topics": int(K_full), "CRPS_h1": _crps_subset(np.ones(K_full, bool))},
        {"model": "Full (VI)", "topic_set": "active_set",
         "n_topics": n_active, "CRPS_h1": _crps_subset(active)},
        {"model": "Full (VI)", "topic_set": "sparse_tail",
         "n_topics": int(K_full - n_active), "CRPS_h1": _crps_subset(~active)},
    ]
    for r in rows:
        print(f"      {r['topic_set']:16s} ({r['n_topics']:5d} topics)  "
              f"CRPS {r['CRPS_h1']:.3f}")

    # 4. BSTS-U on the complete population (appendix comparison)
    if not args.skip_bsts:
        print("[4/4] BSTS-U per series, rolling "
              f"{args.bsts_origin_step}-month refits over the test window ...")
        from cassandra_threatcast.evaluation.baselines import BstsUnivariate

        series_idx = np.arange(K_full)
        if args.smoke:
            series_idx = series_idx[:8]

        test_positions = np.where(test_mask)[0]
        origins = list(range(0, len(test_positions), args.bsts_origin_step))
        rng = np.random.default_rng(args.seed)
        n_sim = 400
        crps_cells = {"full_population": [], "active_set": [], "sparse_tail": []}

        t0 = time.time()
        for count, k in enumerate(series_idx):
            series_crps = []
            for o in origins:
                train_end = test_positions[o]
                horizon = min(args.bsts_origin_step, len(test_positions) - o)
                try:
                    bsts = BstsUnivariate(num_warmup=args.bsts_warmup,
                                          num_samples=args.bsts_samples
                                          ).fit(obs[k, :train_end], seed=args.seed + int(k))
                    pq = np.maximum(bsts.predict_quantiles(horizon, [0.1, 0.5, 0.9]), 0.0)
                except Exception:
                    continue
                med, iqr = pq[:, 1], pq[:, 2] - pq[:, 0]
                sigma = iqr / (2 * 0.6745)
                sims = np.maximum(
                    med[None, :] + rng.standard_normal((n_sim, horizon)) * sigma[None, :], 0.0)
                target = obs[k, test_positions[o]: test_positions[o] + horizon]
                series_crps.extend(crps_ensemble(target, np.moveaxis(sims, 0, -1)).tolist())
            if not series_crps:
                continue
            m = float(np.mean(series_crps))
            crps_cells["full_population"].append(m)
            crps_cells["active_set" if active[k] else "sparse_tail"].append(m)
            if count % 50 == 0:
                rate = (count + 1) / max(time.time() - t0, 1)
                eta_min = (len(series_idx) - count - 1) / max(rate, 1e-9) / 60
                print(f"      series {count + 1}/{len(series_idx)}  "
                      f"(~{eta_min:.0f} min remaining)", flush=True)

        for tset, vals in crps_cells.items():
            if vals:
                rows.append({"model": "BSTS-U", "topic_set": tset,
                             "n_topics": len(vals), "CRPS_h1": float(np.mean(vals))})
                print(f"      BSTS-U {tset:16s} CRPS {np.mean(vals):.3f}")
    else:
        print("[4/4] BSTS-U skipped (--skip-bsts).")

    scores_df = pd.DataFrame(rows)
    scores_path = os.path.join(args.output_dir, "appendix_scores.csv")
    scores_df.to_csv(scores_path, index=False)

    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "K_full": int(K_full), "n_active": n_active,
            "active_min_cves": args.active_min_cves,
            "active_min_months": args.active_min_months,
            "test_window": [args.test_start, args.test_end],
            "vi_steps": args.vi_steps, "num_samples": args.num_samples,
            "notes": "E/B channels fully missing at population granularity; "
                     "exposure map uniform. Full scored with sequential "
                     "one-step-ahead prediction; BSTS-U with rolling "
                     f"{args.bsts_origin_step}-month refits.",
            "runtime_min": round((time.time() - t_start) / 60, 1),
        }, fh, indent=2)
    print(f"\nAppendix scores -> {scores_path}")
    print(scores_df.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
