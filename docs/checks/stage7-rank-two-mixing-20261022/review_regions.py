"""Describe declared persistence regions without admitting or modifying any fit."""

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr

from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    options = run["options"]
    if options["scenario"] != "fallback_zero" or options["seed"] not in (7101, 7102):
        raise ValueError("These markers were recorded for fallback seeds 7101 and 7102 only.")
    reference = Path('artifacts/stage7/rank-two-joint-v1/primary') / (
        f"fallback_zero-{options['seed']}")
    marker_record = Path(__file__).with_name("README.md")
    if options['seed'] == 7102:
        marker_record = Path(__file__).parent / 'fallback7102_attempt1' / 'README.md'
    if file_hash(args.run / "data.npz") != file_hash(reference / "data.npz"):
        raise ValueError("The observations differ from the diagnosed dataset.")
    if run["real_data_read"] or run["held_out_scoring_enabled"]:
        raise ValueError("This review is synthetic only.")
    attempts = []
    for path in sorted(args.run.glob("attempt_*.json")):
        attempt = json.loads(path.read_text(encoding="utf-8"))
        retry = attempt["retry"]
        posterior_path = args.run / f"posterior_{retry}.nc"
        entry = {"retry": retry, "attempt_sha256": file_hash(path),
                 "numerical_diagnostics": attempt["diagnostics"],
                 "attempt_accepted": attempt["accepted"], "regions": {}}
        if not posterior_path.exists():
            entry["review"] = "No complete saved posterior; no region inference."
            attempts.append(entry)
            continue
        with xr.open_dataset(posterior_path, group="posterior") as posterior:
            phi = posterior["f_phi"].values[:, :, 1].copy()
            mean = posterior["f_states"].values[:, :, 1:, 1].mean(axis=2)
        if phi.shape[0] != 4 or phi.shape[1] != attempt["draws"]:
            raise ValueError("Incomplete saved posterior.")
        entry["posterior_sha256"] = file_hash(posterior_path)
        masks = {"lower_persistence": phi < 0.8,
                 "high_positive_mean": (phi > 0.95) & (mean > 0.5),
                 "high_negative_mean": (phi > 0.95) & (mean < -0.5)}
        for name, mask in masks.items():
            values = mask.astype(float)
            counts = mask.sum(axis=1)
            transitions = (mask[:, 1:] != mask[:, :-1]).sum(axis=1)
            item = {"visits_per_chain": counts.tolist(),
                    "fraction_per_chain": values.mean(axis=1).tolist(),
                    "transitions_per_chain": transitions.tolist(),
                    "draws_per_chain": phi.shape[1],
                    "every_chain_visits": bool(np.all(counts > 0)),
                    "every_chain_enters_and_leaves": bool(np.all(transitions >= 2))}
            if np.all((counts > 0) & (counts < phi.shape[1])):
                item["indicator_rhat"] = float(az.rhat(values, method="rank"))
                item["indicator_bulk_ess"] = float(az.ess(values, method="bulk"))
            else:
                item["indicator_diagnostics"] = (
                    "Not reported: at least one chain has a constant indicator."
                )
            entry["regions"][name] = item
        attempts.append(entry)
    record = {"directory": args.run.as_posix(), "candidate": options["prior_candidate"],
              "source_sha256": file_hash(Path(__file__)),
              "marker_record": marker_record.as_posix(),
              "marker_record_sha256": file_hash(marker_record),
              "run_sha256": file_hash(args.run / "run.json"),
              "data_sha256": file_hash(args.run / "data.npz"),
              "attempts": attempts,
              "interpretation": "Visits describe exploration, not posterior region weights. "
              "No admission or recovery eligibility is changed by this review. Sensitivity "
              "priors have different posterior targets; compare their regions descriptively."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
