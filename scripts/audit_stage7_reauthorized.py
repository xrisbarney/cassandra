"""Recompute complete-fit diagnostics and verify every chain checkpoint exactly."""

import json
from contextlib import closing
from pathlib import Path

import arviz as az
import numpy as np

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.models.bayesian_controls import diagnostic_report
from cyber_threat_modelling.provenance import file_hash

root = Path(__file__).resolve().parents[1]
base = root / "artifacts/stage7/reauthorized-20261023"
activation_path = root / "docs/decisions/STAGE7_DIAGNOSTIC_REAUTHORIZATION_20261023.json"
activation = json.loads(activation_path.read_text())
assert {p.name for p in (base / "fits").iterdir()} == set(activation["cases"])
for name, digest in activation["files_sha256"].items():
    assert file_hash(root / name) == digest, name
rows = []
for case in activation["cases"]:
    folder = base / "fits" / case
    supervisor_path = base / (case + "-supervisor.json")
    supervisor = json.loads(supervisor_path.read_text())
    run = json.loads((folder / "run.json").read_text())
    assert run["activation_sha256"] == file_hash(activation_path)
    assert run["case"] == case and run["attempt_count"] == 1 and run["chains"] == 4
    assert run["warmup"] == run["draws"] == 2000
    assert run["target_accept"] == 0.99 and run["max_treedepth"] == 15
    assert run["sampler_seed"] == 7103 and run["initialization_seed"] == 7301
    assert run["source_control"] == {
        "dirty": False,
        "revision": "d67871e9c1f1eb4c08c474d3015b68e231babfa7",
        "status": "available",
    }
    assert run["fixed_phi"] == (None if case == "joint_free" else [0.65])
    assert run["fresh_attempt_not_continuation"]
    assert run["numba_opt"] == "3" and run["numerical_threads"] == 1
    assert supervisor["case"] == case and supervisor["attempt_count"] == 1
    assert supervisor["timeout_seconds"] == 5400 and not supervisor["retry_permitted"]
    assert supervisor["global_deadline_utc"] == activation["global_deadline_utc"]
    assert not run["stage8_enabled"] and not run["held_out_scoring_enabled"]
    markers = sorted(folder.glob("chain_*.json"))
    markers = [p for p in markers if p.stem.split("_")[-1].isdigit()]
    checkpoints = []
    for marker in markers:
        saved = json.loads(marker.read_text())
        chain = saved["chain"]
        assert marker.name == f"chain_{chain}.json" and chain in range(4)
        assert saved["warmup_excluded"] == saved["retained_draws"] == 2000
        assert file_hash(folder / f"chain_{chain}.nc") == saved["posterior_sha256"]
        with closing(az.from_netcdf(folder / f"chain_{chain}.nc")) as idata:
            assert idata.posterior.sizes["chain"] == 1 and idata.posterior.sizes["draw"] == 2000
            np.testing.assert_array_equal(idata.posterior.chain, [chain])
        checkpoints.append({"chain": chain, "marker_sha256": file_hash(marker), **saved})
    row = {
        "case": case,
        "run": run,
        "supervisor": supervisor,
        "supervisor_sha256": file_hash(supervisor_path),
        "log_sha256": file_hash(base / (case + "-fit.log")),
        "checkpoints": checkpoints,
        "diagnostics_independently_recomputed": False,
        "checkpoint_arrays_equal_final_posterior": False,
    }
    if (folder / "result.json").exists():
        result = json.loads((folder / "result.json").read_text())
        assert set(p.name for p in folder.iterdir()) == set(result["artifacts"]) | {"result.json"}
        for name, digest in result["artifacts"].items():
            assert file_hash(folder / name) == digest, name
        if result["diagnostics"] is not None:
            assert [p["chain"] for p in checkpoints] == list(range(4))
            checkpoint_only_empty_warning_columns = []
            with closing(az.from_netcdf(folder / "posterior.nc")) as idata:
                assert idata.posterior.sizes["chain"] == 4
                assert idata.posterior.sizes["draw"] == 2000
                computed, table = diagnostic_report(idata, 15)
                for chain in range(4):
                    with closing(az.from_netcdf(folder / f"chain_{chain}.nc")) as saved:
                        for group in ("posterior", "sample_stats"):
                            extra = set(getattr(saved, group)) - set(getattr(idata, group))
                            assert not (set(getattr(idata, group)) - set(getattr(saved, group)))
                            if extra:
                                # pm.sample drops this column by default after conversion;
                                # direct callback conversion preserves its empty values.
                                assert group == "sample_stats" and extra == {"warning"}
                                assert np.isnan(saved.sample_stats.warning.values).all()
                                checkpoint_only_empty_warning_columns.append(chain)
                            for name in getattr(idata, group):
                                np.testing.assert_array_equal(
                                    getattr(saved, group)[name],
                                    getattr(idata, group)[name].sel(chain=[chain]),
                                )
            for key, value in result["diagnostics"].items():
                if value is None:
                    assert not np.isfinite(computed[key])
                else:
                    np.testing.assert_allclose(computed[key], value, rtol=1e-10, atol=1e-10)
            row["diagnostics_independently_recomputed"] = True
            row["checkpoint_arrays_equal_final_posterior"] = True
            row["checkpoint_only_empty_warning_columns"] = checkpoint_only_empty_warning_columns
            # Preserve non-finite failures as JSON nulls, never silently drop rows.
            table = table.astype(object).where(np.isfinite(table), None)
            row["worst_rhat"] = (
                table.sort_values("r_hat", ascending=False).head(12).to_dict("index")
            )
            row["lowest_bulk_ess"] = table.sort_values("ess_bulk").head(12).to_dict("index")
            row["lowest_tail_ess"] = table.sort_values("ess_tail").head(12).to_dict("index")
            admitted = computed["accepted"] and not result["unresolved_warnings"]
            assert (result["status"] == "accepted") == admitted
        else:
            assert result["status"] != "accepted"
        row.update(
            result=result, status=result["status"], result_sha256=file_hash(folder / "result.json")
        )
    else:
        assert supervisor["timed_out"] or supervisor["returncode"] != 0
        row["status"] = "time_budget_exhausted" if supervisor["timed_out"] else "execution_failure"
        row["partial_artifacts"] = {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()}
    rows.append(row)
output = root / "docs/simulations/stage7-reauthorized-20261023"
output.mkdir(parents=True, exist_ok=True)
assert not (output / "diagnostics.json").exists()
write_json(
    output / "diagnostics.json",
    {
        "cases": rows,
        "attempts": 2,
        "source_sha256": file_hash(Path(__file__)),
        "activation_sha256": file_hash(activation_path),
        "stage8_enabled": False,
        "held_out_scoring_enabled": False,
        "exploration_review_required": True,
        "fixed_persistence_fit_is_conditional_only": True,
    },
)
print(
    json.dumps(
        [
            {
                "case": r["case"],
                "status": r["status"],
                "completed_checkpoints": len(r["checkpoints"]),
            }
            for r in rows
        ]
    )
)
