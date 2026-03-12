from devtools.batch import run_micro_batch_from_config
from devtools.config import load_config

# --- optional, no-op if missing
try:
    from sim_db import log_run  # noqa
except Exception:
    def log_run(**kwargs):  # type: ignore
        pass    


if __name__ == "__main__":
    cfg, _ = load_config()
    net_cfg = cfg["network"]
    virus_cfg = cfg["virus"]
    sim_cfg = cfg["micro"]
    sim_common = cfg["simulation"]
    initial_node = sim_common.get("initial_node")
    run_uids = run_micro_batch_from_config()
    n_sims = len(run_uids)
    out_folder = sim_cfg["out_folder"]
    T_END = float(sim_common["T_end"])
    DT_OUT = float(sim_cfg["dt_out"])

    print(f"Done. Stored {n_sims} Micro runs into DB. CSV export path (if enabled): {out_folder}")

    log_run(
        simulator="Micro",
        sim_version="1.0.3",
        network_params=net_cfg,
        virus_params=virus_cfg,
        sim_params={
            "T_end": T_END,
            "dt_out": DT_OUT,
            "n_runs": n_sims,
            "base_seed": int(sim_common["base_seed"]),
            "initial_node": initial_node,
            "out_folder": str(out_folder),
        },
        output_path=str(out_folder),
    )

    if run_uids:
        print(f"DB v1 ingest completed for {len(run_uids)} Micro runs.")


