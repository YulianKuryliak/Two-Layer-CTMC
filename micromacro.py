from devtools.batch import run_micromacro_batch_from_config
from devtools.config import load_config
import time


if __name__ == "__main__":
    cfg, _ = load_config()
    net_cfg = cfg["network"]
    virus_cfg = cfg["virus"]
    sim_common = cfg["simulation"]
    sim_cfg = cfg["micromacro"]
    initial_node = sim_common.get("initial_node")
    community_size = int(net_cfg["community_size"])

    t_sim_start = time.perf_counter()
    run_uids = run_micromacro_batch_from_config(variant="micromacro")
    t_sim_done = time.perf_counter()
    runs = len(run_uids)
    output_dir = sim_cfg["out_folder"]
    base_seed = int(sim_common["base_seed"])
    tau_micro = float(sim_cfg["tau_micro"])
    T_end = float(sim_common["T_end"])
    macro_T = float(sim_cfg["macro_T"])
    avg_sim_time = (t_sim_done - t_sim_start) / runs if runs > 0 else 0.0
    print(f"Average simulation time per run: {avg_sim_time:.3f}s")

    # Optional lazy import: avoids DB stack import cost before simulation starts.
    try:
        from sim_db import log_run  # noqa
    except Exception:
        def log_run(**kwargs):  # type: ignore
            pass

    log_run(
        simulator="MicroMacro",
        sim_version="1.0.3",
        network_params=net_cfg,
        virus_params=virus_cfg,
        sim_params={
            "n_runs": runs,
            "base_seed": base_seed,
            "initial_node": initial_node,
            "T_end": T_end,
            "tau_micro": tau_micro,
            "macro_T": macro_T,
            "out_folder": str(output_dir),
            "k": int(net_cfg["communities"]),
            "size": community_size,
            "inter_links": int(net_cfg["inter_links"]),
        },
        output_path=str(output_dir),
    )

    if run_uids:
        print(f"DB v1 ingest completed for {len(run_uids)} MicroMacro runs.")
