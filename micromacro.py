from devtools.batch import run_micromacro_batch_from_config
from devtools.config import load_config
from sim_db import log_run


if __name__ == "__main__":
    cfg, _ = load_config()
    net_cfg = cfg["network"]
    virus_cfg = cfg["virus"]
    sim_common = cfg["simulation"]
    sim_cfg = cfg["micromacro"]
    initial_node = sim_common.get("initial_node")

    paths = run_micromacro_batch_from_config(variant="micromacro")
    runs = len(paths)
    output_dir = paths[0].parent if paths else sim_cfg["out_folder"]
    base_seed = int(sim_common["base_seed"])
    tau_micro = float(sim_cfg["tau_micro"])
    T_end = float(sim_common["T_end"])
    macro_T = float(sim_cfg["macro_T"])

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
            "size": int(net_cfg["community_size"]),
            "inter_links": int(net_cfg["inter_links"]),
        },
        output_path=str(output_dir),
    )
