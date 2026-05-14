import argparse

from devtools.config import load_config
from sim_db import backfill_from_outputs, init_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill v1 DB tables from existing simulation outputs.")
    parser.add_argument("--limit", type=int, default=100, help="How many CSV runs to import per dataset.")
    parser.add_argument(
        "--db-path",
        default="simulations.db",
        help="SQLite DB path (default: simulations.db).",
    )
    args = parser.parse_args()

    cfg, _ = load_config()
    init_db(args.db_path)

    micro_out = cfg["micro"]["out_folder"]
    micromacro_out = cfg["micromacro"]["out_folder"]

    micro_uids = backfill_from_outputs(
        simulator="Micro",
        sim_version="1.0.3",
        config=cfg,
        out_folder=micro_out,
        limit=args.limit,
        config_source="assumed_backfill",
        db_path=args.db_path,
    )
    print(f"Backfilled Micro runs: {len(micro_uids)}")

    mm_uids = backfill_from_outputs(
        simulator="MicroMacro",
        sim_version="1.0.3",
        config=cfg,
        out_folder=micromacro_out,
        limit=args.limit,
        config_source="assumed_backfill",
        db_path=args.db_path,
    )
    print(f"Backfilled MicroMacro runs: {len(mm_uids)}")
    print(f"Total backfilled runs: {len(micro_uids) + len(mm_uids)}")


if __name__ == "__main__":
    main()
