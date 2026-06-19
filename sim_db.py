from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path_like: str | Path) -> Path:
    normalized = str(path_like).replace("\\", "/")
    path = Path(normalized).expanduser()
    return path if path.is_absolute() else (BASE_DIR / path)


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _json_dumps(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def _json_loads_or_empty(payload: str | None) -> Dict[str, Any]:
    if not payload:
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {}


def _to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _canonical_macro_graph_type(value: Any) -> Optional[str]:
    raw = _to_str_or_none(value)
    if raw is None:
        return None
    v = raw.strip().lower()
    if v in {"complete", "clique", "fully_connected"}:
        return "complete"
    if v in {"chain", "path", "line"}:
        return "chain"
    if v in {"star", "hub_spoke", "hub-and-spoke", "hubspoke"}:
        return "star"
    return v


def _canonical_micro_graph_type(value: Any) -> Optional[str]:
    raw = _to_str_or_none(value)
    if raw is None:
        return None
    v = raw.strip().lower()
    if v in {"complete", "clique", "fully_connected"}:
        return "complete"
    if v in {"random", "erdos_renyi", "erdos-renyi", "er"}:
        return "random"
    if v in {"chain", "path", "line"}:
        return "chain"
    if v in {"random_connected", "connected_random", "connected_er", "connected_erdos_renyi"}:
        return "random_connected"
    if v in {"clique_leaves", "clique_with_leaves", "clique_plus_leaves", "full_with_leaves", "full_leaves"}:
        return "clique_leaves"
    return v


def _canonical_star_leaf_attachment(value: Any) -> Optional[str]:
    raw = _to_str_or_none(value)
    if raw is None:
        return None
    v = raw.strip().lower()
    if v in {"random", "uniform"}:
        return "random"
    if v in {"node0", "to_zero", "zero", "hub0"}:
        return "node0"
    if v in {"random_nonzero", "random_not_zero", "nonzero"}:
        return "random_nonzero"
    return v


def _extract_run_fields_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    net = config.get("network", {}) if isinstance(config.get("network", {}), dict) else {}
    virus = config.get("virus", {}) if isinstance(config.get("virus", {}), dict) else {}
    sim = config.get("simulation", {}) if isinstance(config.get("simulation", {}), dict) else {}
    micro = config.get("micro", {}) if isinstance(config.get("micro", {}), dict) else {}
    mm = config.get("micromacro", {}) if isinstance(config.get("micromacro", {}), dict) else {}

    communities = _to_int_or_none(net.get("communities"))
    community_size = _to_int_or_none(net.get("community_size"))
    macro_graph_type = _canonical_macro_graph_type(net.get("macro_graph_type"))
    micro_graph_type = _canonical_micro_graph_type(net.get("micro_graph_type"))
    star_leaf_attachment = _canonical_star_leaf_attachment(net.get("star_leaf_attachment"))
    inter_links = _to_int_or_none(net.get("inter_links"))
    leaf_count = _to_int_or_none(net.get("leaf_count"))
    leaf_degree = _to_int_or_none(net.get("leaf_degree"))

    topology_is_hub_leaf: Optional[int] = None
    if macro_graph_type is not None or micro_graph_type is not None:
        topology_is_hub_leaf = 1 if (macro_graph_type == "star" or micro_graph_type == "clique_leaves") else 0

    effective_micro_topology: Optional[str] = None
    effective_leaf_count: Optional[int] = None
    effective_leaf_degree: Optional[int] = None

    network_size_total: Optional[int] = None
    if macro_graph_type == "star":
        effective_micro_topology = "hub_clique_singleton_leaves"
        effective_leaf_count = 0
        effective_leaf_degree = 0
        if communities is not None and community_size is not None:
            network_size_total = community_size + max(0, communities - 1)
    elif micro_graph_type == "clique_leaves":
        effective_micro_topology = "clique_leaves"
        if leaf_count is not None:
            effective_leaf_count = max(0, leaf_count)
        if leaf_degree is not None:
            effective_leaf_degree = max(0, leaf_degree)
        if communities is not None and community_size is not None and leaf_count is not None:
            network_size_total = communities * (community_size + max(0, leaf_count))
    elif micro_graph_type is not None:
        effective_micro_topology = micro_graph_type
        effective_leaf_count = 0
        effective_leaf_degree = 0
        if communities is not None and community_size is not None:
            network_size_total = communities * community_size

    topology_key: Optional[str] = None
    if (
        macro_graph_type is not None
        and micro_graph_type is not None
        and communities is not None
        and community_size is not None
        and inter_links is not None
        and leaf_count is not None
        and leaf_degree is not None
        and star_leaf_attachment is not None
        and effective_micro_topology is not None
        and effective_leaf_count is not None
        and effective_leaf_degree is not None
    ):
        topology_key = (
            f"macro={macro_graph_type}|micro={micro_graph_type}|"
            f"k={communities}|n={community_size}|inter={inter_links}|"
            f"leaf_count={leaf_count}|leaf_degree={leaf_degree}|star_attach={star_leaf_attachment}|"
            f"effective_micro={effective_micro_topology}|effective_leaf_count={effective_leaf_count}|"
            f"effective_leaf_degree={effective_leaf_degree}"
        )

    return {
        "communities": communities,
        "community_size": community_size,
        "inter_links": inter_links,
        "seed": _to_int_or_none(net.get("seed")),
        "macro_graph_type": macro_graph_type,
        "micro_graph_type": micro_graph_type,
        "edge_prob": _to_float_or_none(net.get("edge_prob")),
        "leaf_count": leaf_count,
        "leaf_degree": leaf_degree,
        "star_leaf_attachment": star_leaf_attachment,
        "effective_micro_topology": effective_micro_topology,
        "effective_leaf_count": effective_leaf_count,
        "effective_leaf_degree": effective_leaf_degree,
        "beta": _to_float_or_none(virus.get("beta")),
        "gamma": _to_float_or_none(virus.get("gamma")),
        "model": _to_int_or_none(virus.get("model")),
        "n_runs": _to_int_or_none(sim.get("n_runs")),
        "base_seed": _to_int_or_none(sim.get("base_seed")),
        "T_end": _to_float_or_none(sim.get("T_end")),
        "initial_community": _to_int_or_none(sim.get("initial_community")),
        "initial_node": _to_int_or_none(sim.get("initial_node")),
        "dt_out": _to_float_or_none(micro.get("dt_out")),
        "tau_micro": _to_float_or_none(mm.get("tau_micro")),
        "macro_T": _to_float_or_none(mm.get("macro_T")),
        "print_infection_events": _to_int_or_none(mm.get("print_infection_events")),
        "export_infection_events_csv": _to_int_or_none(mm.get("export_infection_events_csv")),
        "verbose_steps": _to_int_or_none(mm.get("verbose_steps")),
        "micro_out_folder": _to_str_or_none(micro.get("out_folder")),
        "micromacro_out_folder": _to_str_or_none(mm.get("out_folder")),
        "network_size_total": network_size_total,
        "topology_key": topology_key,
        "topology_is_hub_leaf": topology_is_hub_leaf,
    }


def _default_run_uid(simulator: str, run_csv_path: Path) -> str:
    payload = f"{simulator}|{run_csv_path.resolve()}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()
    return f"{simulator}:{digest}"


def make_run_uid(
    *,
    simulator: str,
    config: Dict[str, Any],
    run_seed: Optional[int],
    created_at: str,
    run_index: Optional[int] = None,
) -> str:
    config_hash = hashlib.sha1(_json_dumps(config).encode("utf-8")).hexdigest()
    payload = f"{simulator}|{run_seed}|{created_at}|{config_hash}|{run_index}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()
    return f"{simulator}:{digest}"


def _sorted_run_csv_files(folder: Path) -> List[Path]:
    files = list(folder.glob("*.csv"))

    def key_fn(path: Path) -> Tuple[int, str]:
        stem = path.stem
        if stem.isdigit():
            return (0, f"{int(stem):012d}")
        return (1, stem)

    return sorted(files, key=key_fn)


def _load_metrics_rows(metrics_path: Path) -> List[Dict[str, Any]]:
    if metrics_path.suffix.lower() == ".json":
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        out: List[Dict[str, Any]] = []
        for row in payload.get("metrics", []):
            out.append(
                {
                    "community": _to_int_or_none(row.get("community")),
                    "t0": _to_float_or_none(row.get("t0")),
                    "t_bridge": _to_float_or_none(row.get("t_bridge")),
                    "t_export": _to_float_or_none(row.get("t_export")),
                }
            )
        return out

    out = []
    with open(metrics_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"community", "t0", "t_bridge", "t_export"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"Unexpected metrics columns in {metrics_path}: {reader.fieldnames}")
        for row in reader:
            out.append(
                {
                    "community": _to_int_or_none(row.get("community")),
                    "t0": _to_float_or_none(row.get("t0")),
                    "t_bridge": _to_float_or_none(row.get("t_bridge")),
                    "t_export": _to_float_or_none(row.get("t_export")),
                }
            )
    return out


def _load_events_rows(events_path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(events_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "time",
            "mode",
            "kind",
            "src_community",
            "dst_community",
            "src_node",
            "dst_node",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Unexpected event columns in {events_path}: {reader.fieldnames}")
        for row in reader:
            out.append(
                {
                    "time": _to_float_or_none(row.get("time")),
                    "mode": _to_str_or_none(row.get("mode")),
                    "kind": _to_str_or_none(row.get("kind")),
                    "src_community": _to_int_or_none(row.get("src_community")),
                    "dst_community": _to_int_or_none(row.get("dst_community")),
                    "src_node": _to_int_or_none(row.get("src_node")),
                    "dst_node": _to_int_or_none(row.get("dst_node")),
                }
            )
    return out


def _read_sir_payload(csv_path: Path) -> Tuple[bytes, int, str]:
    payload = csv_path.read_bytes()
    row_count = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for idx, _ in enumerate(f):
            row_count = idx
    # idx is 0 for header-only, so row_count is number of data rows
    sha = hashlib.sha256(payload).hexdigest()
    return payload, row_count, sha


def _sir_bytes_row_count_sha(sir_csv_bytes: bytes) -> Tuple[int, str]:
    row_count = 0
    with io.StringIO(sir_csv_bytes.decode("utf-8")) as f:
        for idx, _ in enumerate(f):
            row_count = idx
    sha = hashlib.sha256(sir_csv_bytes).hexdigest()
    return row_count, sha


def _serialize_df_for_artifact(df) -> Tuple[str, str, bytes, int, str]:
    if pd is None:
        raise ImportError("pandas is required for dataframe artifact storage")
    buf = io.BytesIO()
    df.to_pickle(buf, compression=None)
    payload = buf.getvalue()
    row_count = int(len(df.index))
    sha = hashlib.sha256(payload).hexdigest()
    return "pickle", "none", payload, row_count, sha


def _deserialize_artifact_to_csv_bytes(*, fmt: str, compression: str, payload_blob: bytes) -> bytes:
    if pd is None:
        raise ImportError("pandas is required to decode dataframe artifacts")
    if fmt != "pickle":
        raise ValueError(f"Unsupported artifact format: {fmt}")
    if compression not in {"none", ""}:
        raise ValueError(f"Unsupported artifact compression for pickle: {compression}")
    return pd.read_pickle(io.BytesIO(payload_blob)).to_csv(index=False).encode("utf-8")


def _deserialize_artifact_to_df(*, fmt: str, compression: str, payload_blob: bytes):
    if pd is None:
        raise ImportError("pandas is required to load dataframe artifacts")
    if fmt != "pickle":
        raise ValueError(f"Unsupported artifact format: {fmt}")
    if compression not in {"none", ""}:
        raise ValueError(f"Unsupported artifact compression for pickle: {compression}")
    return pd.read_pickle(io.BytesIO(payload_blob))


def _find_metrics_file(run_csv_path: Path) -> Optional[Path]:
    metrics_dir = run_csv_path.parent / "metrics"
    base = run_csv_path.stem
    json_path = metrics_dir / f"{base}_metrics.json"
    csv_path = metrics_dir / f"{base}_metrics.csv"
    if json_path.exists():
        return json_path
    if csv_path.exists():
        return csv_path
    return None


def _find_events_file(run_csv_path: Path) -> Optional[Path]:
    events_dir = run_csv_path.parent / "events"
    path = events_dir / f"{run_csv_path.stem}_infection_events.csv"
    return path if path.exists() else None


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _init_legacy_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sim_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            simulator TEXT NOT NULL,
            sim_version TEXT NOT NULL,
            network_params TEXT NOT NULL,
            virus_params   TEXT NOT NULL,
            macro_params   TEXT NOT NULL,
            sim_params     TEXT NOT NULL DEFAULT '{}',
            output_path    TEXT,
            runtime_seconds REAL,
            notes          TEXT
        );
        """
    )
    cur.execute("PRAGMA table_info(sim_runs);")
    cols = {row[1] for row in cur.fetchall()}
    if "sim_params" not in cols:
        cur.execute("ALTER TABLE sim_runs ADD COLUMN sim_params TEXT NOT NULL DEFAULT '{}';")


def _init_v1_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_uid TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            simulator TEXT NOT NULL,
            sim_version TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'completed',
            config_source TEXT NOT NULL DEFAULT 'runtime',
            config_json TEXT NOT NULL,
            config_hash TEXT,
            communities INTEGER,
            community_size INTEGER,
            inter_links INTEGER,
            seed INTEGER,
            run_seed INTEGER,
            macro_graph_type TEXT,
            micro_graph_type TEXT,
            edge_prob REAL,
            leaf_count INTEGER,
            leaf_degree INTEGER,
            star_leaf_attachment TEXT,
            effective_micro_topology TEXT,
            effective_leaf_count INTEGER,
            effective_leaf_degree INTEGER,
            beta REAL,
            gamma REAL,
            model INTEGER,
            n_runs INTEGER,
            base_seed INTEGER,
            T_end REAL,
            initial_community INTEGER,
            initial_node INTEGER,
            dt_out REAL,
            tau_micro REAL,
            macro_T REAL,
            network_size_total INTEGER,
            topology_key TEXT,
            topology_is_hub_leaf INTEGER,
            print_infection_events INTEGER,
            export_infection_events_csv INTEGER,
            verbose_steps INTEGER,
            micro_out_folder TEXT,
            micromacro_out_folder TEXT,
            output_csv_path TEXT,
            metrics_path TEXT,
            events_path TEXT,
            scenario_id TEXT,
            scenario_group_id TEXT,
            replication_id INTEGER,
            n_total_events INTEGER,
            n_micro_events INTEGER,
            n_macro_events INTEGER,
            n_inter_events INTEGER,
            runtime_seconds REAL,
            notes TEXT
        );
        """
    )
    cur.execute("PRAGMA table_info(runs);")
    existing_cols = {row[1] for row in cur.fetchall()}
    runs_columns_to_add = {
        "config_hash": "TEXT",
        "run_seed": "INTEGER",
        "effective_micro_topology": "TEXT",
        "effective_leaf_count": "INTEGER",
        "effective_leaf_degree": "INTEGER",
        "initial_community": "INTEGER",
        "topology_key": "TEXT",
        "topology_is_hub_leaf": "INTEGER",
        "print_infection_events": "INTEGER",
        "export_infection_events_csv": "INTEGER",
        "verbose_steps": "INTEGER",
        "micro_out_folder": "TEXT",
        "micromacro_out_folder": "TEXT",
        "scenario_id": "TEXT",
        "scenario_group_id": "TEXT",
        "replication_id": "INTEGER",
        "n_total_events": "INTEGER",
        "n_micro_events": "INTEGER",
        "n_macro_events": "INTEGER",
        "n_inter_events": "INTEGER",
    }
    for col, col_type in runs_columns_to_add.items():
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE runs ADD COLUMN {col} {col_type};")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS community_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_uid TEXT NOT NULL,
            community INTEGER NOT NULL,
            t0 REAL,
            t_bridge REAL,
            t_export REAL,
            FOREIGN KEY(run_uid) REFERENCES runs(run_uid) ON DELETE CASCADE,
            UNIQUE(run_uid, community)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS infection_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_uid TEXT NOT NULL,
            time REAL NOT NULL,
            mode TEXT,
            kind TEXT,
            src_community INTEGER,
            dst_community INTEGER,
            src_node INTEGER,
            dst_node INTEGER,
            FOREIGN KEY(run_uid) REFERENCES runs(run_uid) ON DELETE CASCADE
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sir_artifacts (
            run_uid TEXT PRIMARY KEY,
            format TEXT NOT NULL,
            compression TEXT NOT NULL,
            payload_blob BLOB NOT NULL,
            row_count INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            FOREIGN KEY(run_uid) REFERENCES runs(run_uid) ON DELETE CASCADE
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS infection_artifacts (
            run_uid TEXT PRIMARY KEY,
            format TEXT NOT NULL,
            compression TEXT NOT NULL,
            payload_blob BLOB NOT NULL,
            row_count INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            FOREIGN KEY(run_uid) REFERENCES runs(run_uid) ON DELETE CASCADE
        );
        """
    )

    # Benchmark analysis artifacts (scenario-level timing comparison outputs).
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_analyses (
            analysis_uid TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            censoring_mode TEXT NOT NULL,
            t_max REAL,
            paired_mode INTEGER NOT NULL,
            notes TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_summary_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_uid TEXT NOT NULL,
            scenario_id TEXT NOT NULL,
            variable TEXT NOT NULL,
            community INTEGER,
            metric TEXT NOT NULL,
            value REAL,
            FOREIGN KEY(analysis_uid) REFERENCES benchmark_analyses(analysis_uid) ON DELETE CASCADE,
            UNIQUE(analysis_uid, scenario_id, variable, community, metric)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_curve_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_uid TEXT NOT NULL,
            scenario_id TEXT NOT NULL,
            model TEXT NOT NULL,
            curve_kind TEXT NOT NULL,
            community INTEGER,
            x REAL,
            y REAL,
            bin_left REAL,
            bin_right REAL,
            FOREIGN KEY(analysis_uid) REFERENCES benchmark_analyses(analysis_uid) ON DELETE CASCADE
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_boxplot_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_uid TEXT NOT NULL,
            scenario_id TEXT NOT NULL,
            model TEXT NOT NULL,
            community INTEGER NOT NULL,
            n_total INTEGER NOT NULL,
            n_reached INTEGER NOT NULL,
            q25 REAL,
            q50 REAL,
            q75 REAL,
            iqr REAL,
            whisker_low REAL,
            whisker_high REAL,
            outlier_count INTEGER NOT NULL,
            FOREIGN KEY(analysis_uid) REFERENCES benchmark_analyses(analysis_uid) ON DELETE CASCADE,
            UNIQUE(analysis_uid, scenario_id, model, community)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_paired_deltas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_uid TEXT NOT NULL,
            scenario_id TEXT NOT NULL,
            replication_id INTEGER NOT NULL,
            community INTEGER NOT NULL,
            delta_t REAL,
            FOREIGN KEY(analysis_uid) REFERENCES benchmark_analyses(analysis_uid) ON DELETE CASCADE,
            UNIQUE(analysis_uid, scenario_id, replication_id, community)
        );
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_runs_search
        ON runs (
            simulator, macro_graph_type, micro_graph_type, communities,
            community_size, network_size_total, beta, gamma, model, T_end, seed, run_seed, created_at
        );
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_runs_topology
        ON runs (
            macro_graph_type, micro_graph_type, effective_micro_topology,
            communities, community_size, inter_links, leaf_count, leaf_degree,
            star_leaf_attachment, topology_is_hub_leaf, network_size_total
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_output_csv_path ON runs(output_csv_path);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run_community ON community_metrics(run_uid, community);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_run_time_kind_mode ON infection_events(run_uid, time, kind, mode);")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_pairing ON runs(scenario_id, replication_id, simulator, run_seed);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_bench_summary_lookup "
        "ON benchmark_summary_metrics(analysis_uid, scenario_id, variable, community);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_bench_curve_lookup "
        "ON benchmark_curve_points(analysis_uid, scenario_id, model, curve_kind, community, x);"
    )


def init_db(db_path: Path | str = "simulations.db") -> Path:
    db_path = resolve_path(db_path)
    with _connect(db_path) as conn:
        _init_legacy_schema(conn)
        _init_v1_schema(conn)
        conn.commit()
    return db_path


# ---------- Legacy API (kept for backward compatibility) ----------
def log_run(
    simulator: str,
    sim_version: str,
    network_params: Dict[str, Any],
    virus_params: Dict[str, Any],
    sim_params: Dict[str, Any],
    macro_params: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    runtime_seconds: Optional[float] = None,
    notes: Optional[str] = None,
    db_path: Path | str = "simulations.db",
) -> int:
    db_path = resolve_path(db_path)
    init_db(db_path)
    created_at = _utc_now()
    if macro_params is None:
        macro_params = sim_params

    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO sim_runs (
                created_at, simulator, sim_version,
                network_params, virus_params, macro_params, sim_params,
                output_path, runtime_seconds, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                created_at,
                simulator,
                sim_version,
                _json_dumps(network_params),
                _json_dumps(virus_params),
                _json_dumps(macro_params),
                _json_dumps(sim_params),
                output_path,
                runtime_seconds,
                notes,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_runs(db_path: Path | str = "simulations.db") -> List[Dict[str, Any]]:
    db_path = resolve_path(db_path)
    if not db_path.exists():
        return []
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM sim_runs ORDER BY id DESC;")
        rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "simulator": r["simulator"],
                "sim_version": r["sim_version"],
                "network_params": _json_loads_or_empty(r["network_params"]),
                "virus_params": _json_loads_or_empty(r["virus_params"]),
                "sim_params": _json_loads_or_empty(r["sim_params"]),
                "macro_params": _json_loads_or_empty(r["macro_params"]),
                "output_path": r["output_path"],
                "runtime_seconds": r["runtime_seconds"],
                "notes": r["notes"],
            }
        )
    return out


# ---------- v1 API ----------
_RUN_FILTER_FIELDS = {
    "run_uid",
    "created_at",
    "simulator",
    "sim_version",
    "status",
    "config_source",
    "config_json",
    "config_hash",
    "communities",
    "community_size",
    "inter_links",
    "seed",
    "run_seed",
    "macro_graph_type",
    "micro_graph_type",
    "edge_prob",
    "leaf_count",
    "leaf_degree",
    "star_leaf_attachment",
    "effective_micro_topology",
    "effective_leaf_count",
    "effective_leaf_degree",
    "beta",
    "gamma",
    "model",
    "n_runs",
    "base_seed",
    "T_end",
    "initial_community",
    "initial_node",
    "dt_out",
    "tau_micro",
    "macro_T",
    "network_size_total",
    "topology_key",
    "topology_is_hub_leaf",
    "print_infection_events",
    "export_infection_events_csv",
    "verbose_steps",
    "micro_out_folder",
    "micromacro_out_folder",
    "output_csv_path",
    "metrics_path",
    "events_path",
    "scenario_id",
    "scenario_group_id",
    "replication_id",
    "n_total_events",
    "n_micro_events",
    "n_macro_events",
    "n_inter_events",
    "runtime_seconds",
    "notes",
}


def ingest_run_bundle(
    *,
    simulator: str,
    sim_version: str,
    config: Dict[str, Any],
    run_csv_path: str | Path,
    metrics_path: str | Path | None = None,
    events_path: str | Path | None = None,
    run_uid: str | None = None,
    created_at: str | None = None,
    status: str = "completed",
    config_source: str = "runtime",
    runtime_seconds: float | None = None,
    scenario_id: str | None = None,
    scenario_group_id: str | None = None,
    replication_id: int | None = None,
    n_total_events: int | None = None,
    n_micro_events: int | None = None,
    n_macro_events: int | None = None,
    n_inter_events: int | None = None,
    notes: str | None = None,
    run_seed: int | None = None,
    db_path: Path | str = "simulations.db",
) -> str:
    db_path = resolve_path(db_path)
    init_db(db_path)

    csv_path = resolve_path(run_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing run CSV: {csv_path}")

    if metrics_path is None:
        metrics_file = _find_metrics_file(csv_path)
    else:
        metrics_file = resolve_path(metrics_path)
    if metrics_file is not None and not metrics_file.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_file}")

    if events_path is None:
        events_file = _find_events_file(csv_path)
    else:
        events_file = resolve_path(events_path)
    if events_file is not None and not events_file.exists():
        raise FileNotFoundError(f"Missing events file: {events_file}")

    run_uid = run_uid or _default_run_uid(simulator=simulator, run_csv_path=csv_path)
    created_at = created_at or _utc_now()

    run_fields = _extract_run_fields_from_config(config)
    config_json = _json_dumps(config)
    config_hash = hashlib.sha1(config_json.encode("utf-8")).hexdigest()

    sir_payload, _, _ = _read_sir_payload(csv_path)

    metrics_rows = _load_metrics_rows(metrics_file) if metrics_file is not None else []
    event_rows = _load_events_rows(events_file) if events_file is not None else []
    if pd is None:
        raise ImportError("pandas is required for artifact ingestion")
    sir_df = pd.read_csv(io.StringIO(sir_payload.decode("utf-8")))

    with _connect(db_path) as conn:
        cur = conn.cursor()

        run_insert_columns = [
            "run_uid", "created_at", "simulator", "sim_version", "status", "config_source", "config_json", "config_hash",
            "communities", "community_size", "inter_links", "seed", "run_seed", "macro_graph_type", "micro_graph_type",
            "edge_prob", "leaf_count", "leaf_degree", "star_leaf_attachment",
            "effective_micro_topology", "effective_leaf_count", "effective_leaf_degree",
            "beta", "gamma", "model", "n_runs", "base_seed", "T_end", "initial_community", "initial_node",
            "dt_out", "tau_micro", "macro_T", "network_size_total", "topology_key", "topology_is_hub_leaf",
            "print_infection_events", "export_infection_events_csv", "verbose_steps",
            "micro_out_folder", "micromacro_out_folder",
            "output_csv_path", "metrics_path", "events_path",
            "scenario_id", "scenario_group_id", "replication_id",
            "n_total_events", "n_micro_events", "n_macro_events", "n_inter_events",
            "runtime_seconds", "notes",
        ]
        run_values = (
            run_uid,
            created_at,
            simulator,
            sim_version,
            status,
            config_source,
            config_json,
            config_hash,
            run_fields["communities"],
            run_fields["community_size"],
            run_fields["inter_links"],
            run_fields["seed"],
            run_seed,
            run_fields["macro_graph_type"],
            run_fields["micro_graph_type"],
            run_fields["edge_prob"],
            run_fields["leaf_count"],
            run_fields["leaf_degree"],
            run_fields["star_leaf_attachment"],
            run_fields["effective_micro_topology"],
            run_fields["effective_leaf_count"],
            run_fields["effective_leaf_degree"],
            run_fields["beta"],
            run_fields["gamma"],
            run_fields["model"],
            run_fields["n_runs"],
            run_fields["base_seed"],
            run_fields["T_end"],
            run_fields["initial_community"],
            run_fields["initial_node"],
            run_fields["dt_out"],
            run_fields["tau_micro"],
            run_fields["macro_T"],
            run_fields["network_size_total"],
            run_fields["topology_key"],
            run_fields["topology_is_hub_leaf"],
            run_fields["print_infection_events"],
            run_fields["export_infection_events_csv"],
            run_fields["verbose_steps"],
            run_fields["micro_out_folder"],
            run_fields["micromacro_out_folder"],
            str(csv_path),
            str(metrics_file) if metrics_file is not None else None,
            str(events_file) if events_file is not None else None,
            scenario_id,
            scenario_group_id,
            replication_id,
            n_total_events,
            n_micro_events,
            n_macro_events,
            n_inter_events,
            runtime_seconds,
            notes,
        )
        run_placeholder_sql = ", ".join(["?"] * len(run_values))
        run_columns_sql = ", ".join(run_insert_columns)

        cur.execute(
            f"""
            INSERT INTO runs ({run_columns_sql})
            VALUES ({run_placeholder_sql})
            ON CONFLICT(run_uid) DO UPDATE SET
                created_at=excluded.created_at,
                simulator=excluded.simulator,
                sim_version=excluded.sim_version,
                status=excluded.status,
                config_source=excluded.config_source,
                config_json=excluded.config_json,
                config_hash=excluded.config_hash,
                communities=excluded.communities,
                community_size=excluded.community_size,
                inter_links=excluded.inter_links,
                seed=excluded.seed,
                run_seed=excluded.run_seed,
                macro_graph_type=excluded.macro_graph_type,
                micro_graph_type=excluded.micro_graph_type,
                edge_prob=excluded.edge_prob,
                leaf_count=excluded.leaf_count,
                leaf_degree=excluded.leaf_degree,
                star_leaf_attachment=excluded.star_leaf_attachment,
                effective_micro_topology=excluded.effective_micro_topology,
                effective_leaf_count=excluded.effective_leaf_count,
                effective_leaf_degree=excluded.effective_leaf_degree,
                beta=excluded.beta,
                gamma=excluded.gamma,
                model=excluded.model,
                n_runs=excluded.n_runs,
                base_seed=excluded.base_seed,
                T_end=excluded.T_end,
                initial_community=excluded.initial_community,
                initial_node=excluded.initial_node,
                dt_out=excluded.dt_out,
                tau_micro=excluded.tau_micro,
                macro_T=excluded.macro_T,
                network_size_total=excluded.network_size_total,
                topology_key=excluded.topology_key,
                topology_is_hub_leaf=excluded.topology_is_hub_leaf,
                print_infection_events=excluded.print_infection_events,
                export_infection_events_csv=excluded.export_infection_events_csv,
                verbose_steps=excluded.verbose_steps,
                micro_out_folder=excluded.micro_out_folder,
                micromacro_out_folder=excluded.micromacro_out_folder,
                output_csv_path=excluded.output_csv_path,
                metrics_path=excluded.metrics_path,
                events_path=excluded.events_path,
                scenario_id=excluded.scenario_id,
                scenario_group_id=excluded.scenario_group_id,
                replication_id=excluded.replication_id,
                n_total_events=excluded.n_total_events,
                n_micro_events=excluded.n_micro_events,
                n_macro_events=excluded.n_macro_events,
                n_inter_events=excluded.n_inter_events,
                runtime_seconds=excluded.runtime_seconds,
                notes=excluded.notes;
            """,
            run_values,
        )

        cur.execute("DELETE FROM community_metrics WHERE run_uid = ?;", (run_uid,))
        cur.execute("DELETE FROM infection_events WHERE run_uid = ?;", (run_uid,))

        if metrics_rows:
            cur.executemany(
                """
                INSERT INTO community_metrics (run_uid, community, t0, t_bridge, t_export)
                VALUES (?, ?, ?, ?, ?);
                """,
                [
                    (
                        run_uid,
                        row["community"],
                        row["t0"],
                        row["t_bridge"],
                        row["t_export"],
                    )
                    for row in metrics_rows
                    if row.get("community") is not None
                ],
            )

        if event_rows:
            cur.executemany(
                """
                INSERT INTO infection_events (
                    run_uid, time, mode, kind, src_community, dst_community, src_node, dst_node
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    (
                        run_uid,
                        row["time"],
                        row["mode"],
                        row["kind"],
                        row["src_community"],
                        row["dst_community"],
                        row["src_node"],
                        row["dst_node"],
                    )
                    for row in event_rows
                    if row.get("time") is not None
                ],
            )

        sir_fmt, sir_comp, sir_blob, sir_row_count, sir_sha256 = _serialize_df_for_artifact(sir_df)
        cur.execute(
            """
            INSERT INTO sir_artifacts (run_uid, format, compression, payload_blob, row_count, sha256)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_uid) DO UPDATE SET
                format=excluded.format,
                compression=excluded.compression,
                payload_blob=excluded.payload_blob,
                row_count=excluded.row_count,
                sha256=excluded.sha256;
            """,
            (
                run_uid,
                sir_fmt,
                sir_comp,
                sqlite3.Binary(sir_blob),
                sir_row_count,
                sir_sha256,
            ),
        )
        if event_rows:
            events_df = pd.DataFrame(event_rows)
            event_fmt, event_comp, event_blob, event_row_count, event_sha256 = _serialize_df_for_artifact(events_df)
            cur.execute(
                """
                INSERT INTO infection_artifacts (run_uid, format, compression, payload_blob, row_count, sha256)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_uid) DO UPDATE SET
                    format=excluded.format,
                    compression=excluded.compression,
                    payload_blob=excluded.payload_blob,
                    row_count=excluded.row_count,
                    sha256=excluded.sha256;
                """,
                (
                    run_uid,
                    event_fmt,
                    event_comp,
                    sqlite3.Binary(event_blob),
                    event_row_count,
                    event_sha256,
                ),
            )
        else:
            cur.execute("DELETE FROM infection_artifacts WHERE run_uid = ?;", (run_uid,))

        conn.commit()

    return run_uid


def ingest_run_payload(
    *,
    simulator: str,
    sim_version: str,
    config: Dict[str, Any],
    sir_csv_bytes: bytes,
    community_metrics: List[Dict[str, Any]] | None = None,
    infection_events: List[Dict[str, Any]] | None = None,
    run_uid: str | None = None,
    created_at: str | None = None,
    status: str = "completed",
    config_source: str = "runtime",
    runtime_seconds: float | None = None,
    scenario_id: str | None = None,
    scenario_group_id: str | None = None,
    replication_id: int | None = None,
    n_total_events: int | None = None,
    n_micro_events: int | None = None,
    n_macro_events: int | None = None,
    n_inter_events: int | None = None,
    notes: str | None = None,
    run_seed: int | None = None,
    output_csv_path: str | None = None,
    metrics_path: str | None = None,
    events_path: str | None = None,
    db_path: Path | str = "simulations.db",
) -> str:
    db_path = resolve_path(db_path)
    init_db(db_path)

    if not isinstance(sir_csv_bytes, (bytes, bytearray)) or len(sir_csv_bytes) == 0:
        raise ValueError("sir_csv_bytes must be non-empty bytes")

    created_at = created_at or _utc_now()
    if run_uid is None:
        run_uid = make_run_uid(
            simulator=simulator,
            config=config,
            run_seed=run_seed,
            created_at=created_at,
        )

    run_fields = _extract_run_fields_from_config(config)
    config_json = _json_dumps(config)
    config_hash = hashlib.sha1(config_json.encode("utf-8")).hexdigest()
    metrics_rows = community_metrics or []
    event_rows = infection_events or []
    sir_payload_bytes = bytes(sir_csv_bytes)
    if pd is None:
        raise ImportError("pandas is required for artifact ingestion")
    sir_df = pd.read_csv(io.StringIO(sir_payload_bytes.decode("utf-8")))

    with _connect(db_path) as conn:
        cur = conn.cursor()
        run_insert_columns = [
            "run_uid", "created_at", "simulator", "sim_version", "status", "config_source", "config_json", "config_hash",
            "communities", "community_size", "inter_links", "seed", "run_seed", "macro_graph_type", "micro_graph_type",
            "edge_prob", "leaf_count", "leaf_degree", "star_leaf_attachment",
            "effective_micro_topology", "effective_leaf_count", "effective_leaf_degree",
            "beta", "gamma", "model", "n_runs", "base_seed", "T_end", "initial_community", "initial_node",
            "dt_out", "tau_micro", "macro_T", "network_size_total", "topology_key", "topology_is_hub_leaf",
            "print_infection_events", "export_infection_events_csv", "verbose_steps",
            "micro_out_folder", "micromacro_out_folder",
            "output_csv_path", "metrics_path", "events_path",
            "scenario_id", "scenario_group_id", "replication_id",
            "n_total_events", "n_micro_events", "n_macro_events", "n_inter_events",
            "runtime_seconds", "notes",
        ]
        run_values = (
            run_uid,
            created_at,
            simulator,
            sim_version,
            status,
            config_source,
            config_json,
            config_hash,
            run_fields["communities"],
            run_fields["community_size"],
            run_fields["inter_links"],
            run_fields["seed"],
            run_seed,
            run_fields["macro_graph_type"],
            run_fields["micro_graph_type"],
            run_fields["edge_prob"],
            run_fields["leaf_count"],
            run_fields["leaf_degree"],
            run_fields["star_leaf_attachment"],
            run_fields["effective_micro_topology"],
            run_fields["effective_leaf_count"],
            run_fields["effective_leaf_degree"],
            run_fields["beta"],
            run_fields["gamma"],
            run_fields["model"],
            run_fields["n_runs"],
            run_fields["base_seed"],
            run_fields["T_end"],
            run_fields["initial_community"],
            run_fields["initial_node"],
            run_fields["dt_out"],
            run_fields["tau_micro"],
            run_fields["macro_T"],
            run_fields["network_size_total"],
            run_fields["topology_key"],
            run_fields["topology_is_hub_leaf"],
            run_fields["print_infection_events"],
            run_fields["export_infection_events_csv"],
            run_fields["verbose_steps"],
            run_fields["micro_out_folder"],
            run_fields["micromacro_out_folder"],
            output_csv_path,
            metrics_path,
            events_path,
            scenario_id,
            scenario_group_id,
            replication_id,
            n_total_events,
            n_micro_events,
            n_macro_events,
            n_inter_events,
            runtime_seconds,
            notes,
        )
        run_placeholder_sql = ", ".join(["?"] * len(run_values))
        run_columns_sql = ", ".join(run_insert_columns)

        cur.execute(
            f"""
            INSERT INTO runs ({run_columns_sql})
            VALUES ({run_placeholder_sql})
            ON CONFLICT(run_uid) DO UPDATE SET
                created_at=excluded.created_at,
                simulator=excluded.simulator,
                sim_version=excluded.sim_version,
                status=excluded.status,
                config_source=excluded.config_source,
                config_json=excluded.config_json,
                config_hash=excluded.config_hash,
                communities=excluded.communities,
                community_size=excluded.community_size,
                inter_links=excluded.inter_links,
                seed=excluded.seed,
                run_seed=excluded.run_seed,
                macro_graph_type=excluded.macro_graph_type,
                micro_graph_type=excluded.micro_graph_type,
                edge_prob=excluded.edge_prob,
                leaf_count=excluded.leaf_count,
                leaf_degree=excluded.leaf_degree,
                star_leaf_attachment=excluded.star_leaf_attachment,
                effective_micro_topology=excluded.effective_micro_topology,
                effective_leaf_count=excluded.effective_leaf_count,
                effective_leaf_degree=excluded.effective_leaf_degree,
                beta=excluded.beta,
                gamma=excluded.gamma,
                model=excluded.model,
                n_runs=excluded.n_runs,
                base_seed=excluded.base_seed,
                T_end=excluded.T_end,
                initial_community=excluded.initial_community,
                initial_node=excluded.initial_node,
                dt_out=excluded.dt_out,
                tau_micro=excluded.tau_micro,
                macro_T=excluded.macro_T,
                network_size_total=excluded.network_size_total,
                topology_key=excluded.topology_key,
                topology_is_hub_leaf=excluded.topology_is_hub_leaf,
                print_infection_events=excluded.print_infection_events,
                export_infection_events_csv=excluded.export_infection_events_csv,
                verbose_steps=excluded.verbose_steps,
                micro_out_folder=excluded.micro_out_folder,
                micromacro_out_folder=excluded.micromacro_out_folder,
                output_csv_path=excluded.output_csv_path,
                metrics_path=excluded.metrics_path,
                events_path=excluded.events_path,
                scenario_id=excluded.scenario_id,
                scenario_group_id=excluded.scenario_group_id,
                replication_id=excluded.replication_id,
                n_total_events=excluded.n_total_events,
                n_micro_events=excluded.n_micro_events,
                n_macro_events=excluded.n_macro_events,
                n_inter_events=excluded.n_inter_events,
                runtime_seconds=excluded.runtime_seconds,
                notes=excluded.notes;
            """,
            run_values,
        )

        cur.execute("DELETE FROM community_metrics WHERE run_uid = ?;", (run_uid,))
        cur.execute("DELETE FROM infection_events WHERE run_uid = ?;", (run_uid,))
        if metrics_rows:
            cur.executemany(
                """
                INSERT INTO community_metrics (run_uid, community, t0, t_bridge, t_export)
                VALUES (?, ?, ?, ?, ?);
                """,
                [
                    (
                        run_uid,
                        _to_int_or_none(row.get("community")),
                        _to_float_or_none(row.get("t0")),
                        _to_float_or_none(row.get("t_bridge")),
                        _to_float_or_none(row.get("t_export")),
                    )
                    for row in metrics_rows
                    if row.get("community") is not None
                ],
            )
        if event_rows:
            cur.executemany(
                """
                INSERT INTO infection_events (
                    run_uid, time, mode, kind, src_community, dst_community, src_node, dst_node
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    (
                        run_uid,
                        _to_float_or_none(row.get("time")),
                        _to_str_or_none(row.get("mode")),
                        _to_str_or_none(row.get("kind")),
                        _to_int_or_none(row.get("src_community")),
                        _to_int_or_none(row.get("dst_community")),
                        _to_int_or_none(row.get("src_node")),
                        _to_int_or_none(row.get("dst_node")),
                    )
                    for row in event_rows
                    if row.get("time") is not None
                ],
            )
        sir_fmt, sir_comp, sir_blob, row_count, sir_sha256 = _serialize_df_for_artifact(sir_df)
        cur.execute(
            """
            INSERT INTO sir_artifacts (run_uid, format, compression, payload_blob, row_count, sha256)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_uid) DO UPDATE SET
                format=excluded.format,
                compression=excluded.compression,
                payload_blob=excluded.payload_blob,
                row_count=excluded.row_count,
                sha256=excluded.sha256;
            """,
            (
                run_uid,
                sir_fmt,
                sir_comp,
                sqlite3.Binary(sir_blob),
                row_count,
                sir_sha256,
            ),
        )
        if event_rows:
            events_df = pd.DataFrame(event_rows)
            event_fmt, event_comp, event_blob, event_row_count, event_sha256 = _serialize_df_for_artifact(events_df)
            cur.execute(
                """
                INSERT INTO infection_artifacts (run_uid, format, compression, payload_blob, row_count, sha256)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_uid) DO UPDATE SET
                    format=excluded.format,
                    compression=excluded.compression,
                    payload_blob=excluded.payload_blob,
                    row_count=excluded.row_count,
                    sha256=excluded.sha256;
                """,
                (
                    run_uid,
                    event_fmt,
                    event_comp,
                    sqlite3.Binary(event_blob),
                    event_row_count,
                    event_sha256,
                ),
            )
        else:
            cur.execute("DELETE FROM infection_artifacts WHERE run_uid = ?;", (run_uid,))
        conn.commit()
    return run_uid


def ingest_paths_from_batch(
    *,
    paths: Iterable[str | Path],
    simulator: str,
    sim_version: str,
    config: Dict[str, Any],
    config_source: str = "runtime",
    db_path: Path | str = "simulations.db",
) -> List[str]:
    run_uids: List[str] = []
    for path in paths:
        run_uid = ingest_run_bundle(
            simulator=simulator,
            sim_version=sim_version,
            config=config,
            run_csv_path=path,
            config_source=config_source,
            db_path=db_path,
        )
        run_uids.append(run_uid)
    return run_uids


def backfill_from_outputs(
    *,
    simulator: str,
    sim_version: str,
    config: Dict[str, Any],
    out_folder: str | Path,
    limit: int = 100,
    config_source: str = "assumed_backfill",
    db_path: Path | str = "simulations.db",
) -> List[str]:
    folder = resolve_path(out_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Output folder does not exist: {folder}")

    files = _sorted_run_csv_files(folder)
    if limit > 0:
        files = files[:limit]

    run_uids: List[str] = []
    for csv_path in files:
        run_uid = ingest_run_bundle(
            simulator=simulator,
            sim_version=sim_version,
            config=config,
            run_csv_path=csv_path,
            config_source=config_source,
            db_path=db_path,
        )
        run_uids.append(run_uid)
    return run_uids


def _build_where_clause(filters: Dict[str, Any]) -> Tuple[str, List[Any]]:
    clauses: List[str] = []
    params: List[Any] = []
    for key, value in filters.items():
        if key not in _RUN_FILTER_FIELDS:
            raise ValueError(f"Unsupported filter field: {key}")
        if isinstance(value, dict):
            vmin = value.get("min")
            vmax = value.get("max")
            if vmin is not None:
                clauses.append(f"{key} >= ?")
                params.append(vmin)
            if vmax is not None:
                clauses.append(f"{key} <= ?")
                params.append(vmax)
            continue
        if isinstance(value, (list, tuple, set)):
            vals = list(value)
            if not vals:
                clauses.append("1=0")
                continue
            placeholders = ",".join(["?"] * len(vals))
            clauses.append(f"{key} IN ({placeholders})")
            params.extend(vals)
            continue
        if value is None:
            clauses.append(f"{key} IS NULL")
            continue
        clauses.append(f"{key} = ?")
        params.append(value)
    where_sql = " AND ".join(clauses) if clauses else "1=1"
    return where_sql, params


def _sanitize_sort(sort: str) -> str:
    allowed = set(_RUN_FILTER_FIELDS) | {"created_at"}
    parts = sort.strip().split()
    if not parts:
        return "created_at DESC"
    field = parts[0]
    if field not in allowed:
        raise ValueError(f"Unsupported sort field: {field}")
    direction = "DESC"
    if len(parts) > 1:
        cand = parts[1].upper()
        if cand not in {"ASC", "DESC"}:
            raise ValueError(f"Unsupported sort direction: {cand}")
        direction = cand
    return f"{field} {direction}"


def _row_to_run_dict(row: sqlite3.Row) -> Dict[str, Any]:
    out = dict(row)
    out["config_json"] = _json_loads_or_empty(out.get("config_json"))
    return out


def search_by_filters(
    filters: Dict[str, Any],
    *,
    sort: str = "created_at DESC",
    limit: int = 100,
    offset: int = 0,
    db_path: Path | str = "simulations.db",
) -> List[Dict[str, Any]]:
    db_path = resolve_path(db_path)
    init_db(db_path)
    where_sql, params = _build_where_clause(filters)
    order_sql = _sanitize_sort(sort)

    query = f"""
        SELECT *
        FROM runs
        WHERE {where_sql}
        ORDER BY {order_sql}
        LIMIT ? OFFSET ?;
    """
    params = [*params, int(limit), int(offset)]

    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
    return [_row_to_run_dict(row) for row in rows]


def search_by_config(
    config_like: Dict[str, Any],
    *,
    exact: bool = False,
    limit: int = 100,
    offset: int = 0,
    db_path: Path | str = "simulations.db",
) -> List[Dict[str, Any]]:
    if exact:
        payload = _json_dumps(config_like)
        return search_by_filters(
            {"config_json": payload},
            sort="created_at DESC",
            limit=limit,
            offset=offset,
            db_path=db_path,
        )

    flattened = _extract_run_fields_from_config(config_like)
    filters: Dict[str, Any] = {k: v for k, v in flattened.items() if v is not None}
    derived_fields = {
        "network_size_total",
        "topology_key",
        "topology_is_hub_leaf",
        "effective_micro_topology",
        "effective_leaf_count",
        "effective_leaf_degree",
    }
    for key in derived_fields:
        filters.pop(key, None)

    for key, value in config_like.items():
        if key in _RUN_FILTER_FIELDS and value is not None:
            filters[key] = value

    return search_by_filters(
        filters,
        sort="created_at DESC",
        limit=limit,
        offset=offset,
        db_path=db_path,
    )


def get_distinct_configs(
    *,
    filter_by: Dict[str, Any],
    fields: Sequence[str],
    db_path: Path | str = "simulations.db",
) -> List[Dict[str, Any]]:
    if not fields:
        return []
    for field in fields:
        if field not in _RUN_FILTER_FIELDS:
            raise ValueError(f"Unsupported distinct field: {field}")

    db_path = resolve_path(db_path)
    init_db(db_path)
    where_sql, params = _build_where_clause(filter_by)
    cols = ", ".join(fields)
    query = f"""
        SELECT DISTINCT {cols}
        FROM runs
        WHERE {where_sql}
        ORDER BY {cols};
    """
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
    return [dict(row) for row in rows]


def get_run_details(run_uid: str, *, db_path: Path | str = "simulations.db") -> Dict[str, Any]:
    db_path = resolve_path(db_path)
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT * FROM runs WHERE run_uid = ?;", (run_uid,))
        run_row = cur.fetchone()
        if run_row is None:
            raise KeyError(f"run_uid not found: {run_uid}")

        cur.execute(
            """
            SELECT community, t0, t_bridge, t_export
            FROM community_metrics
            WHERE run_uid = ?
            ORDER BY community;
            """,
            (run_uid,),
        )
        metrics_rows = [dict(row) for row in cur.fetchall()]

        cur.execute(
            """
            SELECT time, mode, kind, src_community, dst_community, src_node, dst_node
            FROM infection_events
            WHERE run_uid = ?
            ORDER BY time, id;
            """,
            (run_uid,),
        )
        event_rows = [dict(row) for row in cur.fetchall()]

        cur.execute(
            """
            SELECT format, compression, row_count, sha256, length(payload_blob) AS payload_size
            FROM sir_artifacts
            WHERE run_uid = ?;
            """,
            (run_uid,),
        )
        sir_row = cur.fetchone()
        cur.execute(
            """
            SELECT format, compression, row_count, sha256, length(payload_blob) AS payload_size
            FROM infection_artifacts
            WHERE run_uid = ?;
            """,
            (run_uid,),
        )
        infection_art_row = cur.fetchone()

    run_data = _row_to_run_dict(run_row)
    return {
        "run": run_data,
        "community_metrics": metrics_rows,
        "infection_events": event_rows,
        "sir_artifact": dict(sir_row) if sir_row is not None else None,
        "infection_artifact": dict(infection_art_row) if infection_art_row is not None else None,
    }


def export_sir_csv(
    run_uid: str,
    out_path: str | Path,
    *,
    db_path: Path | str = "simulations.db",
) -> Path:
    db_path = resolve_path(db_path)
    init_db(db_path)
    out = resolve_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT format, compression, payload_blob
            FROM sir_artifacts
            WHERE run_uid = ?;
            """,
            (run_uid,),
        )
        row = cur.fetchone()

    if row is None:
        raise KeyError(f"sir_artifact not found for run_uid: {run_uid}")

    fmt, compression, payload_blob = row
    data = _deserialize_artifact_to_csv_bytes(
        fmt=fmt,
        compression=compression,
        payload_blob=payload_blob,
    )

    out.write_bytes(data)
    return out


def load_sir_dataframe(run_uid: str, *, db_path: Path | str = "simulations.db"):
    db_path = resolve_path(db_path)
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT format, compression, payload_blob
            FROM sir_artifacts
            WHERE run_uid = ?;
            """,
            (run_uid,),
        )
        row = cur.fetchone()
    if row is None:
        raise KeyError(f"sir_artifact not found for run_uid: {run_uid}")
    fmt, compression, payload_blob = row
    return _deserialize_artifact_to_df(fmt=fmt, compression=compression, payload_blob=payload_blob)


def load_infection_events_dataframe(run_uid: str, *, db_path: Path | str = "simulations.db"):
    db_path = resolve_path(db_path)
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT format, compression, payload_blob
            FROM infection_artifacts
            WHERE run_uid = ?;
            """,
            (run_uid,),
        )
        row = cur.fetchone()
    if row is None:
        raise KeyError(f"infection_artifact not found for run_uid: {run_uid}")
    fmt, compression, payload_blob = row
    return _deserialize_artifact_to_df(fmt=fmt, compression=compression, payload_blob=payload_blob)


__all__ = [
    "init_db",
    "log_run",
    "list_runs",
    "ingest_run_bundle",
    "ingest_run_payload",
    "ingest_paths_from_batch",
    "make_run_uid",
    "backfill_from_outputs",
    "search_by_config",
    "search_by_filters",
    "get_distinct_configs",
    "get_run_details",
    "export_sir_csv",
    "load_sir_dataframe",
    "load_infection_events_dataframe",
]
