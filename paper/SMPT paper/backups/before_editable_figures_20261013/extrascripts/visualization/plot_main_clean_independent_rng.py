#!/usr/bin/env python3
"""Rebuild the figures referenced by paper/main.tex from completed benchmark data.

This deliberately never imports or executes a simulator.  Aggregate dynamics are
read from the per-replication gzip checkpoints so the 9-GB trajectories.csv does
not have to be scanned.
"""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "output" / "main_clean_benchmarks_independent_rng"
CHECKPOINTS = DATA / ".checkpoints"
OUT = ROOT / "plots" / "main_clean_independent_rng"
PAPER = ROOT / "paper" / "Images"
MODELS = ("Full Micro", "Production MicroMacro + independent RNG")
SHORT = {MODELS[0]: "Micro", MODELS[1]: "MicroMacro"}
COLORS = {MODELS[0]: "#2878b5", MODELS[1]: "#d0604c"}

CLIQUE = {
    "baseline": "chain_cliques_L10_n100_k5_baseline",
    "k1": "chain_cliques_L10_n100_k1",
    "k1dyn": "chain_cliques_L10_n100_k1_dynamics_T5000",
    "k10": "chain_cliques_L10_n100_k10", "k20": "chain_cliques_L10_n100_k20",
    "L5": "chain_cliques_L5_n100_k5", "L15": "chain_cliques_L15_n100_k5",
    "n50": "chain_cliques_L10_n50_k5", "n200": "chain_cliques_L10_n200_k5",
}
RANDOM = {
    "baseline": "chain_random_connected_k5_d6_baseline",
    "k1": "chain_random_connected_k1_d6", "k10": "chain_random_connected_k10_d6",
    "k20": "chain_random_connected_k20_d6", "d4": "chain_random_connected_k5_d4",
    "d10": "chain_random_connected_k5_d10",
}


def msg(s: str) -> None:
    print(s, flush=True)


def save(fig, family: str, name: str, made: list[Path]) -> None:
    dest = OUT / family / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paper = PAPER / family / name
    paper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dest, paper)
    made.extend([dest, paper])


def terminal_summary(runs: pd.DataFrame, runtime: pd.DataFrame, benchmark: str, meta: dict) -> dict:
    d = runs[runs.benchmark.eq(benchmark)]
    out = {"benchmark": benchmark, **meta}
    vals = {}
    for model in MODELS:
        g = d[d.simulator.eq(model)]
        done = g[g.completion.astype(bool)].T_last.dropna().to_numpy(float)
        vals[model] = done
        out[f"completion_{SHORT[model]}"] = float(g.completion.astype(bool).mean())
        out[f"median_Tlast_{SHORT[model]}"] = float(np.median(done)) if len(done) else np.nan
        out[f"mean_Tlast_{SHORT[model]}"] = float(np.mean(done)) if len(done) else np.nan
    a, b = vals[MODELS[0]], vals[MODELS[1]]
    out["N_completed_Micro"] = len(a); out["N_completed_MicroMacro"] = len(b)
    out["W1_Tlast_completed"] = wasserstein_distance(a, b) if len(a) and len(b) else np.nan
    micro_mean = float(np.mean(a)) if len(a) else np.nan
    out["relative_W1_Tlast_completed"] = out["W1_Tlast_completed"] / micro_mean if np.isfinite(micro_mean) and micro_mean else np.nan
    out["KS_Tlast_completed"] = ks_2samp(a, b).statistic if len(a) and len(b) else np.nan
    out["mean_shift_completed_MM_minus_Micro"] = float(np.mean(b)-np.mean(a)) if len(a) and len(b) else np.nan
    out["median_shift_completed_MM_minus_Micro"] = float(np.median(b)-np.median(a)) if len(a) and len(b) else np.nan
    out["q90_shift_completed_MM_minus_Micro"] = float(np.quantile(b,.9)-np.quantile(a,.9)) if len(a) and len(b) else np.nan
    rt = runtime[runtime.benchmark.eq(benchmark)].groupby("simulator").runtime_seconds.mean()
    out["runtime_ratio_Micro_over_MicroMacro"] = float(rt.get(MODELS[0], np.nan) / rt.get(MODELS[1], np.nan))
    return out


def paired_layer(arrivals: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    d = arrivals[arrivals.benchmark.eq(benchmark)].copy()
    w = d.pivot_table(index=["replication_id", "community"], columns="simulator",
                      values="first_infection_time", aggfunc="first").reset_index()
    w["error"] = w[MODELS[1]] - w[MODELS[0]]
    rows = []
    for layer, g in w.groupby("community"):
        e = g.error.dropna().to_numpy(float)
        a = g[MODELS[0]].dropna().to_numpy(float); b = g[MODELS[1]].dropna().to_numpy(float)
        mean_a = float(np.mean(a)) if len(a) else np.nan
        rows.append({"benchmark": benchmark, "community": int(layer), "paired_N": len(e),
                     "median_error_MM_minus_Micro": np.median(e) if len(e) else np.nan,
                     "q25_error": np.quantile(e,.25) if len(e) else np.nan,
                     "q75_error": np.quantile(e,.75) if len(e) else np.nan,
                     "W1_arrival_completed": wasserstein_distance(a,b) if len(a) and len(b) else np.nan,
                     "relative_W1_arrival_completed": wasserstein_distance(a,b)/mean_a if len(a) and len(b) and mean_a else np.nan,
                     "KS_arrival_completed": ks_2samp(a,b).statistic if len(a) and len(b) else np.nan,
                     "mean_shift_completed_MM_minus_Micro": float(np.mean(b)-np.mean(a)) if len(a) and len(b) else np.nan,
                     "median_shift_completed_MM_minus_Micro": float(np.median(b)-np.median(a)) if len(a) and len(b) else np.nan,
                     "q90_shift_completed_MM_minus_Micro": float(np.quantile(b,.9)-np.quantile(a,.9)) if len(a) and len(b) else np.nan})
    return pd.DataFrame(rows).sort_values("community")


def bootstrap_ecdf_band(values: np.ndarray, grid: np.ndarray, n_boot: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic pointwise 95% percentile bands, as in the legacy plots."""
    if len(values) <= 1:
        y = np.searchsorted(np.sort(values), grid, side="right") / max(len(values), 1)
        return y, y
    rng = np.random.default_rng(12345)
    samples = np.sort(rng.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
    cdfs = np.vstack([np.searchsorted(s, grid, side="right") for s in samples]) / len(values)
    return np.quantile(cdfs,.025,axis=0), np.quantile(cdfs,.975,axis=0)


def cdf_plot(runs, benchmark, family, cdf_name, survival_name, made):
    d = runs[runs.benchmark.eq(benchmark)]
    fig, ax = plt.subplots(figsize=(7.1,4.6))
    for model in MODELS:
        x = np.sort(d[(d.simulator.eq(model)) & d.completion.astype(bool)].T_last.dropna().to_numpy(float))
        if len(x):
            grid=np.linspace(x.min(),x.max(),240); lo,hi=bootstrap_ecdf_band(x,grid)
            ax.fill_between(grid,lo,hi,color=COLORS[model],alpha=.16,linewidth=0)
            ax.step(x, np.arange(1,len(x)+1)/len(x), where="post", color=COLORS[model], label=f"{SHORT[model]} (completed)")
    ax.set(xlabel=r"$T_{last}$", ylabel="empirical CDF", ylim=(-.02,1.02)); ax.grid(alpha=.25); ax.legend(frameon=False)
    save(fig,family,cdf_name,made)
    fig, ax = plt.subplots(figsize=(7.1,4.6))
    for model in MODELS:
        g=d[d.simulator.eq(model)].sort_values("T_last")
        times=np.where(g.completion.astype(bool),g.T_last,g.T_end).astype(float)
        events=g.completion.astype(bool).to_numpy()
        # Kaplan-Meier product-limit estimate, treating horizon observations as censored.
        order=np.argsort(times); times=times[order]; events=events[order]; n=len(times); surv=1.; xs=[0.]; ys=[1.]; i=0
        while i<n:
            t=times[i]; j=i
            while j<n and times[j]==t: j+=1
            deaths=events[i:j].sum(); risk=n-i
            if deaths: surv *= 1-deaths/risk; xs.extend([t,t]); ys.extend([ys[-1],surv])
            i=j
        ax.step(xs,ys,where="post",color=COLORS[model],label=SHORT[model])
    ax.set(xlabel="time", ylabel="survival probability: not yet reached last community", ylim=(-.02,1.02)); ax.grid(alpha=.25); ax.legend(frameon=False)
    save(fig,family,survival_name,made)


def read_total_stack(benchmark: str, runs: pd.DataFrame) -> tuple[np.ndarray, dict[str,np.ndarray]]:
    d = runs[runs.benchmark.eq(benchmark)]; horizon=int(d.T_end.iloc[0]); reps=sorted(d.replication_id.unique())
    stacks={m:np.full((len(reps),horizon+1),np.nan) for m in MODELS}
    msg(f"  reading {benchmark}: {len(reps)} gzip checkpoints for aggregate dynamics")
    for ri, rep in enumerate(reps):
        p=CHECKPOINTS/benchmark/f"rep_{int(rep):04d}.json.gz"
        with gzip.open(p,"rt") as f: payload=json.load(f)
        sums={m:np.zeros(horizon+1) for m in MODELS}
        counts={m:np.zeros(horizon+1,dtype=int) for m in MODELS}
        for row in payload["trajectories"]:
            m=row["simulator"]; t=int(round(row["time"]))
            if m in sums and 0<=t<=horizon: sums[m][t]+=row["I"]; counts[m][t]+=1
        for m in MODELS:
            y=sums[m]; valid=counts[m]>0
            if valid.any():
                last=np.maximum.accumulate(np.where(valid,np.arange(horizon+1),0)); y=y[last]
                stacks[m][ri]=y
        if (ri+1)%25==0 or ri+1==len(reps): msg(f"    {ri+1}/{len(reps)}")
    return np.arange(horizon+1), stacks


def aggregate_plot(t, stacks, family, name, made, title="Total infected"):
    fig,ax=plt.subplots(figsize=(8,4.8))
    for m in MODELS:
        q=np.nanquantile(stacks[m],[.25,.5,.75],axis=0)
        ax.fill_between(t,q[0],q[2],color=COLORS[m],alpha=.20,linewidth=0)
        ax.plot(t,q[1],color=COLORS[m],label=SHORT[m],lw=1.7)
    ax.set(xlabel="time",ylabel=r"total infected $I(t)$",title=title); ax.grid(alpha=.25);ax.legend(frameon=False)
    save(fig,family,name,made)


def representative_plot(benchmark,runs,family,name,made, manifest):
    d=runs[runs.benchmark.eq(benchmark)]
    both=(d.pivot(index="replication_id",columns="simulator",values="completion").fillna(False)).all(axis=1)
    rep=int(both[both].index.min())
    p=CHECKPOINTS/benchmark/f"rep_{rep:04d}.json.gz"
    with gzip.open(p,"rt") as f: rows=json.load(f)["trajectories"]
    q=pd.DataFrame(rows); L=int(q.community.max()+1)
    fig,axes=plt.subplots(L,1,figsize=(8,1.55*L),sharex=True)
    axes=np.atleast_1d(axes)
    for c,ax in enumerate(axes):
        for m in MODELS:
            z=q[(q.community.eq(c))&q.simulator.eq(m)].sort_values("time")
            ax.plot(z.time,z.I,color=COLORS[m],lw=1.15,label=SHORT[m] if c==0 else None)
        ax.set_ylabel(f"I$_{{{c+1}}}$");ax.grid(alpha=.22)
    axes[0].legend(frameon=False,ncol=2); axes[-1].set_xlabel("time")
    fig.suptitle(f"Paired completed replication {rep}",y=1.0)
    save(fig,family,name,made); manifest[benchmark]={"representative_completed_pair":rep}


def layer_plots(layer, family, paired_name, growth_name, made):
    fig,ax=plt.subplots(figsize=(7.2,4.6)); x=layer.community
    ax.fill_between(x,layer.q25_error,layer.q75_error,color=COLORS[MODELS[1]],alpha=.22)
    ax.plot(x,layer.median_error_MM_minus_Micro,marker="o",color=COLORS[MODELS[1]])
    ax.axhline(0,color="black",lw=.8);ax.set(xlabel="community index",ylabel=r"paired $T_i^{MM}-T_i^{Micro}$");ax.grid(alpha=.25)
    save(fig,family,paired_name,made)
    fig,axes=plt.subplots(1,4,figsize=(16.5,4.2))
    axes[0].plot(x,layer.W1_arrival_completed,marker="o",color="#2878b5"); axes[0].set(title=r"$W_1(T_i)$",ylabel="time")
    axes[1].plot(x,layer.relative_W1_arrival_completed,marker="o",color="#2878b5"); axes[1].set(title=r"Relative $W_1(T_i)$",ylabel=r"$W_1/E[T_i^{Micro}]$")
    axes[2].plot(x,layer.mean_shift_completed_MM_minus_Micro,marker="o",color="#d0604c"); axes[2].axhline(0,color="black",lw=.8); axes[2].set(title=r"Mean shift in $T_i$")
    axes[3].plot(x,layer.median_shift_completed_MM_minus_Micro,marker="o",label=r"$\Delta q_{0.5}$",color="#4c8c5a"); axes[3].plot(x,layer.q90_shift_completed_MM_minus_Micro,marker="s",label=r"$\Delta q_{0.9}$",color="#7f5aa2"); axes[3].axhline(0,color="black",lw=.8);axes[3].set(title="Quantile shifts");axes[3].legend(frameon=False)
    for ax in axes: ax.set(xlabel="community index"); ax.grid(alpha=.25)
    save(fig,family,growth_name,made)


def clique_sweep_plot(summary, group, xcol, family, name, made):
    s=summary[summary.group.eq(group)].sort_values(xcol)
    fig,axes=plt.subplots(2,2,figsize=(9.2,6.3)); axes=axes.ravel(); x=s[xcol]
    axes[0].plot(x,s.W1_Tlast_completed,marker="o");axes[0].set_ylabel(r"completed-only $W_1(T_{last})$")
    axes[1].plot(x,s.relative_W1_Tlast_completed,marker="o");axes[1].set_ylabel(r"relative $W_1(T_{last})$")
    axes[2].plot(x,s.completion_Micro,marker="o",label="Micro");axes[2].plot(x,s.completion_MicroMacro,marker="s",label="MicroMacro");axes[2].set_ylabel("completion probability");axes[2].set_ylim(-.03,1.03);axes[2].legend(frameon=False)
    axes[3].plot(x,s.runtime_ratio_Micro_over_MicroMacro,marker="o",color="#4c8c5a");axes[3].axhline(1,color="black",lw=.8);axes[3].set_ylabel("runtime Micro / MicroMacro")
    xlabel = {"k": "interface width k", "L": "chain length L", "n": "community size n"}[xcol]
    for ax in axes: ax.set_xlabel(xlabel);ax.grid(alpha=.25)
    save(fig,family,name,made)


def random_sweep_plot(summary, group, xcol, family, name, made):
    s=summary[summary.group.eq(group)].sort_values(xcol); fig,axes=plt.subplots(2,2,figsize=(9.2,6.3));axes=axes.ravel();x=s[xcol]
    axes[0].plot(x,s.relative_W1_Tlast_completed,marker="o");axes[0].set_ylabel(r"relative $W_1(T_{last})$")
    axes[1].plot(x,s.mean_shift_completed_MM_minus_Micro,marker="o",label="mean");axes[1].plot(x,s.median_shift_completed_MM_minus_Micro,marker="s",label="median");axes[1].axhline(0,color="black",lw=.8);axes[1].set_ylabel("MicroMacro − Micro");axes[1].legend(frameon=False)
    axes[2].plot(x,s.completion_Micro,marker="o",label="Micro");axes[2].plot(x,s.completion_MicroMacro,marker="s",label="MicroMacro");axes[2].set_ylabel("completion probability");axes[2].set_ylim(-.03,1.03);axes[2].legend(frameon=False)
    axes[3].plot(x,s.runtime_ratio_Micro_over_MicroMacro,marker="o");axes[3].axhline(1,color="black",lw=.8);axes[3].set_ylabel("runtime Micro / MicroMacro")
    xlabel = "interface width k" if group == "interface" else "mean internal degree d"
    for ax in axes:ax.set_xlabel(xlabel);ax.grid(alpha=.25)
    save(fig,family,name,made)


def clique_density(observables, made):
    d=observables[observables.benchmark.eq("clique_leaf_n100_k2")]
    metrics=[("bridge_infection_time","bridge infection time"),("bridge_to_export_delay","bridge-to-export delay"),("total_export_time","total export time")]
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for ax,(col,title) in zip(axes,metrics):
        for m in MODELS:
            x=d[d.simulator.eq(m)][col].dropna().to_numpy(float)
            if len(x): ax.hist(x,bins=45,density=True,histtype="step",lw=1.4,color=COLORS[m],label=SHORT[m])
        ax.set(title=title,xlabel="time",ylabel="density");ax.grid(alpha=.2)
    axes[0].legend(frameon=False); save(fig,"clique_leaf","metric_density_panels.png",made)


def interface_dynamics_plot(
    runs: pd.DataFrame,
    arrivals: pd.DataFrame,
    scenarios: list[tuple[int, str]],
    family: str,
    filename: str,
    made: list[Path],
) -> None:
    """Plot aggregate dynamics and paired arrival error for an interface sweep."""
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(11, 2.7 * len(scenarios)))
    axes = np.atleast_2d(axes)
    for row, (k, benchmark) in enumerate(scenarios):
        t, stack = read_total_stack(benchmark, runs)
        ax = axes[row, 0]
        for model in MODELS:
            q = np.nanquantile(stack[model], [0.25, 0.5, 0.75], axis=0)
            ax.fill_between(t, q[0], q[2], color=COLORS[model], alpha=0.18)
            ax.plot(t, q[1], color=COLORS[model], label=SHORT[model])
        ax.set(title=f"k={k}: total infected", ylabel=r"$I(t)$")
        ax.grid(alpha=0.2)
        if row == 0:
            ax.legend(frameon=False)
        layer = paired_layer(arrivals, benchmark)
        ax = axes[row, 1]
        ax.fill_between(
            layer.community,
            layer.q25_error,
            layer.q75_error,
            color=COLORS[MODELS[1]],
            alpha=0.22,
        )
        ax.plot(
            layer.community,
            layer.median_error_MM_minus_Micro,
            marker="o",
            color=COLORS[MODELS[1]],
        )
        ax.axhline(0, color="black", lw=0.8)
        ax.set(title=f"k={k}: paired arrival error", ylabel="MM − Micro")
        ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("time")
    axes[-1, 1].set_xlabel("community index")
    save(fig, family, filename, made)


def main():
    if not DATA.exists(): raise SystemExit(f"Missing data directory: {DATA}")
    OUT.mkdir(parents=True,exist_ok=True)
    msg("[load] compact benchmark CSVs")
    runs=pd.read_csv(DATA/"runs.csv"); arrivals=pd.read_csv(DATA/"community_arrivals.csv"); obs=pd.read_csv(DATA/"benchmark_observables.csv"); runtime=pd.read_csv(DATA/"runtime.csv")
    made=[]; manifest={"data_root":str(DATA),"trajectories_csv_not_scanned":True,"figures":{},"representative_pairs":{}}
    clique_density(obs,made); msg("[plot] clique–bridge–leaf density panels")
    clique_meta=[("baseline",CLIQUE["baseline"],10,100,5,"baseline"),("interface",CLIQUE["k1"],10,100,1,"interface"),("interface",CLIQUE["k10"],10,100,10,"interface"),("interface",CLIQUE["k20"],10,100,20,"interface"),("length",CLIQUE["L5"],5,100,5,"length"),("length",CLIQUE["baseline"],10,100,5,"length"),("length",CLIQUE["L15"],15,100,5,"length"),("size",CLIQUE["n50"],10,50,5,"size"),("size",CLIQUE["baseline"],10,100,5,"size"),("size",CLIQUE["n200"],10,200,5,"size")]
    random_meta=[("interface",RANDOM["k1"],1,6),("interface",RANDOM["baseline"],5,6),("interface",RANDOM["k10"],10,6),("interface",RANDOM["k20"],20,6),("density",RANDOM["d4"],4,4),("density",RANDOM["baseline"],6,6),("density",RANDOM["d10"],10,10)]
    crows=[]
    for group,b,L,n,k,_ in clique_meta:
        crows.append(terminal_summary(runs,runtime,b,{"group":group,"L":L,"n":n,"k":k}))
    cs=pd.DataFrame(crows).drop_duplicates("benchmark");cs.to_csv(OUT/"chain_cliques_summary.csv",index=False)
    rrows=[]
    for group,b,x,dg in random_meta:rrows.append(terminal_summary(runs,runtime,b,{"group":group,"x":x,"degree":dg}))
    rs=pd.DataFrame(rrows).drop_duplicates("benchmark");rs.to_csv(OUT/"chain_random_connected_summary.csv",index=False)
    msg("[plot] chain-of-cliques baseline figures")
    t,stack=read_total_stack(CLIQUE["baseline"],runs); aggregate_plot(t,stack,"chain_cliques","calibrated_total_infected_iqr.png",made)
    representative_plot(CLIQUE["baseline"],runs,"chain_cliques","calibrated_per_community_pair.png",made,manifest["representative_pairs"])
    cdf_plot(runs,CLIQUE["baseline"],"chain_cliques","calibrated_tlast_cdf.png","clique_baseline_tlast_survival.png",made)
    layer=paired_layer(arrivals,CLIQUE["baseline"]);layer.to_csv(OUT/"chain_cliques_baseline_layer_metrics.csv",index=False);layer_plots(layer,"chain_cliques","calibrated_paired_arrival_error.png","calibrated_error_growth.png",made)
    clique_sweep_plot(cs,"interface","k","chain_cliques","calibrated_interface_summary.png",made);clique_sweep_plot(cs,"length","L","chain_cliques","calibrated_chain_length_summary.png",made);clique_sweep_plot(cs,"size","n","chain_cliques","calibrated_community_size_summary.png",made)
    msg("[plot] chain-of-cliques interface dynamics")
    dyn=[(1,CLIQUE["k1dyn"]),(5,CLIQUE["baseline"]),(10,CLIQUE["k10"]),(20,CLIQUE["k20"])]
    fig,axes=plt.subplots(4,2,figsize=(11,10.8))
    for row,(k,b) in enumerate(dyn):
        t,st=read_total_stack(b,runs); ax=axes[row,0]
        for m in MODELS:
            q=np.nanquantile(st[m],[.25,.5,.75],axis=0);ax.fill_between(t,q[0],q[2],color=COLORS[m],alpha=.18);ax.plot(t,q[1],color=COLORS[m],label=SHORT[m])
        ax.set(title=f"k={k}: total infected",ylabel=r"$I(t)$");ax.grid(alpha=.2)
        if row==0:ax.legend(frameon=False)
        lm=paired_layer(arrivals,b);ax=axes[row,1];ax.fill_between(lm.community,lm.q25_error,lm.q75_error,color=COLORS[MODELS[1]],alpha=.22);ax.plot(lm.community,lm.median_error_MM_minus_Micro,marker="o",color=COLORS[MODELS[1]]);ax.axhline(0,color="black",lw=.8);ax.set(title=f"k={k}: paired arrival error",ylabel="MM − Micro");ax.grid(alpha=.2)
    axes[-1,0].set_xlabel("time");axes[-1,1].set_xlabel("community index");save(fig,"chain_cliques","calibrated_interface_dynamics_by_k.png",made)
    msg("[plot] connected-random baseline figures")
    b=RANDOM["baseline"];t,stack=read_total_stack(b,runs);aggregate_plot(t,stack,"chain_random_connected","random_connected_total_infected_iqr.png",made)
    representative_plot(b,runs,"chain_random_connected","random_connected_per_community_pair.png",made,manifest["representative_pairs"])
    cdf_plot(runs,b,"chain_random_connected","random_connected_tlast_cdf.png","sparse_baseline_tlast_survival.png",made)
    layer=paired_layer(arrivals,b);layer.to_csv(OUT/"chain_random_connected_baseline_layer_metrics.csv",index=False);layer_plots(layer,"chain_random_connected","random_connected_paired_arrival_error.png","random_connected_error_growth.png",made)
    random_sweep_plot(rs,"interface","x","chain_random_connected","random_connected_interface_summary.png",made);random_sweep_plot(rs,"density","x","chain_random_connected","random_connected_density_summary.png",made)
    msg("[plot] connected-random interface dynamics")
    interface_dynamics_plot(
        runs,
        arrivals,
        [(1, RANDOM["k1"]), (5, RANDOM["baseline"]), (10, RANDOM["k10"]), (20, RANDOM["k20"])],
        "chain_random_connected",
        "calibrated_interface_dynamics_by_k.png",
        made,
    )
    expected=[PAPER/"clique_leaf"/"metric_density_panels.png"] + [PAPER/"chain_cliques"/x for x in ("calibrated_total_infected_iqr.png","calibrated_per_community_pair.png","calibrated_tlast_cdf.png","clique_baseline_tlast_survival.png","calibrated_paired_arrival_error.png","calibrated_error_growth.png","calibrated_interface_summary.png","calibrated_interface_dynamics_by_k.png","calibrated_chain_length_summary.png","calibrated_community_size_summary.png")] + [PAPER/"chain_random_connected"/x for x in ("random_connected_total_infected_iqr.png","random_connected_per_community_pair.png","random_connected_tlast_cdf.png","sparse_baseline_tlast_survival.png","random_connected_paired_arrival_error.png","random_connected_error_growth.png","random_connected_interface_summary.png","random_connected_density_summary.png","calibrated_interface_dynamics_by_k.png")]
    for p in expected:
        if not p.exists() or p.stat().st_size==0: raise RuntimeError(f"missing plot: {p}")
        with Image.open(p) as im: im.verify()
    manifest["targets"]=[str(p) for p in expected];manifest["derived_csvs"]=["chain_cliques_summary.csv","chain_random_connected_summary.csv","chain_cliques_baseline_layer_metrics.csv","chain_random_connected_baseline_layer_metrics.csv"]
    manifest["figures"]={str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "dimensions_px": Image.open(p).size} for p in expected}
    (OUT/"plot_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    msg(f"[done] validated {len(expected)} paper figure targets; outputs: {OUT}")

if __name__=="__main__": main()
