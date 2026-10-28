#!/usr/bin/env python3
"""Hierarchical attribution of structural approximation error from Experiment 06.

Uses the 100 paired terminal RMST observations for A_U, S_oracle_U, D_oracle_U,
SD_marginal_U, E_U and E_E. Source and destination boundary effects form a
complete 2x2 factorial and are attributed by two-player Shapley values.
The edge-pair correction is the conditional increment SD->E_U. Placement is the
conditional increment E_U->E_E. This hierarchy is intentional: exact
edge-conditioned placement is not generally defined under a spurious aggregate
macro event, so a symmetric hazard x placement 2x2 factorial would require an
artificial null-event convention and would not isolate the intended mechanisms.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = ["A_U","S_oracle_U","D_oracle_U","SD_marginal_U","E_U","E_E","FullMicro"]

def paired_terminal(path: Path) -> pd.DataFrame:
    x=pd.read_csv(path)
    layer=int(x.layer.max())
    t=x[(x.layer==layer)&x.model.isin(MODELS)].pivot(index='replication_id',columns='model',values='restricted_T_i')
    missing=[m for m in MODELS if m not in t.columns]
    if missing: raise ValueError(f"missing models: {missing}")
    if t[MODELS].isna().any().any(): raise ValueError("unpaired terminal observations")
    return t[MODELS].sort_index()

def components(t: pd.DataFrame) -> pd.DataFrame:
    out=pd.DataFrame(index=t.index)
    # Complete 2x2 factorial for source and destination boundary marginals.
    out['source_boundary'] = 0.5*((t.S_oracle_U-t.A_U)+(t.SD_marginal_U-t.D_oracle_U))
    out['destination_boundary'] = 0.5*((t.D_oracle_U-t.A_U)+(t.SD_marginal_U-t.S_oracle_U))
    out['source_destination_interaction'] = t.SD_marginal_U-t.S_oracle_U-t.D_oracle_U+t.A_U
    # Conditional refinements.
    out['edge_pair_joint'] = t.E_U-t.SD_marginal_U
    out['placement'] = t.E_E-t.E_U
    out['hazard_total'] = t.E_U-t.A_U
    out['structural_total'] = t.E_E-t.A_U
    out['full_vs_EE_residual'] = t.FullMicro-t.E_E
    # Identity check: Shapley(source,dest)+edge+placement == structural total.
    out['reconstruction'] = out.source_boundary+out.destination_boundary+out.edge_pair_joint+out.placement
    out['reconstruction_error'] = out.reconstruction-out.structural_total
    return out

def bootstrap(t: pd.DataFrame, B:int, seed:int):
    rng=np.random.default_rng(seed); n=len(t)
    c=components(t)
    keys=["source_boundary","destination_boundary","edge_pair_joint","placement","structural_total","hazard_total","full_vs_EE_residual","source_destination_interaction"]
    arr={k:c[k].to_numpy(float) for k in keys}
    out={k:np.empty(B,float) for k in keys}
    for begin in range(0,B,1000):
        end=min(B,begin+1000); idx=rng.integers(0,n,size=(end-begin,n))
        for k in keys: out[k][begin:end]=arr[k][idx].mean(axis=1)
    data={"draw":np.arange(B)}
    data.update(out)
    for k in ["source_boundary","destination_boundary","edge_pair_joint","placement"]:
        data[k+"_pct"]=100*out[k]/out["structural_total"]
    return pd.DataFrame(data)

def ci(s): return np.quantile(np.asarray(s,float),[.025,.975])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--arrivals',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=20000);p.add_argument('--seed',type=int,default=260814)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    t=paired_terminal(a.arrivals); c=components(t)
    pd.concat([t,c],axis=1).to_csv(a.out/'structural_attribution_per_rep.csv')
    boot=bootstrap(t,a.bootstrap,a.seed); boot.to_csv(a.out/'bootstrap_draws.csv',index=False)
    names=['source_boundary','destination_boundary','edge_pair_joint','placement']
    total=float(c.structural_total.mean())
    summary=[]
    for name in names:
        v=float(c[name].mean()); lo,hi=ci(boot[name]); plo,phi=ci(boot[name+'_pct'])
        summary.append({'component':name,'rmst_contribution':v,'ci95_low':lo,'ci95_high':hi,'percent_of_directional_structural_gap':100*v/total,'percent_ci95_low':plo,'percent_ci95_high':phi})
    for name in ['hazard_total','structural_total','full_vs_EE_residual','source_destination_interaction']:
        lo,hi=ci(boot[name]); summary.append({'component':name,'rmst_contribution':float(c[name].mean()),'ci95_low':lo,'ci95_high':hi,'percent_of_directional_structural_gap':(100*float(c[name].mean())/total if name!='full_vs_EE_residual' else np.nan),'percent_ci95_low':np.nan,'percent_ci95_high':np.nan})
    pd.DataFrame(summary).to_csv(a.out/'structural_attribution_summary.csv',index=False)
    model_summary=[]
    for m in MODELS: model_summary.append({'model':m,'RMST':float(t[m].mean()),'completion':float((t[m]<4000).mean())})
    pd.DataFrame(model_summary).to_csv(a.out/'model_summary.csv',index=False)
    meta={'n_replications':len(t),'terminal_layer':9,'horizon':4000,'bootstrap_reps':a.bootstrap,'seed':a.seed,'identity_max_abs_error':float(c.reconstruction_error.abs().max()),'method':'hierarchical Shapley: complete source x destination 2x2 factorial; conditional edge-pair increment; conditional placement increment','denominator':'E_E - A_U directional RMST gap'}
    (a.out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__': main()
