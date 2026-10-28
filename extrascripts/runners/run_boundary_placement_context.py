"""Experiment 06 supplement: boundary-weighted placement under source-aware hazards."""
from __future__ import annotations
import concurrent.futures,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.experiments.hazard_placement_ablation.config import ExperimentConfig,ModelVariant,PlacementMode,stream_seeds
from extrascripts.runners.run_mechanistic_error_decomposition import SourceOracleUniform,SourceTargetMarginalUniform

def atom(p,x): p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(x),encoding='utf-8'); q.replace(p)
class SourceBoundary(SourceOracleUniform):
    def __init__(self,*a,**k): super().__init__(*a,**k); self.placement_mode=PlacementMode.BOUNDARY_WEIGHTED
class SDBoundary(SourceTargetMarginalUniform):
    def __init__(self,*a,**k): super().__init__(*a,**k); self.placement_mode=PlacementMode.BOUNDARY_WEIGHTED
def run_one(t):
    rep,seed,initial,h=t; gs,fg,_=generate_two_scale_network(10,100,5,seed=seed,macro_graph_type='chain',micro_graph_type='random_connected',edge_prob=6/99); comm=[list(g.nodes()) for g in gs]
    c=ExperimentConfig(experiment_name='experiment_06_boundary_placement',L=10,n=100,mean_internal_degree=6,k=5,beta_in=.003,beta_out=.003,gamma=0,T_end=h,base_seed=270000,initial_node=initial)
    rows=[]; var=ModelVariant.ORACLE_EXACT_HAZARD_UNIFORM_PLACEMENT; seedvar=ModelVariant.AGGREGATE_HAZARD_BOUNDARY_PLACEMENT
    for name,cls in [('S_oracle_B',SourceBoundary),('SD_marginal_B',SDBoundary)]:
        sim=cls(fg,comm,c,var,stream_seeds(c,rep,seedvar),False,False,False); r=sim.run(initial)
        rows += [{'replication_id':rep,'model':name,'layer':i,'T_i':v,'observed':v is not None,'restricted_T_i':float(v) if v is not None else h} for i,v in enumerate(r['arrivals'])]
    return rep,rows
def ci(x,seed,B=5000):
    x=np.asarray(x,float); rng=np.random.default_rng(seed); z=np.empty(B)
    for b in range(B): idx=rng.integers(0,len(x),len(x)); z[b]=x[idx].mean()
    return np.quantile(z,[.025,.975])
def main():
    src=ROOT/'output/ablation_hazard_placement/final_100'; main=ROOT/'output/mechanistic_error_decomposition/final_100'; out=ROOT/'output/mechanistic_error_decomposition/boundary_placement_100'; reps=out/'replications'; reps.mkdir(parents=True,exist_ok=True); rr=pd.read_csv(src/'runs.csv'); ref=rr[rr.model_variant=='FULL_MICRO_REFERENCE'].set_index('replication_id'); h=4000.
    tasks=[(r,int(ref.loc[r,'network_seed']),int(ref.loc[r,'initial_node']),h) for r in range(1,101)]; pending=[t for t in tasks if not (reps/f'rep_{t[0]:04d}.json').exists()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as p:
        fs={p.submit(run_one,t):t[0] for t in pending}
        for k,f in enumerate(concurrent.futures.as_completed(fs),1): rep,rows=f.result(); atom(reps/f'rep_{rep:04d}.json',{'replication_id':rep,'rows':rows}); print(f'[boundary] {k}/{len(pending)}',flush=True) if k%10==0 or k==len(pending) else None
    rows=[]
    for q in sorted(reps.glob('rep_*.json')): rows += json.loads(q.read_text())['rows']
    f=pd.DataFrame(rows); f.to_csv(out/'arrivals.csv',index=False); b=f[f.layer==9].pivot(index='replication_id',columns='model',values='restricted_T_i'); m=pd.read_csv(main/'arrivals.csv'); mt=m[m.layer==9].pivot(index='replication_id',columns='model',values='restricted_T_i')
    combos=[('S_oracle_U','S_oracle_B','boundary_placement_given_source'),('SD_marginal_U','SD_marginal_B','boundary_placement_given_SD'),('FullMicro','S_oracle_B','source_plus_boundary_vs_FM'),('FullMicro','SD_marginal_B','SD_plus_boundary_vs_FM')]
    cs=[]
    for l,r,label in combos:
        lv=mt[l] if l in mt else b[l]; rv=b[r] if r in b else mt[r]; d=rv-lv; lo,hi=ci(d,20260811+sum(map(ord,label))); cs.append({'contrast':label,'left':l,'right':r,'delta_RMST_right_minus_left':d.mean(),'ci95_low':lo,'ci95_high':hi})
    o=pd.DataFrame(cs); o.to_csv(out/'contrasts.csv',index=False); atom(out/'manifest.json',{'experiment':'boundary_placement_context','complete':True,'n_runs':100}); print(o.to_string(index=False))
if __name__=='__main__': main()
