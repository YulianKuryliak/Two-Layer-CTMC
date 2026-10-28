"""Experiment 06 supplement: suppress all further imports after a community's first infection."""
from __future__ import annotations
import concurrent.futures,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.experiments.hazard_placement_ablation.config import ExperimentConfig,ModelVariant,stream_seeds
from two_layer_ctmc.experiments.hazard_placement_ablation.simulator import HazardPlacementSimulator

def atom(p,x): p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(x),encoding='utf-8'); q.replace(p)
class SingleEntryAggregate(HazardPlacementSimulator):
    def aggregate_pair_hazards(self,state):
        h=super().aggregate_pair_hazards(state); counts=self._community_counts(state)
        return {pair:(0.0 if counts[pair[1]][1]>0 else v) for pair,v in h.items()}
class SingleEntryExact(HazardPlacementSimulator):
    def exact_pair_hazards(self,state):
        h=super().exact_pair_hazards(state); counts=self._community_counts(state)
        return {pair:(0.0 if counts[pair[1]][1]>0 else v) for pair,v in h.items()}
def run_one(t):
    rep,seed,initial,h=t; gs,fg,_=generate_two_scale_network(10,100,5,seed=seed,macro_graph_type='chain',micro_graph_type='random_connected',edge_prob=6/99); comm=[list(g.nodes()) for g in gs]
    c=ExperimentConfig(experiment_name='experiment_06_reimport',L=10,n=100,mean_internal_degree=6,k=5,beta_in=.003,beta_out=.003,gamma=0,T_end=h,base_seed=270000,initial_node=initial)
    rows=[]
    for name,cls,var in [('A_U_single_entry',SingleEntryAggregate,ModelVariant.AGGREGATE_HAZARD_UNIFORM_PLACEMENT),('E_U_single_entry',SingleEntryExact,ModelVariant.ORACLE_EXACT_HAZARD_UNIFORM_PLACEMENT)]:
        sim=cls(fg,comm,c,var,stream_seeds(c,rep,var),False,False,False); r=sim.run(initial)
        rows += [{'replication_id':rep,'model':name,'layer':i,'T_i':v,'observed':v is not None,'restricted_T_i':float(v) if v is not None else h} for i,v in enumerate(r['arrivals'])]
    return rep,rows
def ci(x,seed,B=5000):
    x=np.asarray(x,float); rng=np.random.default_rng(seed); z=np.empty(B)
    for b in range(B): idx=rng.integers(0,len(x),len(x)); z[b]=x[idx].mean()
    return np.quantile(z,[.025,.975])
def main():
    src=ROOT/'output/ablation_hazard_placement/final_100'; out=ROOT/'output/mechanistic_error_decomposition/reimport_100'; reps=out/'replications'; reps.mkdir(parents=True,exist_ok=True); rr=pd.read_csv(src/'runs.csv'); ref=rr[rr.model_variant=='FULL_MICRO_REFERENCE'].set_index('replication_id'); h=4000.
    tasks=[(r,int(ref.loc[r,'network_seed']),int(ref.loc[r,'initial_node']),h) for r in range(1,101)]; pending=[t for t in tasks if not (reps/f'rep_{t[0]:04d}.json').exists()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as p:
        fs={p.submit(run_one,t):t[0] for t in pending}
        for k,f in enumerate(concurrent.futures.as_completed(fs),1): rep,rows=f.result(); atom(reps/f'rep_{rep:04d}.json',{'replication_id':rep,'rows':rows}); print(f'[reimport] {k}/{len(pending)}',flush=True) if k%10==0 or k==len(pending) else None
    rows=[]
    for q in sorted(reps.glob('rep_*.json')): rows += json.loads(q.read_text())['rows']
    f=pd.DataFrame(rows); f.to_csv(out/'arrivals.csv',index=False); term=f[f.layer==9]; pv=term.pivot(index='replication_id',columns='model',values='restricted_T_i')
    gap=pv.A_U_single_entry-pv.E_U_single_entry; lo,hi=ci(gap,20260811)
    # normal archived gap with same arms
    a=pd.read_csv(src/'arrival_times.csv'); t=a[a.layer_index==9].pivot(index='replication_id',columns='model_variant',values='T_i').fillna(h); norm=t.AGGREGATE_HAZARD_UNIFORM_PLACEMENT-t.ORACLE_EXACT_HAZARD_UNIFORM_PLACEMENT; nlo,nhi=ci(norm,20260812)
    change=gap-norm; clo,chi=ci(change,20260813)
    outdf=pd.DataFrame([{'contrast':'aggregate_minus_exact_normal','delta_RMST':norm.mean(),'ci95_low':nlo,'ci95_high':nhi},{'contrast':'aggregate_minus_exact_single_entry','delta_RMST':gap.mean(),'ci95_low':lo,'ci95_high':hi},{'contrast':'change_in_hazard_gap_when_reimports_removed','delta_RMST':change.mean(),'ci95_low':clo,'ci95_high':chi}]); outdf.to_csv(out/'contrasts.csv',index=False); atom(out/'manifest.json',{'experiment':'reimport_ablation','complete':True,'n_runs':100,'policy':'macro hazards into any already-infected target community are zero'}); print(outdf.to_string(index=False))
if __name__=='__main__': main()
