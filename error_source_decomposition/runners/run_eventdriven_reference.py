#!/usr/bin/env python3
from __future__ import annotations
import argparse,concurrent.futures,json
from pathlib import Path
import pandas as pd
from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.experiments.hazard_placement_ablation.config import ExperimentConfig,ModelVariant,stream_seeds
from two_layer_ctmc.experiments.hazard_placement_ablation.simulator import HazardPlacementSimulator

def one(t):
 rep,ns,L,h=t;gs,fg,_=generate_two_scale_network(L,100,5,seed=int(ns),macro_graph_type='chain',micro_graph_type='random_connected',edge_prob=6/99);com=[list(g.nodes()) for g in gs];init=int(com[0][0]);cfg=ExperimentConfig(L=L,n=100,mean_internal_degree=6,k=5,beta_in=.003,beta_out=.003,gamma=0,T_end=h,base_seed=270000,initial_node=init);v=ModelVariant.AGGREGATE_HAZARD_UNIFORM_PLACEMENT;sim=HazardPlacementSimulator(fg,com,cfg,v,stream_seeds(cfg,rep,v),False,False,False);r=sim.run(init);a=r['arrivals'];return {'replication_id':rep,'network_seed':int(ns),'arm':'event_driven_A_U','L':L,'terminal_T':a[-1] if a[-1] is not None else h,'terminal_observed':a[-1] is not None,'arrivals':a}
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs-csv',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=50);p.add_argument('--L',type=int,default=5);p.add_argument('--workers',type=int,default=10);p.add_argument('--horizon',type=float,default=4000);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);parts=a.out/'parts';parts.mkdir(exist_ok=True);rs=pd.read_csv(a.runs_csv);ns=(rs[rs.model_variant=='FULL_MICRO_REFERENCE'].drop_duplicates('replication_id').set_index('replication_id').network_seed.astype(int).to_dict());tasks=[(r,ns[r],a.L,a.horizon) for r in range(1,a.n+1) if not (parts/f'{r:04d}.json').exists()]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
  fs=[ex.submit(one,t) for t in tasks]
  for i,f in enumerate(concurrent.futures.as_completed(fs),1):
   x=f.result();(parts/f"{x['replication_id']:04d}.json").write_text(json.dumps(x));
   if i%10==0 or i==len(fs):print(i,'/',len(fs),flush=True)
 rows=[json.loads(p.read_text()) for p in sorted(parts.glob('*.json')) if int(p.stem)<=a.n];pd.DataFrame([{k:v for k,v in r.items() if k!='arrivals'} for r in rows]).to_csv(a.out/'runs.csv',index=False)
if __name__=='__main__':main()
