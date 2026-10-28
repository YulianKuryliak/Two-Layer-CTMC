#!/usr/bin/env python3
from __future__ import annotations
import argparse,concurrent.futures,json
from pathlib import Path
import pandas as pd
from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.experiments.hazard_placement_ablation.config import ExperimentConfig,ModelVariant,stream_seeds
from two_layer_ctmc.experiments.hazard_placement_ablation.simulator import HazardPlacementSimulator
FACTORS=(0.85,)
def one(t):
 rep,ns,init,h=t;gs,fg,_=generate_two_scale_network(10,100,5,seed=int(ns),macro_graph_type='chain',micro_graph_type='random_connected',edge_prob=6/99);com=[list(g.nodes()) for g in gs];out=[]
 for c in FACTORS:
  cfg=ExperimentConfig(L=10,n=100,mean_internal_degree=6,k=5,beta_in=.003,beta_out=.003*c,gamma=0,T_end=h,base_seed=270000,initial_node=int(init));v=ModelVariant.AGGREGATE_HAZARD_UNIFORM_PLACEMENT;res=HazardPlacementSimulator(fg,com,cfg,v,stream_seeds(cfg,rep,v),False,False,False).run(int(init));x=res['arrivals'][-1];out.append({'replication_id':rep,'network_seed':int(ns),'calibration_factor':c,'T_i':x,'restricted_T_i':x if x is not None else h,'observed':x is not None})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs-csv',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=100);p.add_argument('--workers',type=int,default=10);p.add_argument('--horizon',type=float,default=4000);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);parts=a.out/'parts';parts.mkdir(exist_ok=True);rs=pd.read_csv(a.runs_csv);ref=rs[rs.model_variant=='FULL_MICRO_REFERENCE'].drop_duplicates('replication_id').set_index('replication_id');tasks=[(r,int(ref.loc[r,'network_seed']),int(ref.loc[r,'initial_node']),a.horizon) for r in range(1,a.n+1) if not (parts/f'{r:04d}.json').exists()]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
  fs=[ex.submit(one,t) for t in tasks]
  for i,f in enumerate(concurrent.futures.as_completed(fs),1):rows=f.result();rep=rows[0]['replication_id'];(parts/f'{rep:04d}.json').write_text(json.dumps(rows));print(i,'/',len(fs),flush=True) if i%10==0 or i==len(fs) else None
 rows=[]
 for q in sorted(parts.glob('*.json')):
  rr=json.loads(q.read_text());
  if rr and int(rr[0]['replication_id'])<=a.n:rows.extend(rr)
 pd.DataFrame(rows).sort_values(['replication_id','calibration_factor']).to_csv(a.out/'calibration_runs.csv',index=False)
if __name__=='__main__':main()
