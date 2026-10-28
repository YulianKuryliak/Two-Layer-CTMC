#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures, json, random, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import run_production_factorial as core

ARMS=[
 ('current','shared','discard','normal'),
 ('cache_preserve','shared','preserve','normal'),
 ('independent_rng','independent','discard','normal'),
 ('current_single_entry','shared','discard','single_entry'),
 ('independent_single_entry','independent','discard','single_entry'),
]

def run_one(arg):
 rep,ns,seed,horizon,L,arm=arg
 name,r,c,q=arm
 # temporarily monkeypatch core.run_variant to L through local duplicate logic
 core.install_patch(r,c,q)
 from two_layer_ctmc.network import generate_two_scale_network
 from two_layer_ctmc.orchestrators.micromacro import Orchestrator
 import numpy as np, math
 micro,fg,W=generate_two_scale_network(n_communities=L,community_size=100,inter_links=5,seed=int(ns),macro_graph_type='chain',micro_graph_type='random_connected',edge_prob=6/99)
 init=int(list(micro[0].nodes())[0]);random.seed(seed);np.random.seed(seed)
 o=Orchestrator(W,micro,.003,0,.003,1.0,horizon,macro_T=1.0,model=2,full_graph=fg)
 if r=='independent':
  for i,m in enumerate(o.micro_models):m.rng=random.Random(seed+10000+1009*i)
 o.micro_models[0]._infect_node(init);o.event_log.append({'time':0.,'event_type':'seed','community':0,'node':init})
 ev=o.run()[3];arr=core.arrivals(ev,L,horizon);mac=[e for e in ev if e.get('mode')=='macro']
 reached={0};repeat=0
 for e in sorted(mac,key=lambda z:float(z['time'])):
  j=int(e.get('community',-1));
  if j in reached: repeat+=1
  elif j>=0: reached.add(j)
 return {'replication_id':rep,'network_seed':int(ns),'arm':name,'rng_mode':r,'cache_mode':c,'reimport_mode':q,'L':L,'terminal_T':arr[-1] if arr[-1] is not None else horizon,'terminal_observed':arr[-1] is not None,'macro_events':len(mac),'repeated_macro_events':repeat,'arrivals':arr}

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs-csv',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=20);p.add_argument('--L',type=int,default=5);p.add_argument('--workers',type=int,default=8);p.add_argument('--base-seed',type=int,default=1510000);p.add_argument('--horizon',type=float,default=4000)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);parts=a.out/'parts';parts.mkdir(exist_ok=True)
 rs=pd.read_csv(a.runs_csv);ns=(rs[rs.model_variant=='FULL_MICRO_REFERENCE'].drop_duplicates('replication_id').set_index('replication_id').network_seed.astype(int).to_dict())
 tasks=[]
 for rep in range(1,a.n+1):
  for arm in ARMS:
   path=parts/f'{rep:04d}_{arm[0]}.json'
   if not path.exists():tasks.append((rep,ns[rep],a.base_seed+rep-1,a.horizon,a.L,arm))
 t0=time.time();done=0
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
  futs={ex.submit(run_one,t):t for t in tasks}
  for f in concurrent.futures.as_completed(futs):
   row=f.result();(parts/f"{row['replication_id']:04d}_{row['arm']}.json").write_text(json.dumps(row),encoding='utf-8');done+=1
   if done%10==0 or done==len(tasks):print(f'{done}/{len(tasks)} new arms, elapsed {time.time()-t0:.1f}s',flush=True)
 rows=[]
 for fp in sorted(parts.glob('*.json')):
  r=json.loads(fp.read_text());
  if int(r['replication_id'])<=a.n: rows.append(r)
 expected=a.n*len(ARMS)
 if len(rows)!=expected: raise RuntimeError(f'expected {expected}, got {len(rows)}')
 pd.DataFrame([{k:v for k,v in r.items() if k!='arrivals'} for r in rows]).sort_values(['replication_id','arm']).to_csv(a.out/'runs.csv',index=False)
 arr=[]
 for r in rows:
  b={k:v for k,v in r.items() if k!='arrivals'}
  for layer,x in enumerate(r['arrivals']):arr.append({**b,'layer':layer,'T_i':x,'restricted_T_i':x if x is not None else a.horizon,'observed':x is not None})
 pd.DataFrame(arr).sort_values(['replication_id','arm','layer']).to_csv(a.out/'arrivals.csv',index=False)
 (a.out/'manifest.json').write_text(json.dumps({'n':a.n,'L':a.L,'horizon':a.horizon,'arms':[x[0] for x in ARMS],'elapsed_seconds':time.time()-t0,'note':'single-entry arms are mediation/sensitivity, not primary source attribution'},indent=2),encoding='utf-8')
if __name__=='__main__':main()
