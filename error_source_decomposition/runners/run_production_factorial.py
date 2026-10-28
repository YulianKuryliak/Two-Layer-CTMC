#!/usr/bin/env python3
"""Production implementation decomposition: RNG architecture x probe cache x re-import mediation.

Primary implementation factors:
 R: shared global micro RNG (current) vs independent per-community RNG streams.
 C: midpoint probes discard the already sampled next-event cache (current) vs preserve it.
Mediation/sensitivity factor:
 Q: normal re-imports vs single-entry (no macro import into an already infected community).

The single-entry factor is NOT interpreted as a primary error source; it is used to test
whether implementation changes act through repeated-import amplification.
"""
from __future__ import annotations
import argparse, concurrent.futures, json, math, random, time
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx
from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.orchestrators.micromacro import Orchestrator
from two_layer_ctmc.engines.micro_model import MicroModel
from two_layer_ctmc.engines.macro_engine import MacroEngine

VARIANTS=[]
for rng_mode in ('shared','independent'):
  for cache_mode in ('discard','preserve'):
    for reimport in ('normal','single_entry'):
      VARIANTS.append((rng_mode,cache_mode,reimport))

ORIG_CLONE=MicroModel.clone; ORIG_REFRESH=MicroModel.refresh_from; ORIG_HAZ=MacroEngine._compute_hazards_matrix

def install_patch(rng_mode:str,cache_mode:str,reimport:str):
    def copy_rng(r):
        if r is random: return random
        rr=random.Random(); rr.setstate(r.getstate()); return rr
    def clone(self):
        rr = self.rng if rng_mode=='shared' else copy_rng(self.rng)
        m=MicroModel(self.infection_rate,self.recovery_rate,self.model,rng=rr)
        m.current_time=self.current_time
        m.G=nx.Graph(); m.G.add_nodes_from(self.G.nodes(data=True))
        m.infected_nodes=list(self.infected_nodes);m.total_infection_rate=self.total_infection_rate;m.total_recovery_rate=self.total_recovery_rate
        m.S_count=self.S_count;m.I_count=self.I_count;m.R_count=self.R_count;m._neighbors=self._neighbors
        if cache_mode=='preserve':
            m._cached_wait=self._cached_wait;m._cached_event=self._cached_event;m._cached_rng_before=self._cached_rng_before;m._cached_rng_after=self._cached_rng_after
        else: m._invalidate_cache()
        return m
    def refresh(self,src):
        self.current_time=src.current_time;self.infected_nodes=list(src.infected_nodes);self.total_infection_rate=src.total_infection_rate;self.total_recovery_rate=src.total_recovery_rate
        self.S_count=src.S_count;self.I_count=src.I_count;self.R_count=src.R_count;self._neighbors=src._neighbors
        for node in self.G.nodes():
            d=self.G.nodes[node]; s=src.G.nodes[node];d['infected']=s['infected'];d['recovered']=s['recovered'];d['sum_of_weights_i']=s['sum_of_weights_i']
        if rng_mode=='independent': self.rng.setstate(src.rng.getstate())
        if cache_mode=='preserve':
            self._cached_wait=src._cached_wait;self._cached_event=src._cached_event;self._cached_rng_before=src._cached_rng_before;self._cached_rng_after=src._cached_rng_after
        else: self._invalidate_cache()
    MicroModel.clone=clone; MicroModel.refresh_from=refresh
    if reimport=='single_entry':
        def haz(self,I_counts,S_counts):
            a=ORIG_HAZ(self,I_counts,S_counts).copy()
            infected=np.asarray(I_counts,dtype=float)>0
            a[:,infected]=0.0
            return a
        MacroEngine._compute_hazards_matrix=haz
    else: MacroEngine._compute_hazards_matrix=ORIG_HAZ

def arrivals(ev,L,horizon):
    a=[None]*L;a[0]=0.0
    for e in sorted(ev,key=lambda z:float(z.get('time',math.inf))):
        if str(e.get('event_type','')).lower() not in {'seed','infection','transfer'}: continue
        c=e.get('community')
        if c is not None and 0<=int(c)<L and a[int(c)] is None: a[int(c)]=float(e['time'])
    return a

def run_variant(rep,network_seed,seed,horizon,rng_mode,cache_mode,reimport):
    install_patch(rng_mode,cache_mode,reimport)
    L=10
    micro_graphs,full_graph,W=generate_two_scale_network(n_communities=L,community_size=100,inter_links=5,seed=int(network_seed),macro_graph_type='chain',micro_graph_type='random_connected',edge_prob=6.0/99.0)
    communities=[list(g.nodes()) for g in micro_graphs]; initial_node=int(communities[0][0])
    random.seed(seed);np.random.seed(seed)
    o=Orchestrator(W,micro_graphs,0.003,0.0,0.003,1.0,horizon,macro_T=1.0,model=2,full_graph=full_graph)
    if rng_mode=='independent':
        for i,m in enumerate(o.micro_models): m.rng=random.Random(seed+10_000+1009*i)
    o.micro_models[0]._infect_node(initial_node);o.event_log.append({'time':0.0,'event_type':'seed','community':0,'node':initial_node})
    ev=o.run()[3]; arr=arrivals(ev,L,horizon)
    macro=[e for e in ev if e.get('mode')=='macro']
    # repeated macro imports: macro transfers after destination had already been first reached.
    first={0:0.0}; repeated=0; null_imports=0
    for e in sorted(macro,key=lambda z:float(z['time'])):
        j=int(e.get('community',-1)); node=e.get('node')
        if j in first: repeated+=1
        elif j>=0: first[j]=float(e['time'])
        if node is None: null_imports+=1
    return {'replication_id':rep,'network_seed':int(network_seed),'rng_mode':rng_mode,'cache_mode':cache_mode,'reimport_mode':reimport,'model':f'prod_{rng_mode}_{cache_mode}_{reimport}','terminal_T':arr[-1] if arr[-1] is not None else horizon,'terminal_observed':arr[-1] is not None,'macro_events':len(macro),'repeated_macro_events':repeated,'null_imports':null_imports,'arrivals':arr}

def worker(task):
    rep,ns,base,horizon=task; rows=[]
    for vi,(r,c,q) in enumerate(VARIANTS):
        # Same base seed per replication across variants; independent architecture intentionally changes stream routing.
        rows.append(run_variant(rep,ns,base+rep-1,horizon,r,c,q))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs-csv',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=100);p.add_argument('--workers',type=int,default=12);p.add_argument('--base-seed',type=int,default=1510000);p.add_argument('--horizon',type=float,default=4000.0)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rs=pd.read_csv(a.runs_csv);ns=(rs[rs.model_variant=='FULL_MICRO_REFERENCE'].drop_duplicates('replication_id').set_index('replication_id').network_seed.astype(int).to_dict())
    tasks=[(rep,ns[rep],a.base_seed,a.horizon) for rep in range(1,a.n+1)]
    allrows=[];t0=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        fs={ex.submit(worker,t):t[0] for t in tasks}
        for k,f in enumerate(concurrent.futures.as_completed(fs),1):
            allrows.extend(f.result())
            if k%5==0 or k==a.n: print(f'{k}/{a.n} reps elapsed={time.time()-t0:.1f}s',flush=True)
    flat=[]
    for r in allrows:
        base={k:v for k,v in r.items() if k!='arrivals'}
        for layer,val in enumerate(r['arrivals']): flat.append({**base,'layer':layer,'T_i':val,'restricted_T_i':val if val is not None else a.horizon,'observed':val is not None})
    pd.DataFrame(flat).sort_values(['replication_id','model','layer']).to_csv(a.out/'arrivals.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!='arrivals'} for r in allrows]).sort_values(['replication_id','model']).to_csv(a.out/'runs.csv',index=False)
    manifest={'n_replications':a.n,'base_seed':a.base_seed,'horizon':a.horizon,'factors':{'rng_mode':['shared','independent'],'cache_mode':['discard','preserve'],'reimport_mode':['normal','single_entry']},'interpretation':'reimport factor is mediation/sensitivity intervention, not primary implementation source','elapsed_seconds':time.time()-t0}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__':main()
