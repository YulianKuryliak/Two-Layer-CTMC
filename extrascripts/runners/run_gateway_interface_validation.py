"""Experiment 06 out-of-sample validation on near/random/far gateway architectures."""
from __future__ import annotations
import concurrent.futures,json,random,sys
from pathlib import Path
import networkx as nx,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.experiments.hazard_placement_ablation.config import ExperimentConfig,ModelVariant,PlacementMode,stream_seeds
from two_layer_ctmc.experiments.hazard_placement_ablation.simulator import HazardPlacementSimulator
from extrascripts.runners.run_mechanistic_error_decomposition import SourceOracleUniform,SourceTargetMarginalUniform
MODES=('near','random','far')
class SourceBoundary(SourceOracleUniform):
    def __init__(self,*a,**k): super().__init__(*a,**k); self.placement_mode=PlacementMode.BOUNDARY_WEIGHTED
class SDBoundary(SourceTargetMarginalUniform):
    def __init__(self,*a,**k): super().__init__(*a,**k); self.placement_mode=PlacementMode.BOUNDARY_WEIGHTED
def choose_gateways(graph,nodes,k,mode,seed):
    rng=random.Random(seed); left=sorted(rng.sample(list(nodes),k)); dist=dict(nx.multi_source_dijkstra_path_length(graph,left,weight=None)); right=[]; tie=random.Random(seed+{'random':11,'near':23,'far':37}[mode])
    for degree in sorted(graph.degree(n) for n in left):
        cand=[n for n in nodes if n not in right and graph.degree(n)==degree]
        if mode=='random': chosen=tie.choice(cand)
        else:
            ranked=sorted(cand,key=lambda n:(dist[n],tie.random())); chosen=ranked[0] if mode=='near' else ranked[-1]
        right.append(chosen)
    pair=[nx.shortest_path_length(graph,u,v) for u in left for v in right]
    return left,sorted(right),float(min(pair)),float(np.mean(pair))
def build(gs,mode,gseed,k=5):
    L=len(gs); comm=[sorted(g.nodes()) for g in gs]; left=[[] for _ in range(L)]; right=[[] for _ in range(L)]; ds=[]
    right[0]=sorted(random.Random(gseed+1).sample(comm[0],k)); left[-1]=sorted(random.Random(gseed+2).sample(comm[-1],k))
    for c in range(1,L-1): l,r,dmin,dmean=choose_gateways(gs[c],comm[c],k,mode,gseed+1000*c); left[c]=l; right[c]=r; ds.append((dmin,dmean))
    fg=nx.Graph()
    for g in gs: fg.add_nodes_from(g.nodes(data=True)); fg.add_edges_from(g.edges(data=True))
    for c in range(L-1):
        ls=list(right[c]); rs=list(left[c+1]); random.Random(gseed+10000+c).shuffle(rs); fg.add_edges_from((u,v,{'weight':1.0}) for u,v in zip(ls,rs))
    return fg,{'D_min':float(np.mean([x[0] for x in ds])),'D_mean':float(np.mean([x[1] for x in ds]))}
def atom(p,x): p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(x),encoding='utf-8'); q.replace(p)
def run_one(rep):
    base=731000; nseed=base+rep-1; gseed=base+100000+rep-1; gs,_,_=generate_two_scale_network(5,100,0,seed=nseed,macro_graph_type='chain',micro_graph_type='random_connected',edge_prob=6/99); comm=[list(g.nodes()) for g in gs]; c=ExperimentConfig(experiment_name='gateway_interface_validation',L=5,n=100,mean_internal_degree=6,k=5,beta_in=.003,beta_out=.003,gamma=0,T_end=2000,base_seed=731000,initial_node=0); rows=[]; desc={}
    for mode in MODES:
        fg,d=build(gs,mode,gseed); desc[mode]=d
        for name,cls,var in [('E_E',HazardPlacementSimulator,ModelVariant.ORACLE_EXACT_HAZARD_EXACT_PLACEMENT),('A_U',HazardPlacementSimulator,ModelVariant.AGGREGATE_HAZARD_UNIFORM_PLACEMENT),('S_B',SourceBoundary,ModelVariant.ORACLE_EXACT_HAZARD_UNIFORM_PLACEMENT),('SD_B',SDBoundary,ModelVariant.ORACLE_EXACT_HAZARD_UNIFORM_PLACEMENT)]:
            # Same stream seed map within replication/mode to reduce incidental RNG differences in this validation.
            sim=cls(fg,comm,c,var,stream_seeds(c,rep,ModelVariant.ORACLE_EXACT_HAZARD_UNIFORM_PLACEMENT),False,False,False); r=sim.run(0)
            rows += [{'replication_id':rep,'mode':mode,'model':name,'layer':i,'T_i':v,'observed':v is not None,'restricted_T_i':float(v) if v is not None else 2000.0} for i,v in enumerate(r['arrivals'])]
    return rep,rows,desc
def ci(x,seed,B=4000):
    x=np.asarray(x,float); rng=np.random.default_rng(seed); z=np.empty(B)
    for b in range(B): idx=rng.integers(0,len(x),len(x)); z[b]=x[idx].mean()
    return np.quantile(z,[.025,.975])
def main():
    out=ROOT/'output/mechanistic_error_decomposition/gateway_validation_100'; reps=out/'replications'; reps.mkdir(parents=True,exist_ok=True); pending=[r for r in range(1,101) if not (reps/f'rep_{r:04d}.json').exists()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as p:
        fs={p.submit(run_one,r):r for r in pending}
        for k,f in enumerate(concurrent.futures.as_completed(fs),1): rep,rows,desc=f.result(); atom(reps/f'rep_{rep:04d}.json',{'replication_id':rep,'rows':rows,'descriptors':desc}); print(f'[gateway] {k}/{len(pending)}',flush=True) if k%10==0 or k==len(pending) else None
    rows=[]; descs=[]
    for q in sorted(reps.glob('rep_*.json')): x=json.loads(q.read_text()); rows+=x['rows']; descs.append({'replication_id':x['replication_id'],**{f'{m}_{k}':v for m,d in x['descriptors'].items() for k,v in d.items()}})
    f=pd.DataFrame(rows); f.to_csv(out/'arrivals.csv',index=False); pd.DataFrame(descs).to_csv(out/'gateway_descriptors.csv',index=False); term=f[f.layer==4]; res=[]
    for mode in MODES:
        pv=term[term['mode']==mode].pivot(index='replication_id',columns='model',values='restricted_T_i'); base=abs((pv.A_U-pv.E_E).mean())
        for m in ['A_U','S_B','SD_B']:
            d=pv[m]-pv.E_E; lo,hi=ci(d,20260811+sum(map(ord,mode+m))); res.append({'mode':mode,'model':m,'RMST':pv[m].mean(),'reference_RMST_EE':pv.E_E.mean(),'delta_RMST_vs_EE':d.mean(),'ci95_low':lo,'ci95_high':hi,'gap_closure_vs_AU':(base-abs(d.mean()))/base if base else np.nan,'n_runs':len(d)})
    o=pd.DataFrame(res); o.to_csv(out/'summary.csv',index=False); atom(out/'manifest.json',{'experiment':'gateway_interface_validation','complete':True,'n_runs':100,'modes':MODES,'L':5,'n':100,'d':6,'k':5,'T_end':2000}); print(o.to_string(index=False))
if __name__=='__main__': main()
