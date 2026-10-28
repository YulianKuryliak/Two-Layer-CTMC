"""Experiment 06: paired mechanistic source/target/edge/placement decomposition."""
from __future__ import annotations
import concurrent.futures, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.experiments.hazard_placement_ablation.config import ExperimentConfig, ModelVariant, stream_seeds
from two_layer_ctmc.experiments.hazard_placement_ablation.simulator import HazardPlacementSimulator, INFECTED, SUSCEPTIBLE

def atom(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); q=path.with_suffix(path.suffix+'.tmp'); q.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8'); q.replace(path)

def rows_for(rep,model,arr,h):
    return [{'replication_id':rep,'scenario':'mechanistic_error_decomposition','model':model,'layer':i,'T_i':v,'observed':v is not None,'restricted_T_i':float(v) if v is not None else h} for i,v in enumerate(arr)]

class SourceOracleUniform(HazardPlacementSimulator):
    def exact_pair_hazards(self,state):
        counts=self._community_counts(state); out={}
        for pair,edges in self.index.directed_edges.items():
            sw=sum(w for s,_,w in edges if state[s]==INFECTED)
            t=pair[1]; out[pair]=self.config.beta_out*sw*counts[t][0]/len(self.index.communities[t])
        return out

class TargetOracleUniform(HazardPlacementSimulator):
    def exact_pair_hazards(self,state):
        counts=self._community_counts(state); out={}
        for pair,edges in self.index.directed_edges.items():
            s=pair[0]; sf=counts[s][1]/len(self.index.communities[s])
            tw=sum(w for _,t,w in edges if state[t]==SUSCEPTIBLE)
            out[pair]=self.config.beta_out*sf*tw
        return out

class SourceTargetMarginalUniform(HazardPlacementSimulator):
    def exact_pair_hazards(self,state):
        out={}
        for pair,edges in self.index.directed_edges.items():
            W=self.index.W[pair]
            sw=sum(w for s,_,w in edges if state[s]==INFECTED)
            tw=sum(w for _,t,w in edges if state[t]==SUSCEPTIBLE)
            out[pair]=self.config.beta_out*sw*tw/W if W else 0.0
        return out

NEW_ARMS=(('S_oracle_U',SourceOracleUniform),('D_oracle_U',TargetOracleUniform),('SD_marginal_U',SourceTargetMarginalUniform))
ARCH={'FULL_MICRO_REFERENCE':'FullMicro','ORACLE_EXACT_HAZARD_EXACT_PLACEMENT':'E_E','ORACLE_EXACT_HAZARD_UNIFORM_PLACEMENT':'E_U','AGGREGATE_HAZARD_BOUNDARY_PLACEMENT':'A_B','AGGREGATE_HAZARD_UNIFORM_PLACEMENT':'A_U'}

def run_one(t):
    rep,seed,initial,h=t
    gs,fg,_=generate_two_scale_network(10,100,5,seed=seed,macro_graph_type='chain',micro_graph_type='random_connected',edge_prob=6/99)
    comm=[list(g.nodes()) for g in gs]
    c=ExperimentConfig(experiment_name='experiment_06_mechanistic_error_decomposition',L=10,n=100,mean_internal_degree=6,k=5,beta_in=.003,beta_out=.003,gamma=0,T_end=h,base_seed=270000,initial_node=initial)
    rs=[]
    for name,cls in NEW_ARMS:
        var=ModelVariant.ORACLE_EXACT_HAZARD_UNIFORM_PLACEMENT
        sim=cls(fg,comm,c,var,stream_seeds(c,rep,var),False,False,False)
        res=sim.run(initial); rs+=rows_for(rep,name,res['arrivals'],h)
    return rep,rs

def boot_ci(x,seed,B=5000):
    x=np.asarray(x,float); rng=np.random.default_rng(seed); z=np.empty(B)
    for b in range(B): idx=rng.integers(0,len(x),len(x)); z[b]=x[idx].mean()
    return np.quantile(z,[.025,.975])

def main():
    src=ROOT/'output/ablation_hazard_placement/final_100'; out=ROOT/'output/mechanistic_error_decomposition/final_100'; reps=out/'replications'; reps.mkdir(parents=True,exist_ok=True); h=4000.; n=100
    ar=pd.read_csv(src/'arrival_times.csv'); rr=pd.read_csv(src/'runs.csv'); ref=rr[rr.model_variant=='FULL_MICRO_REFERENCE'].set_index('replication_id')
    tasks=[(r,int(ref.loc[r,'network_seed']),int(ref.loc[r,'initial_node']),h) for r in range(1,n+1)]
    pending=[t for t in tasks if not (reps/f'rep_{t[0]:04d}.json').exists()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as p:
        fs={p.submit(run_one,t):t[0] for t in pending}
        for k,f in enumerate(concurrent.futures.as_completed(fs),1):
            rep,rs=f.result(); atom(reps/f'rep_{rep:04d}.json',{'replication_id':rep,'rows':rs})
            if k%10==0 or k==len(pending): print(f'[mechanistic] {k}/{len(pending)}',flush=True)
    archived=ar[ar.replication_id.between(1,n)&ar.model_variant.isin(ARCH)].copy(); archived['model']=archived.model_variant.map(ARCH); archived['scenario']='mechanistic_error_decomposition'; archived['layer']=archived.layer_index.astype(int); archived['observed']=archived.T_i.notna(); archived['restricted_T_i']=archived.T_i.fillna(h)
    rows=archived[['replication_id','scenario','model','layer','T_i','observed','restricted_T_i']].to_dict('records')
    for q in sorted(reps.glob('rep_*.json')): rows+=json.loads(q.read_text())['rows']
    f=pd.DataFrame(rows); f.to_csv(out/'arrivals.csv',index=False)
    term=f[f.layer==9]; pvt=term.pivot(index='replication_id',columns='model',values='restricted_T_i')
    base=pvt.FullMicro
    summary=[]
    for m in sorted(pvt.columns):
        d=pvt[m]-base; lo,hi=boot_ci(d,20260811+sum(map(ord,m))); w=float(wasserstein_distance(pvt[m],base))
        obs=term[term.model==m].set_index('replication_id').reindex(base.index).observed.mean()
        summary.append({'model':m,'n_runs':len(d),'completion':obs,'RMST':pvt[m].mean(),'delta_RMST_vs_FullMicro':d.mean(),'ci95_low':lo,'ci95_high':hi,'restricted_W1':w,'rel_W1':w/base.mean()})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    specs=[('A_U','S_oracle_U','source_boundary_state'),('A_U','D_oracle_U','destination_boundary_state'),('A_U','SD_marginal_U','source_destination_margins_joint'),('S_oracle_U','SD_marginal_U','destination_given_source'),('D_oracle_U','SD_marginal_U','source_given_destination'),('SD_marginal_U','E_U','edge_pair_correlation'),('E_U','E_E','exact_edge_conditioned_placement'),('A_U','E_E','full_event_driven_structural_gap')]
    cs=[]
    for left,right,label in specs:
        d=pvt[right]-pvt[left]; lo,hi=boot_ci(d,20260901+sum(map(ord,label))); cs.append({'contrast':label,'left':left,'right':right,'delta_RMST_right_minus_left':d.mean(),'ci95_low':lo,'ci95_high':hi})
    # 2x2 source x destination interaction on outcome scale
    inter=pvt.SD_marginal_U-pvt.S_oracle_U-pvt.D_oracle_U+pvt.A_U; lo,hi=boot_ci(inter,20261001)
    cs.append({'contrast':'source_destination_interaction','left':'(S-A)+(D-A)','right':'SD-A','delta_RMST_right_minus_left':inter.mean(),'ci95_low':lo,'ci95_high':hi})
    pd.DataFrame(cs).to_csv(out/'contrasts.csv',index=False)
    atom(out/'manifest.json',{'experiment':'experiment_06_mechanistic_error_decomposition','complete':True,'n_runs':100,'T_end':4000.0,'network':'chain L=10, random_connected n=100, d~6, k=5','beta_in':.003,'beta_out':.003,'gamma':0.0,'pairing':'CE01 archived networks and initial nodes; variant-specific deterministic streams','new_arms':['S_oracle_U','D_oracle_U','SD_marginal_U'],'archived_arms':ARCH})
    print(pd.DataFrame(summary).to_string(index=False)); print(pd.DataFrame(cs).to_string(index=False))
if __name__=='__main__': main()
