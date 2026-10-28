#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def boot_pair(x,y,B=20000,seed=1):
 d=np.asarray(x,float)-np.asarray(y,float);rng=np.random.default_rng(seed);n=len(d);o=np.empty(B)
 for b in range(0,B,2000):
  e=min(B,b+2000);idx=rng.integers(0,n,size=(e-b,n));o[b:e]=d[idx].mean(1)
 return d.mean(),*np.quantile(o,[.025,.975])
def boot_did(a,b,c,d,B=20000,seed=2):
 # (a-b) - (c-d), paired by replication
 z=(np.asarray(a)-np.asarray(b))-(np.asarray(c)-np.asarray(d));rng=np.random.default_rng(seed);n=len(z);o=np.empty(B)
 for s in range(0,B,2000):
  e=min(B,s+2000);idx=rng.integers(0,n,size=(e-s,n));o[s:e]=z[idx].mean(1)
 return z.mean(),*np.quantile(o,[.025,.975])
def main():
 p=argparse.ArgumentParser();p.add_argument('--prod',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--eventdriven',type=Path);p.add_argument('--archived-arrivals',type=Path);p.add_argument('--n',type=int,required=True);p.add_argument('--L',type=int,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 r=pd.read_csv(a.prod);pv=r.pivot(index='replication_id',columns='arm',values='terminal_T').sort_index();
 if a.eventdriven:
  e=pd.read_csv(a.eventdriven).set_index('replication_id').terminal_T.reindex(pv.index)
 elif a.archived_arrivals:
  x=pd.read_csv(a.archived_arrivals);e=(x[(x.replication_id<=a.n)&(x.layer_index==a.L-1)&(x.model_variant=='AGGREGATE_HAZARD_UNIFORM_PLACEMENT')].set_index('replication_id').T_i.fillna(4000).reindex(pv.index))
 else:e=None
 rows=[]
 def add(label,left,right,seed):
  v,lo,hi=boot_pair(pv[left],pv[right],seed=seed);rows.append({'contrast':label,'left':left,'right':right,'delta_RMST_left_minus_right':v,'ci95_low':lo,'ci95_high':hi})
 add('cache_preserve_minus_current','cache_preserve','current',11)
 add('independent_rng_minus_current','independent_rng','current',12)
 add('remove_reimports_current','current_single_entry','current',13)
 add('remove_reimports_independent','independent_single_entry','independent_rng',14)
 # RNG correction with normal reimports vs when reimports suppressed.
 v,lo,hi=boot_did(pv.independent_rng,pv.current,pv.independent_single_entry,pv.current_single_entry,seed=15)
 rows.append({'contrast':'rng_by_reimport_interaction','left':'(independent-current)_normal','right':'(independent-current)_single_entry','delta_RMST_left_minus_right':v,'ci95_low':lo,'ci95_high':hi})
 if e is not None:
  for arm,seed in [('current',21),('cache_preserve',22),('independent_rng',23)]:
   v,lo,hi=boot_pair(pv[arm],e,seed=seed);rows.append({'contrast':arm+'_minus_event_driven_A_U','left':arm,'right':'event_driven_A_U','delta_RMST_left_minus_right':v,'ci95_low':lo,'ci95_high':hi})
 summary=r.groupby('arm').agg(n=('replication_id','size'),RMST=('terminal_T','mean'),completion=('terminal_observed','mean'),macro_events=('macro_events','mean'),repeated_macro_events=('repeated_macro_events','mean')).reset_index()
 if e is not None: summary=pd.concat([summary,pd.DataFrame([{'arm':'event_driven_A_U','n':len(e),'RMST':e.mean(),'completion':float((e<4000).mean()),'macro_events':np.nan,'repeated_macro_events':np.nan}])],ignore_index=True)
 summary.to_csv(a.out/'production_summary.csv',index=False);pd.DataFrame(rows).to_csv(a.out/'production_contrasts.csv',index=False)
 # Per-rep mediation metrics
 q=pv.copy();q['rng_effect_normal']=q.independent_rng-q.current;q['rng_effect_single_entry']=q.independent_single_entry-q.current_single_entry;q['rng_reimport_interaction']=q.rng_effect_normal-q.rng_effect_single_entry
 if e is not None:q['event_driven_A_U']=e;q['current_impl_gap']=q.current-e;q['independent_impl_gap']=q.independent_rng-e
 q.to_csv(a.out/'production_per_rep.csv')
 (a.out/'manifest.json').write_text(json.dumps({'n':a.n,'L':a.L,'bootstrap_reps':20000,'interpretation':'single-entry is a mediation/sensitivity intervention; cache and RNG are not treated as independent factorial factors because shared-RNG movement makes cache resampling path-inconsistent'},indent=2))
 print(summary.to_string(index=False));print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
