#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument('--sync',type=Path,required=True);p.add_argument('--independent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--B',type=int,default=30000);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 s=pd.read_csv(a.sync);s=s[s.layer==s.layer.max()].pivot(index='replication_id',columns='model',values='restricted_T_i').sort_index();ind=pd.read_csv(a.independent).set_index('replication_id').terminal_T.reindex(s.index)
 per=pd.DataFrame({'FullMicro':s.FullMicro,'EventDriven_AU':s.EventDrivenAggregate,'Production_current':s.ProductionMM_tau1,'Production_independent_rng':ind})
 per['total_production_correction']=per.FullMicro-per.Production_current
 per['implementation_correction']=per.EventDriven_AU-per.Production_current
 per['structural_correction']=per.FullMicro-per.EventDriven_AU
 per['rng_stream_correction']=per.Production_independent_rng-per.Production_current
 per['residual_impl_correction']=per.EventDriven_AU-per.Production_independent_rng
 per.to_csv(a.out/'production_n100_per_rep.csv')
 rng=np.random.default_rng(20260814);n=len(per);B=a.B;metrics={k:np.empty(B) for k in ['total','impl','struct','rng','residual','impl_pct_total','struct_pct_total','rng_pct_impl','residual_pct_impl']}
 arr={c:per[c].to_numpy(float) for c in per.columns}
 for b in range(0,B,2000):
  e=min(B,b+2000);idx=rng.integers(0,n,(e-b,n));m={c:arr[c][idx].mean(1) for c in arr};total=m['total_production_correction'];impl=m['implementation_correction'];struct=m['structural_correction'];rg=m['rng_stream_correction'];res=m['residual_impl_correction'];
  metrics['total'][b:e]=total;metrics['impl'][b:e]=impl;metrics['struct'][b:e]=struct;metrics['rng'][b:e]=rg;metrics['residual'][b:e]=res;metrics['impl_pct_total'][b:e]=100*impl/total;metrics['struct_pct_total'][b:e]=100*struct/total;metrics['rng_pct_impl'][b:e]=100*rg/impl;metrics['residual_pct_impl'][b:e]=100*res/impl
 rows=[]
 mapcols={'total_production_correction':'total','implementation_correction':'impl','structural_correction':'struct','rng_stream_correction':'rng','residual_impl_correction':'residual'}
 for label,k in mapcols.items():rows.append({'metric':label,'mean_RMST':float(per[label].mean()),'ci95_low':float(np.quantile(metrics[k],.025)),'ci95_high':float(np.quantile(metrics[k],.975))})
 pd.DataFrame(rows).to_csv(a.out/'production_n100_contrasts.csv',index=False)
 shares=[
 {'component':'implementation_vs_eventdriven','percent':100*per.implementation_correction.mean()/per.total_production_correction.mean(),'ci95_low':np.quantile(metrics['impl_pct_total'],.025),'ci95_high':np.quantile(metrics['impl_pct_total'],.975),'denominator':'FullMicro - current Production'},
 {'component':'structural_eventdriven_AU','percent':100*per.structural_correction.mean()/per.total_production_correction.mean(),'ci95_low':np.quantile(metrics['struct_pct_total'],.025),'ci95_high':np.quantile(metrics['struct_pct_total'],.975),'denominator':'FullMicro - current Production'},
 {'component':'shared_RNG_stream_coupling_correction','percent':100*per.rng_stream_correction.mean()/per.implementation_correction.mean(),'ci95_low':np.quantile(metrics['rng_pct_impl'],.025),'ci95_high':np.quantile(metrics['rng_pct_impl'],.975),'denominator':'EventDriven A/U - current Production'},
 {'component':'remaining_implementation_after_independent_RNG','percent':100*per.residual_impl_correction.mean()/per.implementation_correction.mean(),'ci95_low':np.quantile(metrics['residual_pct_impl'],.025),'ci95_high':np.quantile(metrics['residual_pct_impl'],.975),'denominator':'EventDriven A/U - current Production'},]
 pd.DataFrame(shares).to_csv(a.out/'production_n100_signed_shares.csv',index=False)
 model=[]
 for c in ['FullMicro','EventDriven_AU','Production_current','Production_independent_rng']:model.append({'model':c,'RMST':float(per[c].mean()),'completion':float((per[c]<4000).mean())})
 pd.DataFrame(model).to_csv(a.out/'production_n100_model_summary.csv',index=False)
 meta={'n':n,'B':B,'interpretation':'directional signed decomposition; RNG intervention removes shared micro/macro stream coupling. Percent can exceed 100 if residual acts in opposite direction. Independent-RNG residual vs event-driven should be judged by its CI, not by point percent alone.'};(a.out/'manifest.json').write_text(json.dumps(meta,indent=2));print(pd.DataFrame(model).to_string(index=False));print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(shares).to_string(index=False))
if __name__=='__main__':main()
