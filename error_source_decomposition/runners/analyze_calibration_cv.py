#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def boot(v,B=20000,seed=42):
 v=np.asarray(v,float);n=len(v);rng=np.random.default_rng(seed);o=np.empty(B)
 for s in range(0,B,2000):
  e=min(B,s+2000);idx=rng.integers(0,n,(e-s,n));o[s:e]=v[idx].mean(1)
 return v.mean(),*np.quantile(o,[.025,.975])
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--grid',type=Path,required=True);p.add_argument('--extra',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 b=pd.read_csv(a.base).set_index('replication_id');g=pd.concat([pd.read_csv(a.grid),pd.read_csv(a.extra)],ignore_index=True);pvt=g.pivot(index='replication_id',columns='calibration_factor',values='restricted_T_i');pvt[1.0]=b.A_U;pvt=pvt.sort_index(axis=1);full=b.FullMicro
 pred=pd.Series(index=b.index,dtype=float); selected=[]
 for fold in range(5):
  test=b.index[b.index%5==fold];train=b.index[b.index%5!=fold]
  errs={float(c):abs((pvt.loc[train,c]-full.loc[train]).mean()) for c in pvt.columns};cstar=min(errs,key=errs.get);pred.loc[test]=pvt.loc[test,cstar];selected.append({'fold':fold,'c_star':cstar,'train_abs_mean_delta':errs[cstar],'n_train':len(train),'n_test':len(test)})
 uncal=b.A_U-full;cal=pred-full; correction=pred-b.A_U
 rows=[]
 for label,v in [('uncalibrated_delta_vs_full',uncal),('cv_calibrated_delta_vs_full',cal),('calibration_correction',correction)]:
  m,lo,hi=boot(v,seed=100+len(rows));rows.append({'metric':label,'mean_RMST':m,'ci95_low':lo,'ci95_high':hi})
 total=(full-b.A_U)
 # directional fraction of correction closed by calibration; bootstrap ratio
 rng=np.random.default_rng(88);B=20000;n=len(b);frac=np.empty(B);resfrac=np.empty(B);tv=total.to_numpy();cv=correction.to_numpy();res=(full-pred).to_numpy()
 for s in range(0,B,2000):
  e=min(B,s+2000);idx=rng.integers(0,n,(e-s,n));den=tv[idx].mean(1);frac[s:e]=100*cv[idx].mean(1)/den;resfrac[s:e]=100*res[idx].mean(1)/den
 pd.DataFrame(selected).to_csv(a.out/'fold_selection.csv',index=False);pd.DataFrame(rows).to_csv(a.out/'calibration_cv_summary.csv',index=False)
 per=pd.DataFrame({'FullMicro':full,'A_U_uncal':b.A_U,'A_U_cv_calibrated':pred,'calibration_correction':correction,'uncal_delta':uncal,'cal_delta':cal});per.to_csv(a.out/'calibration_cv_per_rep.csv')
 meta={'candidate_factors':[float(c) for c in pvt.columns],'selected_factors':selected,'uncalibrated_gap_full_minus_AU':float(total.mean()),'mean_calibration_correction':float(correction.mean()),'fraction_initial_directional_gap_removed_pct':float(100*correction.mean()/total.mean()),'fraction_ci95':list(map(float,np.quantile(frac,[.025,.975]))),'residual_fraction_pct':float(100*(full-pred).mean()/total.mean()),'residual_fraction_ci95':list(map(float,np.quantile(resfrac,[.025,.975])))}
 (a.out/'manifest.json').write_text(json.dumps(meta,indent=2));print(pd.DataFrame(selected).to_string(index=False));print(pd.DataFrame(rows).to_string(index=False));print(meta)
if __name__=='__main__':main()
