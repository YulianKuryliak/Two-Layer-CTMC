#!/usr/bin/env python3
"""Optimistic lower bound for any architecture-blind scalar calibration.
If A/U cannot see near/random/far gateway identity, any scalar calibration produces
the same prediction for all three modes. We therefore allow an even stronger oracle:
choose an arbitrary common RMST x, and compute the best possible residual errors.
If residuals remain, no scalar calibration can remove hidden-architecture error.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument('--arrivals',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 f=pd.read_csv(a.arrivals);term=f[f.layer==f.layer.max()];ref=term[term.model=='E_E'].groupby('mode').restricted_T_i.mean().reindex(['near','random','far']);au=term[term.model=='A_U'].groupby('mode').restricted_T_i.mean().reindex(ref.index)
 vals=ref.to_numpy(float);x_l2=float(vals.mean());x_l1=float(np.median(vals));x_linf=float((vals.min()+vals.max())/2)
 rows=[]
 for objective,x in [('L2_mean',x_l2),('L1_median',x_l1),('Linf_midrange',x_linf)]:
  for mode,y in ref.items():rows.append({'objective':objective,'common_prediction_RMST':x,'mode':mode,'reference_EE_RMST':y,'residual_prediction_minus_reference':x-y,'abs_residual':abs(x-y)})
 pd.DataFrame(rows).to_csv(a.out/'architecture_blind_calibration_bound.csv',index=False)
 pd.DataFrame({'mode':ref.index,'E_E_RMST':ref.values,'A_U_RMST':au.values,'A_U_minus_EE':au.values-ref.values}).to_csv(a.out/'gateway_means.csv',index=False)
 meta={'interpretation':'optimistic lower bound: scalar/architecture-blind calibration cannot distinguish gateway modes; arbitrary common prediction is more flexible than any fixed scalar c','best_common_L2':x_l2,'best_common_L1':x_l1,'best_common_Linf':x_linf,'reference_range':float(vals.max()-vals.min()),'minimax_unavoidable_abs_error':float((vals.max()-vals.min())/2)}
 (a.out/'manifest.json').write_text(json.dumps(meta,indent=2));print(pd.DataFrame({'mode':ref.index,'E_E':ref.values,'A_U':au.values}).to_string(index=False));print(meta)
if __name__=='__main__':main()
