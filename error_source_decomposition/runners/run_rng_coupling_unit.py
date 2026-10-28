#!/usr/bin/env python3
"""Unit diagnostic for cross-community dependence induced by shared RNG + horizon-safe cache rollback."""
import argparse, random
from pathlib import Path
import numpy as np,pandas as pd
from two_layer_ctmc.engines.micro_model import MicroModel

def make(rng):
 m=MicroModel(.5,0,2,rng=rng);m.add_node(0);m.add_node(1);m.add_edge(0,1,1.0);m._infect_node(0);return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=50000);p.add_argument('--out',type=Path,required=True);p.add_argument('--horizon',type=float,default=1.0);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for mode in ('shared','independent'):
  x=np.zeros(a.n,int);y=np.zeros(a.n,int)
  for s in range(a.n):
   if mode=='shared':
    random.seed(100000+s);m1=make(None);m2=make(None)
   else:
    m1=make(random.Random(200000+2*s));m2=make(random.Random(200001+2*s))
   m1.simulate_until(a.horizon);m2.simulate_until(a.horizon);x[s]=m1.I_count;y[s]=m2.I_count
  corr=float(np.corrcoef(x,y)[0,1]);same=float(np.mean(x==y));pinf1=float(np.mean(x==2));pinf2=float(np.mean(y==2))
  rows.append({'mode':mode,'n':a.n,'horizon':a.horizon,'corr_I_counts':corr,'same_state_probability':same,'p_community1_infected_target':pinf1,'p_community2_infected_target':pinf2})
 pd.DataFrame(rows).to_csv(a.out/'rng_coupling_unit.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
