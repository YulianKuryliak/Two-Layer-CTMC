#!/usr/bin/env python3
import argparse,concurrent.futures,json,pandas as pd,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent)); import run_production_factorial as core

def one(t):
 rep,ns,seed=t;return core.run_variant(rep,ns,seed,4000,'independent','discard','normal')
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs-csv',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=50);p.add_argument('--workers',type=int,default=10);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);parts=a.out/'parts';parts.mkdir(exist_ok=True);rs=pd.read_csv(a.runs_csv);ns=(rs[rs.model_variant=='FULL_MICRO_REFERENCE'].drop_duplicates('replication_id').set_index('replication_id').network_seed.astype(int).to_dict());tasks=[(r,ns[r],1510000+r-1) for r in range(1,a.n+1) if not (parts/f'{r:04d}.json').exists()]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
  fs=[ex.submit(one,t) for t in tasks]
  for i,f in enumerate(concurrent.futures.as_completed(fs),1):x=f.result();(parts/f"{x['replication_id']:04d}.json").write_text(json.dumps(x));print(i,'/',len(fs),flush=True) if i%10==0 or i==len(fs) else None
 rows=[json.loads(q.read_text()) for q in sorted(parts.glob('*.json')) if int(q.stem)<=a.n];pd.DataFrame([{k:v for k,v in r.items() if k!='arrivals'} for r in rows]).to_csv(a.out/'runs.csv',index=False)
if __name__=='__main__':main()
