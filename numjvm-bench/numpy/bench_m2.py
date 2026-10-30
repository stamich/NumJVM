#!/usr/bin/env python3
"""Separate NumPy timing protocol; do not equate Python timing uncertainty with JMH."""
import argparse,contextlib,io,json,os,time,numpy as np
p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=1048576);p.add_argument('--shape',default='32x32x32');p.add_argument('--repeat',type=int,default=7);a=p.parse_args()
i=np.arange(a.size,dtype=np.int32);f=np.ones(a.size,dtype=np.float64);out=np.zeros_like(f)
m,k,n=map(int,a.shape.split('x'));left=np.ones((m,k));right=np.ones((k,n));matrix_out=np.zeros((m,n));sum_out=np.zeros((),dtype=np.int64);mean_out=np.zeros((),dtype=np.float64)
cases={'mixedAdd':lambda:np.add(i,f),'mixedAddOut':lambda:np.add(i,f,out=out),'wideningSum':lambda:np.sum(i),'wideningSumOut':lambda:np.sum(i,out=sum_out),'integerMean':lambda:np.mean(i),'integerMeanOut':lambda:np.mean(i,out=mean_out),'matmulAllocating':lambda:np.matmul(left,right),'matmulPreallocated':lambda:np.matmul(left,right,out=matrix_out)}
config=io.StringIO()
with contextlib.redirect_stdout(config):np.show_config()
result={'numpy':np.__version__,'config':config.getvalue(),'size':a.size,'shape':a.shape,'threads':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},'samples_ns':{}}
for name,fn in cases.items():
 for _ in range(10):fn()
 samples=[]
 for _ in range(a.repeat):
  count=0;start=time.perf_counter_ns()
  while time.perf_counter_ns()-start<100_000_000:fn();count+=1
  samples.append((time.perf_counter_ns()-start)/count)
 result['samples_ns'][name]=samples
print(json.dumps(result,indent=2))
