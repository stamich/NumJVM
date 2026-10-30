#!/usr/bin/env python3
"""Small deterministic matmul oracle, separate from the existing M1 fixture set."""
import argparse,math,pathlib,numpy as np
p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
rng=np.random.default_rng(20260128)
shapes=[((3,),(3,)),((3,),(3,4)),((2,3),(3,)),((2,3),(3,4)),((2,1,3,4),(5,4,2)),((0,2,3),(1,3,4)),((2,0),(0,3)),((2,1,1,3),(4,3,2))]
def shape(s):return ','.join(map(str,s)) or '-'
def reference(x,y,out):
 # np.matmul runs through BLAS, whose accumulation order depends on the CPU kernel, so the
 # fixture would differ between CI runners. Summing exactly (math.fsum) gives the correctly
 # rounded result for any hardware; the Java replay compares with a tolerance anyway.
 a=x.reshape((1,)+x.shape) if x.ndim==1 else x
 b=y.reshape(y.shape+(1,)) if y.ndim==1 else y
 batch=np.broadcast_shapes(a.shape[:-2],b.shape[:-2]);m,k=a.shape[-2:];n=b.shape[-1]
 a=np.broadcast_to(a,batch+(m,k));b=np.broadcast_to(b,batch+(k,n))
 r=np.empty(batch+(m,n),dtype=out)
 for idx in np.ndindex(*batch):
  for i in range(m):
   for j in range(n):
    r[idx+(i,j)]=math.fsum(float(a[idx+(i,l)])*float(b[idx+(l,j)]) for l in range(k))
 if x.ndim==1:r=r.reshape(r.shape[:-2]+r.shape[-1:])
 if y.ndim==1:r=r.reshape(r.shape[:-1])
 return r
def values(x):return ','.join(repr(float(v)) for v in x.flat) or '-'
lines=['# dtypeA\tdtypeB\tshapeA\tshapeB\tshapeOut\tA\tB\texpected']
for ta,tb in [('float32','float32'),('float64','float64'),('float32','float64')]:
 for sa,sb in shapes:
  x=rng.uniform(-2,2,sa).astype(ta);y=rng.uniform(-2,2,sb).astype(tb);z=reference(x,y,np.result_type(x,y))
  lines.append('\t'.join([ta,tb,shape(sa),shape(sb),shape(z.shape),values(x),values(y),values(z)]))
text='\n'.join(lines)+'\n'
path=pathlib.Path(__file__).resolve().parent.parent/'numjvm-linalg/src/test/resources/matmul-oracle.tsv'
if a.check:
 if path.read_text()!=text:raise SystemExit('matmul fixtures differ')
 print('24 matmul oracle cases reproduce with numpy '+np.__version__)
else:path.write_text(text)
