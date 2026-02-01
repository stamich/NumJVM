#!/usr/bin/env python3
"""Small deterministic matmul oracle, separate from the existing M1 fixture set."""
import argparse,pathlib,numpy as np
p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
rng=np.random.default_rng(20260128)
shapes=[((3,),(3,)),((3,),(3,4)),((2,3),(3,)),((2,3),(3,4)),((2,1,3,4),(5,4,2)),((0,2,3),(1,3,4)),((2,0),(0,3)),((2,1,1,3),(4,3,2))]
def shape(s):return ','.join(map(str,s)) or '-'
def values(x):return ','.join(repr(float(v)) for v in x.flat) or '-'
lines=['# dtypeA\tdtypeB\tshapeA\tshapeB\tshapeOut\tA\tB\texpected']
for ta,tb in [('float32','float32'),('float64','float64'),('float32','float64')]:
 for sa,sb in shapes:
  x=rng.uniform(-2,2,sa).astype(ta);y=rng.uniform(-2,2,sb).astype(tb);z=np.matmul(x,y)
  lines.append('\t'.join([ta,tb,shape(sa),shape(sb),shape(z.shape),values(x),values(y),values(z)]))
text='\n'.join(lines)+'\n'
path=pathlib.Path(__file__).resolve().parent.parent/'numjvm-linalg/src/test/resources/matmul-oracle.tsv'
if a.check:
 if path.read_text()!=text:raise SystemExit('matmul fixtures differ')
 print('24 matmul oracle cases reproduce with numpy '+np.__version__)
else:path.write_text(text)
