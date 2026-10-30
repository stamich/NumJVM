#!/usr/bin/env python3
"""Run comparable M2 measurements; results never imply a release gate passed."""
import argparse,json,os,pathlib,platform,subprocess,datetime
p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--smoke',action='store_true',help='Short infrastructure check, not performance evidence')
a=p.parse_args();out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
metadata={'measured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'machine':platform.platform(),'cpu':platform.processor(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'smoke':a.smoke,'java':subprocess.run(['java','-version'],capture_output=True,text=True).stderr,'thread_settings':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},'simd_setting':'off'}
(out/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
args='UfuncBenchmark.addOut -p layout=C -p size=1024,65536,1048576 -f 3 -prof gc -rf json -rff '+str(out/'jmh.json')
if a.smoke: args+=' -wi 1 -w 100ms -i 1 -r 100ms'
# The property is inherited by forked JVMs through JAVA_TOOL_OPTIONS as well.
env=os.environ.copy();env['JAVA_TOOL_OPTIONS']=env.get('JAVA_TOOL_OPTIONS','')+' -Dnumjvm.simd=off'
with (out/'jmh.log').open('w') as log:
 subprocess.run(['./gradlew',':numjvm-bench:jmh','--max-workers=4','--console=plain','-PjmhArgs='+args],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
results=json.loads((out/'jmh.json').read_text())
if len(results)!=3 or any(len(r['primaryMetric']['rawData'])!=3 for r in results):
 raise SystemExit('Expected three sizes, each with three completed forks')
print('Three sizes / three forks completed; '+('smoke only' if a.smoke else 'review uncertainty before interpreting results'))
