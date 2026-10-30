#!/usr/bin/env python3
"""Summarize completed test XML and coverage, without claiming unrun external gates."""
import argparse,datetime,hashlib,json,pathlib,xml.etree.ElementTree as ET
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent.parent
report={'validated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'version':'0.3.0-alpha','tests':{},'line_coverage':{},'external_gates':'reference performance campaign and remote CI/Sonar not verified'}
for module,runner in [('numjvm-core','test'),('numjvm-io','test'),('numjvm-vector','test'),('numjvm-linalg','test'),('numjvm-linalg','nativeTest'),('numjvm-linalg','autoTest')]:
 files=list((root/module/'build/test-results'/runner).glob('TEST-*.xml'))
 if not files:raise SystemExit('No test results: '+module+'/'+runner)
 elements=[ET.parse(f).getroot() for f in files]
 counts={k:sum(int(e.get(k,0)) for e in elements) for k in ['tests','failures','errors','skipped']}
 if counts['tests']==0 or counts['failures'] or counts['errors']:raise SystemExit('Failed/empty runner: '+module+'/'+runner)
 report['tests'][module+'/'+runner]=counts
for module in ['numjvm-core','numjvm-io','numjvm-vector','numjvm-linalg']:
 element=ET.parse(root/module/'build/reports/jacoco/test/jacocoTestReport.xml').getroot()
 counter=next(c for c in element.findall('counter') if c.get('type')=='LINE')
 hit=int(counter.get('covered'));total=hit+int(counter.get('missed'))
 if total==0 or hit/total<0.8:raise SystemExit('Coverage gate failed: '+module)
 report['line_coverage'][module]={'covered':hit,'total':total,'percent':round(hit/total*100,2)}
files=sorted(f for f in root.rglob('*') if f.is_file() and not any(p in {'.git','.gradle','.venv','build'} for p in f.relative_to(root).parts) and (f.suffix in {'.java','.toml'} or f.name.endswith('.gradle.kts') or f.name=='gradle.properties'))
h=hashlib.sha256()
for f in files:h.update(str(f.relative_to(root)).encode()+b'\0'+f.read_bytes()+b'\0')
report['java_and_build_source_sha256']=h.hexdigest()
pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
