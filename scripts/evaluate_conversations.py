"""Fixed multi-turn acceptance dialogues; generated locally, never cached answers."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flm.model import FLM
from flm.paths import ROOT
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'runs/conversation-v2');p.add_argument('--device',default='auto');p.add_argument('--seeds',default='42,123');a=p.parse_args()
if not (a.run/'adapter.safetensors').is_file() or not (a.run/'run.json').is_file():
 p.error('No completed training run. Train first, or select --run.')
m=FLM(checkpoint=a.run/'adapter.safetensors',device=a.device,profile='conversation')
cases=json.loads((ROOT/'eval/conversations.json').read_text());results=[]
for seed in map(int,a.seeds.split(',')):
 for case in cases['cases']:
  history=[];replies=[]
  for prompt in case['turns']:
   history.append({'role':'user','content':prompt});events=list(m.generate(history,max_tokens=160,seed=seed));done=events[-1];tokens=[e for e in events if e['type']=='token'];text=done['text'];history.append({'role':'assistant','content':text})
   item={'case':case['id'],'seed':seed,'prompt':prompt,**done,'max_graph_logit_rms':max((e['logit_delta_rms'] for e in tokens),default=0),'repeated_previous_reply':text in replies}
   replies.append(text);results.append(item);print(json.dumps(item),flush=True)
   (a.run/'conversation-check.json').write_text(json.dumps({'cases_sha256':__import__('hashlib').sha256((ROOT/'eval/conversations.json').read_bytes()).hexdigest(),'results':results,'note':'Manual rubric review required; not a broad scientific benchmark.'},indent=2))
