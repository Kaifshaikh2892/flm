"""Compare independent serial recurrences with the optional batched kernel."""
import sys,time,json,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flm.graph import Connectome,Reservoir,ReservoirBatch,NativeBatchKernel
from flm.paths import CACHE,ROOT
g=Connectome();kernel=NativeBatchKernel(CACHE/'native');inputs=np.random.default_rng(28).normal(size=(16,3,2048)).astype(np.float32);report={}
for mode in ['intact','shuffled','no_edges']:
 refs=[Reservoir(g,2048) for _ in range(3)];batch=ReservoirBatch(Reservoir(g,2048),3,kernel=kernel);error=0
 for emb in inputs:
  expected=np.stack([r.step(e,mode) for r,e in zip(refs,emb)]);actual=batch.step(emb,mode);error=max(error,float(np.max(np.abs(expected-actual))))
  np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-6)
  for j,r in enumerate(refs):np.testing.assert_allclose(batch.state[:,j],r.state,rtol=3e-5,atol=3e-6)
 report[mode]={'max_feature_absolute_error':error,'steps':16,'independent_conversations':3}
output=ROOT/'runs/conversation-diagnostics/native-graph-check.json'
output.parent.mkdir(parents=True,exist_ok=True)
print(json.dumps(report),flush=True);output.write_text(json.dumps(report,indent=2))
