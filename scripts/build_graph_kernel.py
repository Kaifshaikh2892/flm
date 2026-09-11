"""Optional portable C kernel for faster offline extraction; not required for inference."""
import json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flm.paths import ROOT,CACHE
from flm.graph import sha256
folder=CACHE/'native';folder.mkdir(parents=True,exist_ok=True);source=ROOT/'native/graph_batch.c';output=folder/'graph-batch.so'
subprocess.run(['cc','-O3','-ffp-contract=off','-shared','-fPIC',str(source),'-o',str(output)],check=True)
(folder/'manifest.json').write_text(json.dumps({'source_sha256':sha256(source),'binary_sha256':sha256(output),'flags':'-O3 -ffp-contract=off -shared -fPIC'}))
print(output)
