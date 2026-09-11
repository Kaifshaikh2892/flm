"""One-time public downloads, pinned to immutable upstream revisions."""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import snapshot_download, hf_hub_download
from flm.paths import MODEL_ID, MODEL_REVISION, MODEL, DATA_ID, DATA_REVISION, CACHE

p=argparse.ArgumentParser();p.add_argument('--legacy',action='store_true');args=p.parse_args()
if not args.legacy:
    from flm.conversation_config import MODEL_ID,MODEL_REVISION,MODEL_DIR as MODEL
snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=MODEL,
    allow_patterns=['config.json','generation_config.json','model.safetensors','tokenizer.json',
                    'tokenizer_config.json','special_tokens_map.json','chat_template.jinja','README.md','LICENSE'])
for split in ['train','test']:
    hf_hub_download(DATA_ID, repo_type='dataset', revision=DATA_REVISION,
        filename=f'data/everyday-conversations/{split}-00000-of-00001.parquet', local_dir=CACHE/'corpus')
hf_hub_download(DATA_ID,repo_type='dataset',revision=DATA_REVISION,filename='README.md',local_dir=CACHE/'corpus')
print('Pinned model and conversation corpus downloaded. Inference is local.')
