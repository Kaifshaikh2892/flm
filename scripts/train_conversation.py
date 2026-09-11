"""Train a conservative fly readout; separate development and untouched test splits."""
import argparse,copy,json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file
from flm.model import FLM,FlyAdapter
from flm.paths import ROOT,CACHE,GRAPH,DATA_ID,DATA_REVISION
from flm.graph import sha256,ReservoirBatch,NativeBatchKernel
from flm import conversation_config as cfg

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/conversation-v2');p.add_argument('--device',default='auto');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
if (a.output/'run.json').exists():raise SystemExit('Completed run exists; use a new output.')
kernel=NativeBatchKernel(CACHE/'native') if (CACHE/'native/manifest.json').exists() else None
started=time.monotonic();torch.manual_seed(27);np.random.seed(27)
m=FLM(checkpoint=None,profile='conversation',device=a.device)
print(json.dumps({'stage':'loaded','device':m.device,'parameters':sum(x.numel() for x in m.base.parameters())}),flush=True)
weights_hash=sha256(m.model_dir/'model.safetensors')

def record(messages,key):
    # Supervise a complete assistant answer against exactly the prefix used at inference.
    context=[dict(x) for x in messages[:-1]];target=messages[-1]['content']
    while True:
        prefix=m.prompt_ids(context,max_context=100000)
        if len(prefix)<=256 or len(context)<=1:break
        context=context[2:]
    if len(prefix)>320:return None
    complete=m.tokenizer.apply_chat_template([m.system]+context+[{'role':'assistant','content':target}],tokenize=True,add_generation_prompt=False,**m.chat_kwargs)
    if complete[:len(prefix)]!=prefix:raise ValueError('Training/inference prefix mismatch.')
    # Only complete answers; no training on cut-off prose or unseen future labels.
    answer=complete[len(prefix):]
    if m.tokenizer.eos_token_id in answer:answer=answer[:answer.index(m.tokenizer.eos_token_id)+1]
    if not 5<=len(answer)<=144:return None
    ids=np.array(prefix+answer,dtype=np.int64)
    mask=np.arange(len(ids)-1)>=len(prefix)-1
    return {'key':key,'ids':ids[:-1],'labels':ids[1:],'mask':mask,'conversation_sha256':__import__('hashlib').sha256(json.dumps(context+[{'role':'assistant','content':target}],sort_keys=True).encode()).hexdigest()}

def corpus(split,count,seed,exclude=()):
    rows=pq.read_table(CACHE/'corpus/data/everyday-conversations'/f'{split}-00000-of-00001.parquet').to_pylist()
    found=[]
    for i in np.random.default_rng(seed).permutation(len(rows)):
        if int(i) in exclude:continue
        messages=rows[int(i)]['messages'][2:]
        # Prefer follow-ups so the training context includes a prior real exchange.
        choices=[j for j in range(1,len(messages)) if messages[j]['role']=='assistant']
        choices=choices[1:]+choices[:1]
        for j in choices:
            item=record(messages[:j+1],f'{split}:{int(i)}:{j}')
            if item is not None:found.append(item);break
        if len(found)==count:return found
    raise ValueError('Not enough suitable conversations.')

train=corpus('train',64,2701)
styles=json.loads((ROOT/'training/conversation-style.json').read_text())['conversations']
for i,lines in enumerate(styles):
    messages=[{'role':'user' if j%2==0 else 'assistant','content':s} for j,s in enumerate(lines)]
    item=record(messages,f'original-style:{i}')
    if item is None:raise ValueError(f'Style example {i} does not fit.')
    train.append(item)
old=json.loads((ROOT/'eval/prototype-holdout.json').read_text())['row_indices']
heldout=corpus('test',40,2702,old);splits={'train':train,'validation':heldout[:16],'test':heldout[16:]}
assert not (set(x['conversation_sha256'] for x in train)&set(x['conversation_sha256'] for x in heldout))
selection={split:[{k:v for k,v in x.items() if k in ('key','conversation_sha256')} for x in records] for split,records in splits.items()}
plan={'train':len(train),'validation':16,'test':24,'epochs':3,'seed':27,'learning_rate':0.0003,'batch_size':32,'KL_weight':0.5,'selection':'lowest validation NLL, separately for fly and matched direct-input control','test_policy':'only after selection','interface':cfg.INTERFACE,'style_sha256':sha256(ROOT/'training/conversation-style.json'),'acceptance_cases_sha256':sha256(ROOT/'eval/conversations.json')}
plan['feature_extraction']={'batch_columns':8,'native_kernel':json.loads((CACHE/'native/manifest.json').read_text()) if kernel else None,'serial_check_rtol':3e-5,'serial_check_atol':3e-6}
(a.output/'selection.json').write_text(json.dumps(selection,indent=2));(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
# Reusing the identical system-prefix state is equivalent to replaying it for every row.
all_ids=[x['ids'] for records in splits.values() for x in records];n=0
while n<min(map(len,all_ids)) and all(ids[n]==all_ids[0][n] for ids in all_ids):n+=1
prefix_states={}
for mode in ['intact','shuffled']:
    r=m.reservoir()
    for token in all_ids[0][:n]:r.step(m.embeddings[int(token)],mode)
    prefix_states[mode]=r.state.copy()
print(json.dumps({'stage':'plan','counts':{k:len(v) for k,v in splits.items()},'shared_prefix_tokens':n}),flush=True)

@torch.no_grad()
def extract(split,records):
    target=a.output/f'{split}-features.npz'
    # Every cache includes its selection and model/interface signature.
    signature={'selection':selection[split],'model_sha256':weights_hash,'interface':cfg.INTERFACE,'system':cfg.SYSTEM_TEXT,'encoder_sha256':__import__('hashlib').sha256(__import__('inspect').getsource(record).encode()).hexdigest(),'graph_code_sha256':sha256(ROOT/'flm/graph.py'),'feature_version':3,'native_kernel':json.loads((CACHE/'native/manifest.json').read_text()) if kernel else None}
    sig=target.with_suffix('.json')
    if target.exists() and sig.exists() and json.loads(sig.read_text())==signature:return dict(np.load(target,allow_pickle=False))
    columns={key:[] for key in ['hidden','intact','direct','labels']+(['shuffled'] if split=='test' else [])}
    # One GPU gather per distinct token, not a synchronization on every graph step.
    token_ids=np.unique(np.concatenate([item['ids'] for item in records]))
    embedding_values=m.embeddings[token_ids]
    lookup={int(token):i for i,token in enumerate(token_ids)}
    embedded=[embedding_values[[lookup[int(token)] for token in item['ids'][n:]]] for item in records]
    for i,item in enumerate(records):
        ids,mask=item['ids'],item['mask']
        assert not mask[:n].any()
        h=m.base.model(torch.as_tensor(ids[None,:],device=m.device),use_cache=False).last_hidden_state[0]
        columns['hidden'].append(h[torch.as_tensor(mask,device=m.device)].float().cpu().numpy().astype(np.float16))
        columns['labels'].append(item['labels'][mask])
    print(json.dumps({'stage':'hidden','split':split,'rows':len(records),'elapsed':round(time.monotonic()-started,1)}),flush=True)
    for mode in ['intact']+(['shuffled'] if split=='test' else []):
        for offset in range(0,len(records),8):
            group=records[offset:offset+8];vectors=embedded[offset:offset+8];r=m.reservoir()
            batch=ReservoirBatch(r,len(group),prefix_states[mode],kernel);pooled=[[] for _ in group];direct=[[] for _ in group]
            # Numerical guard on the actual 25.6M-edge graph, not just a toy test.
            references=[]
            if offset==0:
                for _ in group:
                    ref=m.reservoir();ref.state=prefix_states[mode].copy();references.append(ref)
            for t in range(max(map(len,vectors))):
                emb=np.stack([v[t] if t<len(v) else np.zeros(m.hidden_size,np.float32) for v in vectors])
                f=batch.step(emb,mode)
                if references and t<4:
                    expected=np.stack([ref.step(e,mode) for ref,e in zip(references,emb)])
                    np.testing.assert_allclose(f,expected,rtol=3e-5,atol=3e-6)
                for j,item in enumerate(group):
                    if t<len(vectors[j]) and item['mask'][n+t]:
                        pooled[j].append(f[j].copy())
                        if mode=='intact':direct[j].append(r.project_input(emb[j]))
            columns[mode].extend(np.asarray(row) for row in pooled)
            if mode=='intact':columns['direct'].extend(np.asarray(row) for row in direct)
            print(json.dumps({'stage':'graph-batch','split':split,'mode':mode,'rows':min(offset+8,len(records)),'total':len(records),'elapsed':round(time.monotonic()-started,1)}),flush=True)
    values={k:np.concatenate(v) for k,v in columns.items()};np.savez(target,**values);sig.write_text(json.dumps(signature));return values

features={split:extract(split,records) for split,records in splits.items()}
def batches(data,kind,order):
    for offset in range(0,len(order),plan['batch_size']):
        ix=order[offset:offset+plan['batch_size']]
        yield (torch.as_tensor(data['hidden'][ix],device=m.device),torch.as_tensor(data[kind][ix],device=m.device),torch.as_tensor(data['labels'][ix],device=m.device))

@torch.no_grad()
def evaluate(adapter,split,kind):
    total=kl_sum=flips=rms_sum=count=0
    data=features[split]
    for h,f,y in batches(data,kind,np.arange(len(data['labels']))):
        logits,base,_=m.scores(h,f,'base' if adapter is None else 'intact',adapter)
        total+=float(torch.nn.functional.cross_entropy(logits,y,reduction='sum'))
        logp=torch.log_softmax(base,-1);logq=torch.log_softmax(logits,-1)
        kl_sum+=float((logp.exp()*(logp-logq)).sum());flips+=int((base.argmax(-1)!=logits.argmax(-1)).sum());rms_sum+=float((logits-base).square().mean(-1).sum());count+=len(y)
    return {'nll':total/count,'perplexity':math.exp(total/count),'mean_KL_base_to_mode':max(0,kl_sum/count),'top_token_change_fraction':flips/count,'logit_delta_rms':math.sqrt(rms_sum/count),'targets':count}

curves=[];selected={}
for kind in ['intact','direct']:
    torch.manual_seed(27);adapter=FlyAdapter(m.hidden_size,scale=cfg.INTERFACE['adapter_scale']).to(m.device)
    opt=torch.optim.AdamW(adapter.parameters(),lr=plan['learning_rate'],weight_decay=.01)
    best=float('inf');best_weights=None
    for epoch in range(plan['epochs']):
        adapter.train();total=count=0
        for h,f,y in batches(features['train'],kind,np.random.default_rng(2700+epoch).permutation(len(features['train']['labels']))):
            opt.zero_grad(set_to_none=True);logits,base,_=m.scores(h,f,adapter=adapter)
            ce=torch.nn.functional.cross_entropy(logits,y)
            divergence=torch.nn.functional.kl_div(torch.log_softmax(logits,-1),torch.softmax(base,-1),reduction='batchmean')
            loss=ce+plan['KL_weight']*divergence
            if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1);opt.step();total+=float(ce.detach())*len(y);count+=len(y)
        adapter.eval();dev=evaluate(adapter,'validation',kind)
        item={'stage':'epoch','model':kind,'epoch':epoch+1,'train_nll':total/count,'validation':dev,'elapsed':round(time.monotonic()-started,1)};curves.append(item);print(json.dumps(item),flush=True)
        if dev['nll']<best:best=dev['nll'];best_weights={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
    adapter.load_state_dict(best_weights);adapter.eval();selected[kind]=adapter
    save_file(best_weights,str(a.output/('adapter.safetensors' if kind=='intact' else 'direct-control.safetensors')))
report={'base':evaluate(None,'test','intact'),'fly_adapter':evaluate(selected['intact'],'test','intact'),'direct_input_adapter':evaluate(selected['direct'],'test','direct'),'relabeled_wiring':evaluate(selected['intact'],'test','shuffled')}
report['no_edges']={**report['base'],'exact_base_identity':True}
assert torch.count_nonzero(selected['intact'](torch.zeros(2,128,device=m.device))).item()==0
assert all(not x.requires_grad and x.grad is None for x in m.base.parameters())
assert sha256(m.model_dir/'model.safetensors')==weights_hash
manifest={'schema_version':2,'model_id':cfg.MODEL_ID,'model_revision':cfg.MODEL_REVISION,'model_weights_sha256':weights_hash,'graph_sha256':sha256(GRAPH/'manifest.json'),'adapter_sha256':sha256(a.output/'adapter.safetensors'),'system_text':cfg.SYSTEM_TEXT,'interface':cfg.INTERFACE,'context_tokens':cfg.CONTEXT_TOKENS,'sampling':cfg.SAMPLING,'dataset_id':DATA_ID,'dataset_revision':DATA_REVISION,'selection_sha256':sha256(a.output/'selection.json'),'training':{**plan,'train_examples':len(train)},'trainable_parameters':sum(x.numel() for x in selected['intact'].parameters()),'base_parameters':sum(x.numel() for x in m.base.parameters()),'train_targets':len(features['train']['labels']),'test_targets':len(features['test']['labels']),'device':m.device,'claim':'A stronger frozen language backbone with a bounded, trained readout of the full retained fly graph. Not biological language or online learning.'}
report['notes']=['24 held-out conversations, disjoint from fitting and validation; not a broad chat benchmark.','The corpus may have appeared in backbone pretraining.','Wiring specificity requires stronger matched topology controls; a direct-input readout is included.','Creative training examples are original synthetic data; acceptance dialogues are separate.','All graph edges retained; no response replacement, duplicate-string blocking, or remote language API.']
for name,data in [('run',manifest),('report',report),('training',curves)]: (a.output/f'{name}.json').write_text(json.dumps(data,indent=2))
print(json.dumps({'stage':'complete','report':report,'elapsed':round(time.monotonic()-started,1)}),flush=True)
