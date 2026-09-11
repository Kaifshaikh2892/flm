"""Frozen small chat backbone plus a trainable, bias-free graph readout."""
import json
import os
import time
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from .paths import MODEL, RUN, GRAPH, MODEL_ID, MODEL_REVISION
from .graph import Connectome, Reservoir, sha256
from . import conversation_config as conversation
from .decoding import bound_logit_delta, sample_token

SYSTEM = {'role':'system','content':'You are FLM, a small experimental language model. Answer the user directly and concisely. You are software, not a biological fly.'}


def device_name(requested='auto'):
    if requested != 'auto':
        return requested
    return 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


class FlyAdapter(nn.Module):
    def __init__(self, hidden_size, features=128, width=128, scale=0.15):
        super().__init__()
        self.scale = scale
        self.input = nn.Linear(features, width, bias=False)
        self.output = nn.Linear(width, hidden_size, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, features):
        return self.scale * torch.tanh(self.output(torch.nn.functional.gelu(self.input(features))))


class FLM:
    def __init__(self, device='auto', checkpoint=RUN/'adapter.safetensors', graph=None, profile=None):
        torch.set_num_threads(max(1, min(8, int(os.environ.get('FLM_CPU_THREADS', '2')))))
        self.device = device_name(device)
        manifest = json.loads(checkpoint.with_name('run.json').read_text()) if checkpoint and checkpoint.exists() else None
        self.conversation_profile = profile == 'conversation' or bool(manifest and manifest.get('schema_version') == 2)
        self.model_dir = conversation.MODEL_DIR if self.conversation_profile else MODEL
        self.model_id = conversation.MODEL_ID if self.conversation_profile else MODEL_ID
        self.model_revision = conversation.MODEL_REVISION if self.conversation_profile else MODEL_REVISION
        self.interface = conversation.INTERFACE if self.conversation_profile else {'seed':7301,'dimensions':128,'input_gain':0.4,'recurrence_gain':0.6,'adapter_scale':0.15}
        self.system = {'role':'system','content':conversation.SYSTEM_TEXT} if self.conversation_profile else SYSTEM
        self.max_context = conversation.CONTEXT_TOKENS if self.conversation_profile else 512
        self.chat_kwargs = {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=False)
        dtype = torch.float16 if self.conversation_profile and self.device == 'mps' else torch.float32
        self.base = AutoModelForCausalLM.from_pretrained(self.model_dir, local_files_only=True,
            trust_remote_code=False, dtype=dtype, attn_implementation='eager').to(self.device).eval()
        self.base.requires_grad_(False)
        self.hidden_size = self.base.config.hidden_size
        self.embeddings = EmbeddingView(self.base.get_input_embeddings().weight)
        self.graph = graph or Connectome()
        self.adapter = FlyAdapter(self.hidden_size, scale=self.interface['adapter_scale']).to(self.device).eval()
        self.trained = False
        self.run_manifest = None
        if checkpoint is not None and checkpoint.exists():
            if manifest['graph_sha256'] != sha256(GRAPH/'manifest.json'):
                raise ValueError('Checkpoint belongs to a different graph.')
            if manifest['model_id'] != self.model_id or manifest['model_revision'] != self.model_revision:
                raise ValueError('Checkpoint belongs to a different language backbone.')
            if manifest['model_weights_sha256'] != sha256(self.model_dir/'model.safetensors'):
                raise ValueError('Backbone weights checksum mismatch.')
            if manifest['interface'] != self.interface:
                raise ValueError('Checkpoint interface configuration differs.')
            if self.conversation_profile and (manifest['system_text'] != self.system['content'] or manifest['context_tokens'] != self.max_context or manifest['sampling'] != conversation.SAMPLING):
                raise ValueError('Checkpoint conversation configuration differs.')
            if manifest['adapter_sha256'] != sha256(checkpoint):
                raise ValueError('Checkpoint checksum mismatch.')
            self.adapter.load_state_dict(load_file(str(checkpoint)))
            self.run_manifest = manifest
            self.trained = True
        self.quantized = False
        self.prefix_cache = None
        if self.conversation_profile and self.trained:
            # Cache only public system instructions, never a visitor's conversation.
            prefix = self.tokenizer.apply_chat_template([self.system],tokenize=True,add_generation_prompt=False)
            reservoir = self.reservoir()
            for token in prefix:
                reservoir.step(self.embeddings[token])
            self.prefix_cache = (prefix, reservoir.state.copy())

    def reservoir(self):
        return Reservoir(self.graph, self.hidden_size)

    def prompt_ids(self, messages, max_context=None):
        # Model identity is a transparent system instruction, never a fabricated
        # account of its feelings or a stand-in for graph computation.
        max_context = max_context or self.max_context
        ids = self.tokenizer.apply_chat_template([self.system]+messages, tokenize=True, add_generation_prompt=True, **self.chat_kwargs)
        if len(ids) > max_context:
            raise ValueError(f'Conversation exceeds the {max_context}-token prototype context. Start a new chat.')
        return ids

    def scores(self, hidden, features, mode='intact', adapter=None):
        delta = (adapter or self.adapter)(features.float()) if mode != 'base' else torch.zeros_like(hidden)
        if self.conversation_profile:
            # Read the large vocabulary matrix once for the two score vectors.
            # A shared linear projection preserves the same additive model.
            projected = self.base.lm_head(torch.cat((hidden, delta), dim=0).to(self.base.dtype)).float()
            baseline, change = projected.split(hidden.shape[0], dim=0)
            # This is a model constraint, not a repetition filter or answer replacement.
            change = bound_logit_delta(change, self.interface['max_logit_rms'])
            return baseline + change, baseline, delta
        baseline = self.base.lm_head(hidden.to(self.base.dtype)).float()
        return self.base.lm_head(hidden+delta).float(), baseline, delta

    @torch.no_grad()
    def generate(self, messages, mode='intact', max_tokens=160, temperature=None, seed=42, cancelled=None, telemetry=True):
        if mode not in ('intact','no_edges','shuffled','base'):
            raise ValueError('Unknown comparison mode.')
        if mode != 'base' and not self.trained:
            raise ValueError('Train the graph adapter before chatting in FLM mode.')
        if cancelled is not None and cancelled.is_set(): return
        ids = self.prompt_ids(messages)
        temperature = (conversation.SAMPLING['temperature'] if self.conversation_profile else 0.6) if temperature is None else temperature
        reservoir = self.reservoir()
        started = time.perf_counter()
        features = np.zeros(128, np.float32)
        start = 0
        if mode == 'intact' and self.prefix_cache:
            prefix, state = self.prefix_cache
            if ids[:len(prefix)] == prefix:
                start = len(prefix)
                reservoir.state = state.copy()
                reservoir.tokens = start
        if mode != 'base':
            for i, token in enumerate(ids[start:], start):
                if cancelled is not None and cancelled.is_set(): return
                features = reservoir.step(self.embeddings[token], mode)
                if i % 24 == 0:
                    yield {'type':'progress','processed':i+1,'total':len(ids)}
        if cancelled is not None and cancelled.is_set(): return
        tokens = torch.tensor([ids], device=self.device)
        result = self.base.model(tokens, use_cache=True)
        hidden, past = result.last_hidden_state[:, -1], result.past_key_values
        generated = []
        rng = torch.Generator(device='cpu').manual_seed(seed)
        for index in range(max_tokens):
            if cancelled is not None and cancelled.is_set(): return
            logits, baseline_logits, delta = self.scores(hidden, torch.tensor(features, device=self.device).unsqueeze(0), mode)
            logits, baseline_logits = logits[0], baseline_logits[0]
            token = sample_token(logits, rng, temperature,
                conversation.SAMPLING['top_k'] if self.conversation_profile else 40,
                conversation.SAMPLING['top_p'] if self.conversation_profile else 1.0,
                ids+generated, conversation.SAMPLING['repetition_penalty'] if self.conversation_profile else 1.0)
            if token == self.tokenizer.eos_token_id:
                break
            generated.append(token)
            event={'type':'token','text':self.tokenizer.decode(generated, skip_special_tokens=True),
                   'tokens':len(generated),'elapsed':round(time.perf_counter()-started,3)}
            if telemetry:
                event.update(logit_delta_rms=float(torch.sqrt(torch.mean((logits-baseline_logits)**2))),
                    adapter_rms=float(torch.sqrt(torch.mean(delta**2))),graph=reservoir.telemetry())
            yield event
            if index+1>=max_tokens:break
            if cancelled is not None and cancelled.is_set(): return
            if mode != 'base':
                features = reservoir.step(self.embeddings[token], mode)
            result = self.base.model(torch.tensor([[token]],device=self.device), past_key_values=past, use_cache=True)
            hidden, past = result.last_hidden_state[:, -1], result.past_key_values
        yield {'type':'done','tokens':len(generated),'elapsed':round(time.perf_counter()-started,3),
               'stop_reason':'eos' if len(generated)<max_tokens else 'token_limit',
               'text':self.tokenizer.decode(generated,skip_special_tokens=True)}


class EmbeddingView:
    """Gather only requested rows; avoid duplicating a whole vocabulary in RAM."""
    def __init__(self, weight):
        self.weight = weight.detach()

    def __getitem__(self, index):
        if isinstance(index, np.ndarray):
            index = torch.as_tensor(index, device=self.weight.device)
        return self.weight[index].float().cpu().numpy()
