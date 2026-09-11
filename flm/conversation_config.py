"""Pinned conversation model and graph-interface settings."""
from .paths import CACHE

MODEL_ID = 'LiquidAI/LFM2.5-1.2B-Instruct'
MODEL_REVISION = '0f604ada3f766f9f257460c4c9f0b5d6f69d431b'
MODEL_DIR = CACHE / 'lfm25-1.2b'
SYSTEM_TEXT = 'You are FLM. Be an engaging, inventive conversation partner. Answer directly in a few sentences. Prefer concrete ideas, dry humor, and specific details. No cheerleading, emojis, praise of the question, or routine follow-up questions. Track the conversation and follow the requested format. You are software: a pretrained language model with a trained readout of a fly connectome. Do not claim to be a living fly or invent personal experiences. Say when you do not know.'

INTERFACE = {'seed':7301, 'dimensions':128, 'input_gain':0.4, 'recurrence_gain':0.6,
             'adapter_scale':0.03, 'max_logit_rms':0.25}
CONTEXT_TOKENS = 1536
SAMPLING = {'temperature':0.4, 'top_k':50, 'top_p':0.9, 'repetition_penalty':1.05}
