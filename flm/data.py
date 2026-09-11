"""Assistant-only, causal next-token targets. No evaluation labels in features."""
import hashlib
import json
import numpy as np
from .model import SYSTEM


def conversation_hash(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def encode(tokenizer, messages, max_tokens=192):
    messages = [SYSTEM] + messages
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    mask = np.zeros(len(ids), dtype=bool)
    for i, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        prefix = tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=True)
        end = tokenizer.apply_chat_template(messages[:i+1], tokenize=True, add_generation_prompt=False)
        if ids[:len(prefix)] != prefix or ids[:len(end)] != end:
            raise ValueError('Chat template is not prefix-stable; refusing an incorrect training mask.')
        mask[len(prefix):len(end)] = True
    ids = np.asarray(ids[:max_tokens], dtype=np.int64)
    return ids[:-1], ids[1:], mask[1:len(ids)]
