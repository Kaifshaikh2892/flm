"""Documented probability sampling and a bounded, differentiable graph readout."""
import torch


def bound_logit_delta(delta, maximum):
    """Limit per-position RMS; zero input remains exactly zero."""
    if maximum is None:
        return delta
    rms = (delta.float().square().mean(-1, keepdim=True) + 1e-12).sqrt()
    return delta * (maximum / rms).clamp(max=1)


def sample_token(logits, generator, temperature=0.7, top_k=20, top_p=0.8, previous_tokens=(), repetition_penalty=1.0):
    if previous_tokens and repetition_penalty != 1.0:
        logits = logits.clone()
        seen = torch.tensor(sorted(set(previous_tokens)), device=logits.device)
        values = logits[seen]
        logits[seen] = torch.where(values < 0, values * repetition_penalty, values / repetition_penalty)
    if temperature <= 0:
        return int(logits.argmax())
    values, choices = torch.topk(logits.float() / temperature, min(top_k, logits.numel()))
    probabilities = torch.softmax(values, -1)
    # Keep the first token crossing the nucleus threshold, as well as all before it.
    remove = probabilities.cumsum(-1) - probabilities > top_p
    probabilities = probabilities.masked_fill(remove, 0)
    selected = int(torch.multinomial(probabilities.cpu(), 1, generator=generator))
    return int(choices[selected])
