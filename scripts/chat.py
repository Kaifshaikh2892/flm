"""Run the trained FLM locally in a terminal. No server or conversation logging."""
import argparse
from pathlib import Path
import secrets
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flm.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/conversation-v2')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--prompt', help='Answer once and exit; omit for interactive chat.')
    parser.add_argument('--seed', type=int, help='Fix sampling for reproducible checks.')
    parser.add_argument('--max-tokens', type=int, default=160)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 512:
        parser.error('--max-tokens must be between 1 and 512')
    checkpoint = args.run / 'adapter.safetensors'
    if not checkpoint.is_file() or not (args.run / 'run.json').is_file():
        parser.error('No completed training run. Run python scripts/train_conversation.py first, or select --run.')

    from flm.model import FLM
    print('Loading the trained model and fly graph…', file=sys.stderr, flush=True)
    model = FLM(device=args.device, checkpoint=checkpoint, profile='conversation')
    if not model.trained:
        raise RuntimeError('A trained graph readout is required.')
    if args.prompt is None:
        print('Type /new to clear this conversation, /quit to exit. Chats are not saved.')
    history = []
    while True:
        try:
            prompt = args.prompt if args.prompt is not None else input('\nYou: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            if args.prompt is not None:
                parser.error('--prompt cannot be empty')
            continue
        if args.prompt is None and prompt == '/quit':
            return
        if args.prompt is None and prompt == '/new':
            history.clear()
            print('New conversation.')
            continue
        pending = history + [{'role': 'user', 'content': prompt}]
        # Trim whole earlier exchanges, never half a turn or the current prompt.
        trimmed = False
        while True:
            try:
                model.prompt_ids(pending)
                break
            except ValueError:
                if len(pending) <= 1:
                    if args.prompt is not None:
                        parser.error('This prompt exceeds the model context. Try a shorter prompt.')
                    print('That message is too long. Try a shorter message.')
                    pending = None
                    break
                pending = pending[2:]
                trimmed = True
        if pending is None:
            continue
        if trimmed:
            print('(Older exchanges left the context window.)', file=sys.stderr)
        print('FLM: ', end='', flush=True)
        text = ''
        try:
            for event in model.generate(pending, mode='intact', max_tokens=args.max_tokens,
                                        seed=args.seed if args.seed is not None else secrets.randbits(31), telemetry=False):
                if event['type'] == 'token':
                    new_text = event['text']
                    if new_text.startswith(text):
                        print(new_text[len(text):], end='', flush=True)
                    text = new_text
                elif event['type'] == 'done':
                    text = event['text']
        except KeyboardInterrupt:
            print('\nReply stopped. Conversation unchanged.')
            if args.prompt is not None:
                return
            continue
        print()
        if text.strip():
            history = pending + [{'role': 'assistant', 'content': text}]
        if args.prompt is not None:
            return


if __name__ == '__main__':
    main()
