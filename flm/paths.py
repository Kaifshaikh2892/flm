from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'cache'
GRAPH = CACHE / 'malecns_v1'
MODEL = CACHE / 'language-model'
RUN = ROOT / 'runs' / 'prototype'

MODEL_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct'
MODEL_REVISION = '12fd25f77366fa6b3b4b768ec3050bf629380bac'
DATA_ID = 'HuggingFaceTB/smoltalk'
DATA_REVISION = '5feaf2fd3ffca7c237fc38d1861bc30365d48ffa'
