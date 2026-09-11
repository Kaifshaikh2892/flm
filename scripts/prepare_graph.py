"""Build from public MaleCNS feather files, optionally supplied locally.

Sources are streamed in
Arrow record batches, avoiding a multi-gigabyte decompression on a 16 GB laptop.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from scipy import sparse
from flm.graph import sha256
from flm.paths import GRAPH, CACHE

SOURCES = {
 'annotations.feather': ('body-annotations-male-cns-v1.0-minconf-0.5.feather', '2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2'),
 'edges.feather': ('connectome-weights-male-cns-v1.0-minconf-0.5.feather', 'e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1')
}
BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'

def prepare(source):
    source.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for local, (remote, digest) in SOURCES.items():
        path = source / local
        if not path.exists():
            print(f'Downloading {local}…', flush=True)
            temporary = path.with_suffix('.partial')
            urllib.request.urlretrieve(BASE + remote, temporary)
            if sha256(temporary) != digest:
                temporary.unlink()
                raise ValueError('Source hash mismatch.')
            temporary.replace(path)
        if sha256(path) != digest:
            raise ValueError(f'Unexpected source version: {local}')
        hashes[local] = {'url': BASE + remote, 'sha256': digest}
    annotations = feather.read_table(source / 'annotations.feather', columns=['bodyId','superclass','status'])
    ids = np.asarray(annotations['bodyId'], np.int64)
    keep = np.array([bool(x) for x in annotations['superclass'].to_pylist()])
    keep &= np.array([x != 'Glia' for x in annotations['status'].to_pylist()])
    ids = np.sort(ids[keep])
    assert len(ids) == 166700 and len(np.unique(ids)) == len(ids)
    pre, post, count = [], [], []
    rows = 0
    with pa.memory_map(str(source / 'edges.feather'), 'r') as mapped:
        reader = pa.ipc.open_file(mapped)
        for i in range(reader.num_record_batches):
            b = reader.get_batch(i)
            a, z, w = [np.asarray(b.column(b.schema.get_field_index(k))) for k in ['body_pre','body_post','weight']]
            ai, zi = np.searchsorted(ids, a), np.searchsorted(ids, z)
            valid = (ai < len(ids)) & (zi < len(ids))
            valid &= (ids[np.minimum(ai, len(ids)-1)] == a) & (ids[np.minimum(zi, len(ids)-1)] == z)
            pre.append(ai[valid].astype(np.int32)); post.append(zi[valid].astype(np.int32))
            count.append(w[valid].astype(np.int32)); rows += len(a)
    pre, post, count = map(np.concatenate, (pre, post, count))
    assert len(pre) == 25582938 and int(count.sum(dtype=np.int64)) == 124177617
    W = sparse.coo_matrix((count.astype(np.float32), (post, pre)), shape=(len(ids), len(ids))).tocsr()
    assert W.nnz == len(pre), 'Unexpected duplicate directed edges; inspect before proceeding.'
    del pre, post, count
    sums = np.asarray(W.sum(axis=1)).ravel()
    W.data /= np.repeat(np.maximum(sums, 1), np.diff(W.indptr))
    GRAPH.mkdir(parents=True, exist_ok=True)
    for name, a in [('ids',ids),('data',W.data),('indices',W.indices),('indptr',W.indptr)]:
        np.save(GRAPH / (name + '.npy'), a)
    manifest = {'release':'MaleCNS v1.0', 'neurons':len(ids), 'directed_edges':W.nnz,
                'synaptic_contacts':124177617, 'raw_edge_rows':rows,
                'retention':'Annotated nonempty superclass, excluding status Glia; both endpoints retained.',
                'matrix_orientation':'row=postsynaptic; column=presynaptic',
                'weights':'Unsigned contact counts, divided by total incoming contacts per neuron.',
                'biological_dynamics':False, 'source_files':hashes,
                'arrays':{p.name:sha256(p) for p in GRAPH.glob('*.npy')}}
    (GRAPH / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['release','neurons','directed_edges','synaptic_contacts']}),flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=CACHE/'source')
    prepare(p.parse_args().source)
