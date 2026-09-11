"""Exact retained topology, abstract rate dynamics, explicit causal controls.

Every retained directed edge participates. W[post, pre] is contact count divided
by total incoming contacts. This deliberately does NOT infer transmitter signs,
synaptic efficacy, spikes, dopamine, or biological time from anatomy.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import sparse

from .paths import GRAPH


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Connectome:
    def __init__(self, folder=GRAPH, verify=True):
        folder = Path(folder)
        self.manifest = json.loads((folder / 'manifest.json').read_text())
        if verify:
            for name, digest in self.manifest['arrays'].items():
                if sha256(folder / name) != digest:
                    raise ValueError(f'Graph integrity check failed: {name}')
        self.ids = np.load(folder / 'ids.npy', mmap_mode='r')
        if self.ids.dtype != np.int64 or np.any(np.diff(self.ids) <= 0):
            raise ValueError('Neuron IDs must be sorted, exact int64 values.')
        arrays = [np.load(folder / (name + '.npy'), mmap_mode='r')
                  for name in ['data', 'indices', 'indptr']]
        self.matrix = sparse.csr_matrix(tuple(arrays), shape=(len(self.ids), len(self.ids)), copy=False)
        if self.matrix.nnz != self.manifest['directed_edges']:
            raise ValueError('Edge count differs from manifest.')


class Reservoir:
    """One causal graph update per token. No future tokens or answer labels.

    x_t = tanh(W @ (0.6*x_(t-1) + 0.4*B*embedding_t))
    f_t = normalize(P*x_t)

    B and P are fixed seeded sparse random interfaces, not anatomical language
    pathways. A no-edges control produces exact zero features. Shuffled is a
    fixed node relabeling of W relative to B/P: it preserves topology and tests
    interface alignment, NOT superiority over a random graph.
    """
    def __init__(self, graph, embedding_dim, dimensions=128, seed=7301):
        self.graph = graph
        self.n = graph.matrix.shape[0]
        self.dimensions = dimensions
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.input_projection = (rng.standard_normal((embedding_dim, dimensions)) /
                                 np.sqrt(embedding_dim)).astype(np.float32)
        self.input_bins = rng.integers(0, dimensions, self.n)
        self.input_sign = rng.choice(np.array([-1, 1], np.float32), self.n)
        self.output_bins = rng.integers(0, dimensions, self.n)
        self.output_sign = rng.choice(np.array([-1, 1], np.float32), self.n)
        self.output_scale = np.sqrt(np.maximum(1, np.bincount(self.output_bins, minlength=dimensions))).astype(np.float32)
        self.permutation = rng.permutation(self.n)
        self.inverse = np.argsort(self.permutation)
        self.state = np.zeros(self.n, np.float32)
        self.tokens = 0

    def reset(self):
        self.state.fill(0)
        self.tokens = 0

    def project_input(self, embedding):
        """Matched direct-input baseline; contains no fly graph computation."""
        code = np.asarray(embedding, np.float32) @ self.input_projection
        return code / np.sqrt(np.mean(code * code) + 1e-6)

    def step(self, embedding, mode='intact'):
        if mode not in ('intact', 'no_edges', 'shuffled'):
            raise ValueError('Unknown graph control.')
        code = self.project_input(embedding)
        drive = 0.6 * self.state + 0.4 * code[self.input_bins] * self.input_sign
        if mode == 'no_edges':
            self.state.fill(0)
        elif mode == 'shuffled':
            self.state = np.tanh((self.graph.matrix @ drive[self.permutation])[self.inverse])
        else:
            self.state = np.tanh(self.graph.matrix @ drive)
        self.tokens += 1
        f = np.bincount(self.output_bins, weights=self.state * self.output_sign,
                        minlength=self.dimensions).astype(np.float32) / self.output_scale
        # Bias-free scaling: disconnecting the graph cannot generate an adapter signal.
        f /= np.sqrt(np.mean(f * f) + 1e-6)
        return f

    def sequence(self, embeddings, mode='intact'):
        self.reset()
        return np.stack([self.step(e, mode) for e in embeddings])

    def telemetry(self):
        indices = np.linspace(0, self.n - 1, 96).astype(int)
        return {'updates': self.tokens, 'state_rms': float(np.sqrt(np.mean(self.state ** 2))),
                'sampled_neuron_ids': [str(int(self.graph.ids[i])) for i in indices],
                'sampled_states': self.state[indices].tolist(), 'units': 'abstract rate-model state'}


class ReservoirBatch:
    """Independent conversations as matrix columns; no cross-column connections.

    Used for offline extraction only. Each column is the same full-graph
    recurrence as Reservoir; batching reuses sparse matrix reads.
    """
    def __init__(self, reservoir, size, state=None, kernel=None):
        self.r = reservoir
        self.size = size
        self.kernel = kernel
        self.state = np.repeat((reservoir.state if state is None else state)[:,None],size,axis=1)

    def step(self, embeddings, mode='intact'):
        r = self.r
        code = np.asarray(embeddings,np.float32) @ r.input_projection
        code /= np.sqrt(np.mean(code*code,axis=1,keepdims=True)+1e-6)
        drive = .6*self.state + .4*code[:,r.input_bins].T*r.input_sign[:,None]
        if mode == 'intact':self.state = np.tanh(self.kernel(r.graph.matrix,drive) if self.kernel else r.graph.matrix @ drive)
        elif mode == 'shuffled':self.state = np.tanh((self.kernel(r.graph.matrix,drive[r.permutation]) if self.kernel else r.graph.matrix @ drive[r.permutation])[r.inverse])
        elif mode == 'no_edges':self.state.fill(0)
        else:raise ValueError('Unknown graph control.')
        values = np.stack([np.bincount(r.output_bins,weights=self.state[:,j]*r.output_sign,minlength=r.dimensions).astype(np.float32)/r.output_scale for j in range(self.size)])
        values /= np.sqrt(np.mean(values*values,axis=1,keepdims=True)+1e-6)
        return values


class NativeBatchKernel:
    """Load only a locally built, source/hash-verified optional training kernel."""
    def __init__(self, folder):
        import ctypes
        folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
        if sha256(folder/'graph-batch.so')!=manifest['binary_sha256'] or sha256(Path(__file__).resolve().parents[1]/'native/graph_batch.c')!=manifest['source_sha256']:
            raise ValueError('Native graph kernel integrity mismatch.')
        self.function=ctypes.CDLL(str(folder/'graph-batch.so')).flm_csr8
        self.function.argtypes=[ctypes.c_int32]+[ctypes.c_void_p]*5
        self.function.restype=None

    def __call__(self, matrix, drive):
        assert matrix.indptr.dtype==np.int32 and matrix.indices.dtype==np.int32 and matrix.data.dtype==np.float32
        assert 1<=drive.shape[1]<=8 and drive.shape[0]==matrix.shape[1]
        # Pad lanes, not nodes or edges. Padded lanes are discarded.
        x=np.zeros((len(drive),8),np.float32);x[:,:drive.shape[1]]=drive;y=np.empty_like(x)
        self.function(matrix.shape[0],matrix.indptr.ctypes.data,matrix.indices.ctypes.data,matrix.data.ctypes.data,x.ctypes.data,y.ctypes.data)
        return y[:,:drive.shape[1]]
