import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
import torch
from transformers import AutoTokenizer
from flm.graph import Reservoir, Connectome, sha256
from flm.model import FlyAdapter, SYSTEM
from flm.data import encode
from flm.paths import MODEL


def graph():
    return SimpleNamespace(matrix=csr_matrix(np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)),
                           ids=np.array([2**53+1,2**53+3,2**53+5],np.int64))


class DynamicsTests(unittest.TestCase):
    def test_direction_propagates_pre_to_post(self):
        r=Reservoir(graph(),2,dimensions=2)
        r.input_projection=np.eye(2,dtype=np.float32)
        r.input_bins=np.array([0,1,1]);r.input_sign=np.ones(3,np.float32)
        r.step(np.array([1,0],np.float32))
        self.assertEqual(r.state[0],0)
        self.assertGreater(r.state[1],0)
        self.assertEqual(r.state[2],0)
        r.step(np.zeros(2,np.float32))
        self.assertEqual(r.state[1],0)
        self.assertGreater(r.state[2],0)

    def test_no_future_input_and_reset(self):
        r=Reservoir(graph(),5,dimensions=2)
        inputs=np.random.default_rng(4).normal(size=(8,5)).astype(np.float32)
        full=r.sequence(inputs)
        np.testing.assert_array_equal(full[:4],r.sequence(inputs[:4]))
        np.testing.assert_array_equal(full,r.sequence(inputs))

    def test_disconnect_exactly_zero_and_no_mutation(self):
        g=graph();before=g.matrix.toarray().copy();r=Reservoir(g,2,dimensions=8)
        for mode in ['intact','shuffled','no_edges']:
            values=r.sequence(np.ones((8,2),np.float32),mode)
            self.assertTrue(np.isfinite(values).all())
            if mode=='no_edges':self.assertEqual(np.count_nonzero(values),0)
        np.testing.assert_array_equal(before,g.matrix.toarray())

    def test_neuron_ids_not_rounded(self):
        r=Reservoir(graph(),2,dimensions=2)
        self.assertEqual(r.telemetry()['sampled_neuron_ids'][0],str(2**53+1))

    def test_relabeling_is_reproducible(self):
        r=Reservoir(graph(),2,dimensions=2)
        inputs=np.ones((3,2),np.float32)
        np.testing.assert_array_equal(r.sequence(inputs,'shuffled'),r.sequence(inputs,'shuffled'))


class BatchTests(unittest.TestCase):
    def test_batched_conversations_match_independent_states(self):
        from flm.graph import ReservoirBatch
        inputs=np.random.default_rng(17).normal(size=(12,3,5)).astype(np.float32)
        for mode in ['intact','shuffled','no_edges']:
            reference=[Reservoir(graph(),5,dimensions=8) for _ in range(3)]
            batch=ReservoirBatch(Reservoir(graph(),5,dimensions=8),3)
            for embedding in inputs:
                expected=np.stack([r.step(e,mode) for r,e in zip(reference,embedding)])
                np.testing.assert_allclose(batch.step(embedding,mode),expected,rtol=2e-5,atol=2e-6)
                for j,r in enumerate(reference):np.testing.assert_allclose(batch.state[:,j],r.state,rtol=2e-5,atol=2e-6)

class LearningTests(unittest.TestCase):
    def test_adapter_learns_but_disconnect_stays_zero(self):
        torch.manual_seed(12);adapter=FlyAdapter(7,features=3,width=5)
        features=torch.randn(30,3);target=torch.full((30,7),0.1)
        optimizer=torch.optim.Adam(adapter.parameters(),lr=0.02)
        initial=float(((adapter(features)-target)**2).mean().detach())
        for _ in range(30):
            optimizer.zero_grad();loss=((adapter(features)-target)**2).mean();loss.backward();optimizer.step()
        self.assertLess(float(loss.detach()),initial)
        self.assertEqual(torch.count_nonzero(adapter(torch.zeros(4,3))).item(),0)
        self.assertLessEqual(float(adapter(features).abs().max().detach()),0.15)

    @unittest.skipUnless((MODEL/'tokenizer.json').exists(),'Run scripts/download.py first')
    def test_mask_targets_assistant_content_not_user(self):
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        messages=[{'role':'user','content':'UNIQUE_USER_123'}, {'role':'assistant','content':'A prism separates colors.'},
                  {'role':'user','content':'SECOND_USER_456'}, {'role':'assistant','content':'Rain refracts light.'}]
        ids,labels,mask=encode(tokenizer,messages,512)
        rendered=tokenizer.decode(labels[mask])
        self.assertIn('A prism separates colors.',rendered)
        self.assertIn('Rain refracts light.',rendered)
        self.assertNotIn('UNIQUE_USER',rendered)
        self.assertNotIn('SECOND_USER',rendered)
        np.testing.assert_array_equal(ids[1:],labels[:-1])


class IntegrityTests(unittest.TestCase):
    def test_corrupted_graph_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'data.npy').write_bytes(b'not an array')
            (root/'manifest.json').write_text(json.dumps({'arrays':{'data.npy':'0'*64}}))
            with self.assertRaisesRegex(ValueError,'integrity'):
                Connectome(root)


if __name__=='__main__':unittest.main()
