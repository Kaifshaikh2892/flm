import unittest
import torch
from flm.decoding import bound_logit_delta,sample_token

class DecodingTests(unittest.TestCase):
    def test_bound_is_zero_preserving_and_differentiable_at_initialization(self):
        delta=torch.zeros(3,7,requires_grad=True)
        bounded=bound_logit_delta(delta,.25)
        self.assertEqual(torch.count_nonzero(bounded).item(),0)
        bounded.sum().backward()
        self.assertTrue(torch.isfinite(delta.grad).all())
        self.assertTrue(torch.all(delta.grad==1))
    def test_large_graph_signal_is_bounded_per_position(self):
        delta=torch.tensor([[20.,-10.,4.],[0.01,-0.02,.03]],requires_grad=True)
        bounded=bound_logit_delta(delta,.25)
        self.assertTrue((bounded.square().mean(-1).sqrt()<=.250001).all())
        torch.testing.assert_close(bounded[1],delta[1])
        bounded.square().sum().backward()
        self.assertTrue(torch.isfinite(delta.grad).all())
    def test_research_seed_is_reproducible(self):
        logits=torch.linspace(-1,1,30)
        def draws(seed):
            rng=torch.Generator().manual_seed(seed)
            return [sample_token(logits,rng) for _ in range(20)]
        self.assertEqual(draws(42),draws(42))
        self.assertNotEqual(draws(42),draws(123))
    def test_repetition_penalty_does_not_mutate_model_scores(self):
        logits=torch.tensor([2.,1.9,-2.]);before=logits.clone()
        # A seen positive token is demoted; the underlying logits remain intact.
        token=sample_token(logits,torch.Generator(),temperature=0,previous_tokens=[0],repetition_penalty=1.2)
        self.assertEqual(token,1)
        torch.testing.assert_close(logits,before)
    def test_nucleus_keeps_the_crossing_token(self):
        logits=torch.log(torch.tensor([.6,.3,.1]))
        rng=torch.Generator().manual_seed(3)
        draws={sample_token(logits,rng,temperature=1,top_k=3,top_p=.8) for _ in range(100)}
        self.assertEqual(draws,{0,1})

if __name__=='__main__':unittest.main()
