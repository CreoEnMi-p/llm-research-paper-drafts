"""Independent forward and gradient checks for the GDN2 equation-level reference.

Not an implementation-equivalence test against NVIDIA's Triton kernels.
"""
import unittest
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from mix import gdn2_equation_reference


def model_state(D, H=1):
    d = H*D
    weights = {}
    for name in ("g2be", "g2wv", "g2dt"):
        weights[name] = nn.Parameter(torch.zeros(1,d,d,dtype=torch.float64))
    for name, bias in (("g2beb",0.),("g2wvb",0.),("g2dtb",-30.)):
        weights[name] = nn.Parameter(torch.full((1,d),bias,dtype=torch.float64))
    weights["g2rate"] = nn.Parameter(torch.zeros(1,H,dtype=torch.float64))
    weights["on"] = nn.Parameter(torch.ones(1,H,D,dtype=torch.float64))
    return SimpleNamespace(S=nn.ParameterDict(weights))


class EquationGateTest(unittest.TestCase):
    def inputs(self):
        q = torch.tensor([[[[1.,0.],[1.,0.]]]],dtype=torch.float64)
        k = q.clone()
        v = torch.tensor([[[[1.,0.],[0.,1.]]]],dtype=torch.float64)
        h = torch.zeros(1,2,2,dtype=torch.float64)
        return q,k,v,h

    def test_overwrite_is_distinct_from_write(self):
        q,k,v,h=self.inputs()
        m=model_state(2)
        with torch.no_grad():
            m.S["g2beb"].fill_(40)
            m.S["g2wvb"].fill_(40)
        y=gdn2_equation_reference(m,None,0,h,q,k,v)
        # Full erase removes the old e1 -> [1,0] relation when revised.
        expected=torch.tensor([[[2**0.5,0],[0,2**0.5]]],dtype=torch.float64)
        self.assertTrue(torch.allclose(y,expected,atol=1e-9,rtol=1e-9),(y,expected))
        with torch.no_grad():
            m.S["g2beb"].fill_(-40)
        no_erase=gdn2_equation_reference(m,None,0,h,q,k,v)
        self.assertTrue(torch.allclose(no_erase[:,1],torch.ones(1,2,dtype=torch.float64),
                                       atol=1e-9,rtol=1e-9))

    def test_no_write_gate(self):
        q,k,v,h=self.inputs()
        m=model_state(2)
        with torch.no_grad():
            m.S["g2wvb"].fill_(-40)
        y=gdn2_equation_reference(m,None,0,h,q,k,v)
        self.assertTrue(torch.allclose(y,torch.zeros_like(y),atol=1e-9))

    def test_finite_and_nonzero_gate_gradients(self):
        torch.manual_seed(314)
        B,H,T,D=2,2,7,4
        m=model_state(D,H)
        with torch.no_grad():
            m.S["g2beb"].fill_(0.1)
            m.S["g2wvb"].fill_(-0.2)
            m.S["g2dtb"].fill_(-3.0)
            for name in ("g2be","g2wv","g2dt"):
                m.S[name].normal_(0,0.05)
        h=torch.randn(B,T,H*D,dtype=torch.float64,requires_grad=True)
        q=F.normalize(torch.randn(B,H,T,D,dtype=torch.float64),dim=-1)
        k=F.normalize(torch.randn(B,H,T,D,dtype=torch.float64),dim=-1)
        v=torch.randn(B,H,T,D,dtype=torch.float64,requires_grad=True)
        y=gdn2_equation_reference(m,None,0,h,q,k,v)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertTrue(torch.isfinite(v.grad).all())
        for name in ("g2be","g2wv","g2dt","g2rate"):
            grad=m.S[name].grad
            self.assertIsNotNone(grad,name)
            self.assertTrue(torch.isfinite(grad).all(),name)
            self.assertGreater(grad.abs().sum().item(),0.,name)


if __name__=="__main__":
    unittest.main()
