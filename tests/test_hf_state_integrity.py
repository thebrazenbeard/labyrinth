"""Prevent false matching of training controls with mutable frozen parameters.

Lightweight fake tensors are used because unit CI has no heavy torch dependency.
"""
import unittest
from labyrinth import hf_pilot as pilot


class FakeData:
    def __init__(self, data):
        self.data=data
        self.shape=(len(data),)
        self.dtype='float32'
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def flatten(self): return self
    def view(self, dtype): return self
    def numpy(self): return self
    def tobytes(self): return self.data

class Parameter(FakeData):
    def __init__(self, data, *, trainable):
        super().__init__(data)
        self.requires_grad=trainable

class Model:
    def __init__(self):
        self.frozen=Parameter(b'frozen',trainable=False)
        self.trainable=Parameter(b'trainable',trainable=True)
    def named_parameters(self):
        return [('weight',self.frozen),('lora_adapter',self.trainable)]

class Torch:
    uint8='uint8'

class SafetyTests(unittest.TestCase):
    def test_frozen_value_digest_detects_inplace_mutation(self):
        model=Model()
        digest=getattr(pilot,'_frozen_parameter_digest',lambda *_: None)
        original=digest(model,Torch)
        model.frozen=Parameter(b'changed',trainable=False)
        self.assertNotEqual(original,digest(model,Torch))
        model.trainable=Parameter(b'other',trainable=True)
        self.assertEqual(digest(model,Torch),digest(model,Torch))

    def test_gradient_clipping_fails_closed_on_nonfinite_gradients(self):
        class Utils:
            def __init__(self): self.kwargs=None
            def clip_grad_norm_(self, params, max_norm, **kwargs):
                self.kwargs=kwargs
                return 0
        class TorchStub:
            def __init__(self):
                self.nn=type('NN',(),{})()
                self.nn.utils=Utils()
        torch=TorchStub()
        clip=getattr(pilot,'_clip_trainable_gradients',lambda *_: None)
        clip(torch,[1,2])
        self.assertEqual(torch.nn.utils.kwargs,{'error_if_nonfinite':True})


if __name__=='__main__':
    unittest.main()
