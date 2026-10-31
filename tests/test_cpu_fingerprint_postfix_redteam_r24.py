"""Adversarial source-only checks for toy CPU optimizer-state fingerprints.

No Torch model is loaded. The fake objects preserve the public AdamW shape
that matters here: ordered groups, optimizer states keyed by parameter, and
module tensor metadata. Passing these tests cannot qualify physical training.
"""

from __future__ import annotations

import math
import unittest

from scripts import qualify_v2_checkpoint_cpu_r22 as smoke


class _Array:
    def __init__(self, data: bytes):
        self.data = data

    def tobytes(self) -> bytes:
        return self.data


class _Tensor:
    def __init__(self, data: bytes, *, dtype: str = "torch.float32",
                 shape: tuple[int, ...] = (1,)):
        self.data = data
        self.dtype = dtype
        self.shape = shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self) -> _Array:
        return _Array(self.data)


class _Module:
    def __init__(self, name: str, param: object, tensor: _Tensor):
        self.name = name
        self.param = param
        self.tensor = tensor

    def state_dict(self):
        return {"weight": self.tensor}

    def named_parameters(self):
        return [("weight", self.param)]


class _Optimizer:
    def __init__(self, params: list[object], *, betas=(0.9, 0.999),
                 eps=1e-8, amsgrad=False, maximize=False, lr=0.002,
                 weight_decay=0.01, moments: dict[object, bytes] | None = None):
        self.param_groups = [{
            "params": list(params), "betas": betas, "eps": eps,
            "amsgrad": amsgrad, "maximize": maximize,
            "lr": lr, "weight_decay": weight_decay,
        }]
        moments = moments or {p: b"\x03\x04" for p in params}
        self.state = {p: {
            "step": _Tensor(b"\x01"),
            "exp_avg": _Tensor(moments[p]),
            "exp_avg_sq": _Tensor(b"\x05\x06"),
        } for p in params}
        if amsgrad:
            for p in params:
                self.state[p]["max_exp_avg_sq"] = _Tensor(b"\x07\x08")


def _fixture(*, encoder_tensor: _Tensor | None = None,
             head_tensor: _Tensor | None = None,
             **optimizer_options):
    encoder_parameter = object()
    head_parameter = object()
    encoder = _Module("encoder", encoder_parameter,
                      encoder_tensor or _Tensor(b"\x00\x00\x80?" * 4,
                                                shape=(4,)))
    head = _Module("head", head_parameter,
                   head_tensor or _Tensor(b"\x00\x00\x00@"))
    optimizer = _Optimizer([encoder_parameter, head_parameter], **optimizer_options)
    return encoder, head, optimizer


def _fingerprint(items) -> str:
    return smoke._state_fingerprint(None, *items)


class CpuFingerprintPostfixRedteamTests(unittest.TestCase):
    def test_adamw_beta_and_epsilon_change_fingerprint(self):
        baseline = _fingerprint(_fixture())
        for changed in ({"betas": (0.8, 0.999)},
                        {"betas": (0.9, 0.98)},
                        {"eps": 1e-5}):
            with self.subTest(changed=changed):
                self.assertNotEqual(baseline, _fingerprint(_fixture(**changed)))

    def test_adamw_mode_flags_change_fingerprint(self):
        baseline = _fingerprint(_fixture())
        for changed in ({"amsgrad": True}, {"maximize": True}):
            with self.subTest(changed=changed):
                self.assertNotEqual(baseline, _fingerprint(_fixture(**changed)))

    def test_tensor_dtype_and_shape_cannot_alias_identical_raw_bytes(self):
        baseline = _fingerprint(_fixture())
        for changed in (
            _Tensor(b"\x00\x00\x80?" * 4,
                    dtype="torch.uint8", shape=(16,)),
            _Tensor(b"\x00\x00\x80?" * 4, shape=(2, 2)),
        ):
            with self.subTest(dtype=changed.dtype, shape=changed.shape):
                self.assertNotEqual(
                    baseline, _fingerprint(_fixture(encoder_tensor=changed)))

    def test_moment_to_parameter_association_is_committed(self):
        p_encoder, p_head = object(), object()
        encoder = _Module("encoder", p_encoder, _Tensor(b"enc0"))
        head = _Module("head", p_head, _Tensor(b"head"))
        original = _Optimizer([p_encoder, p_head], moments={
            p_encoder: b"enc-moment", p_head: b"head-moment"})
        swapped = _Optimizer([p_head, p_encoder], moments={
            p_head: b"enc-moment", p_encoder: b"head-moment"})
        self.assertNotEqual(
            _fingerprint((encoder, head, original)),
            _fingerprint((encoder, head, swapped)),
            "same serialized moment order must not hide a changed parameter owner")

    def test_nonfinite_hyperparameter_is_refused(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=str(value)):
                with self.assertRaises((ValueError, TypeError)):
                    _fingerprint(_fixture(lr=value))


if __name__ == "__main__":
    unittest.main()
