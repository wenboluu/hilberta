"""CPU metadata/property mocks: these tests never allocate or initialize CUDA."""

import unittest
import torch
from hilberta.sparse_kernel import _check_inputs, sparse_attention


class TensorProperties:
    def __init__(
        self,
        shape,
        strides=None,
        dtype=torch.bfloat16,
        pointer=1_000_000,
        device="cuda:0",
        requires_grad=False,
    ):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        self.requires_grad = requires_grad
        self.pointer = pointer
        if strides is None:
            strides = []
            running = 1
            for size in reversed(shape):
                strides.insert(0, running)
                running *= size
        self.strides = tuple(strides)

    def stride(self, dim=None):
        return self.strides if dim is None else self.strides[dim]

    def data_ptr(self):
        return self.pointer

    def element_size(self):
        return 4 if self.dtype == torch.int32 else 2

    def is_contiguous(self):
        running = 1
        for size, stride in zip(reversed(self.shape), reversed(self.strides)):
            if size > 1 and stride != running:
                return False
            running *= size
        return True


class SparseContract(unittest.TestCase):
    def setUp(self):
        shape = (1, 2, 512, 128)
        native = (131072, 128, 256, 1)
        self.q = TensorProperties(shape, native, pointer=1_000_000)
        self.k = TensorProperties(shape, pointer=2_000_000)
        self.v = TensorProperties(shape, native, pointer=3_000_000)
        self.ids = TensorProperties((6, 5), dtype=torch.int32, pointer=4_000_000)
        self.counts = TensorProperties((6,), dtype=torch.int32, pointer=5_000_000)
        self.out = TensorProperties(shape, native, pointer=6_000_000)

    def check(self, **changes):
        values = dict(
            q=self.q,
            k=self.k,
            v=self.v,
            kv_indices=self.ids,
            kv_counts=self.counts,
            global_tokens=128,
            block_size=64,
            out=self.out,
        )
        values.update(changes)
        return _check_inputs(**values)

    def test_native_output_and_mixed_input_strides(self):
        self.check()

    def test_head_major_output(self):
        self.check(out=TensorProperties(self.q.shape, pointer=6_000_000))

    def test_input_alias_is_rejected(self):
        self.out.pointer = self.q.pointer
        with self.assertRaisesRegex(ValueError, "alias"):
            self.check()

    def test_partial_alias_is_rejected(self):
        self.out.pointer = self.k.pointer + 1024
        with self.assertRaisesRegex(ValueError, "alias"):
            self.check()

    def test_metadata_alias_is_rejected(self):
        self.out.pointer = self.ids.pointer
        with self.assertRaisesRegex(ValueError, "alias"):
            self.check()

    def test_overlapping_output_is_rejected(self):
        self.out.strides = (131072, 128, 128, 1)
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            self.check()

    def test_zero_stride_output_is_rejected(self):
        self.out.strides = (0, 128, 256, 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            self.check()

    def test_gradient_input_rejected_and_no_grad_allowed(self):
        self.q.requires_grad = True
        with torch.enable_grad(), self.assertRaisesRegex(ValueError, "inference-only"):
            self.check()
        with torch.no_grad():
            self.check()

    def test_gradient_output_rejected(self):
        self.out.requires_grad = True
        with torch.enable_grad(), self.assertRaisesRegex(ValueError, "gradients"):
            self.check()

    def test_noninteger_global_rejected(self):
        for value in (128.0, True):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.check(global_tokens=value)

    def test_empty_batch_rejected(self):
        self.q.shape = self.k.shape = self.v.shape = (0, 2, 512, 128)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.check(out=None)

    def test_tensor_scale_rejected_without_data_transfer(self):
        with self.assertRaisesRegex(TypeError, "never a tensor"):
            sparse_attention(
                self.q,
                self.k,
                self.v,
                self.ids,
                self.counts,
                global_tokens=128,
                scale=torch.tensor(1.0),
            )

    def test_float_launch_config_rejected(self):
        with self.assertRaisesRegex(TypeError, "tuning"):
            sparse_attention(
                self.q, self.k, self.v, self.ids, self.counts, global_tokens=128, num_warps=4.0
            )

    def test_wrong_device_rejected(self):
        self.v.device = torch.device("cuda:1")
        with self.assertRaisesRegex(ValueError, "one CUDA"):
            self.check()


if __name__ == "__main__":
    unittest.main()
