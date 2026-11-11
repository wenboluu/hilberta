"""CPU property mocks verify paired-wrapper guards without initializing CUDA."""

import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import torch
from hilberta.paired_kernel import paired_attention
from tests.test_sparse_kernel_contract import TensorProperties


class PairedContract(unittest.TestCase):
    def setUp(self):
        shape = (1, 2, 512, 128)
        native = (131072, 128, 256, 1)
        self.q = TensorProperties(shape, native, pointer=1_000_000)
        self.k = TensorProperties(shape, pointer=2_000_000)
        self.v = TensorProperties(shape, native, pointer=3_000_000)
        self.meta = SimpleNamespace(
            tokens=512,
            global_tokens=128,
            key_block_size=64,
            kv_indices=TensorProperties((3, 5), dtype=torch.int32, pointer=4_000_000),
            kv_flags=TensorProperties((3, 5), dtype=torch.int32, pointer=5_000_000),
            kv_counts=TensorProperties((3,), dtype=torch.int32, pointer=6_000_000),
        )
        self.out = TensorProperties(shape, native, pointer=7_000_000)

    def call(self, **kwargs):
        with patch("hilberta.paired_kernel._paired_fwd", MagicMock()):
            return paired_attention(
                self.q, self.k, self.v, self.meta, out=self.out, global_strategy="triton", **kwargs
            )

    def test_mixed_stride_native_output(self):
        self.assertIs(self.call(), self.out)

    def test_input_and_metadata_output_alias_rejected(self):
        for source in (
            self.q,
            self.k,
            self.v,
            self.meta.kv_indices,
            self.meta.kv_flags,
            self.meta.kv_counts,
        ):
            with self.subTest(source=source.pointer), self.assertRaisesRegex(ValueError, "alias"):
                self.out.pointer = source.pointer
                self.call()

    def test_partial_alias(self):
        self.out.pointer = self.k.pointer + 64
        with self.assertRaisesRegex(ValueError, "alias"):
            self.call()

    def test_overlap_and_zero_stride_output(self):
        for strides in ((131072, 128, 128, 1), (0, 128, 256, 1)):
            self.out.strides = strides
            with self.assertRaisesRegex(ValueError, "non-overlapping"):
                self.call()

    def test_grad_inputs_outputs(self):
        self.q.requires_grad = True
        with torch.enable_grad(), self.assertRaisesRegex(ValueError, "Inference"):
            self.call()
        with torch.no_grad():
            self.call()
        self.q.requires_grad = False
        self.out.requires_grad = True
        with torch.enable_grad(), self.assertRaisesRegex(ValueError, "gradients"):
            self.call()

    def test_tensor_scale_not_transferred(self):
        with self.assertRaisesRegex(TypeError, "never a tensor"):
            self.call(scale=torch.tensor(1.0))

    def test_tuning_rejects_noninteger(self):
        for value in (4.0, True):
            with self.assertRaisesRegex(TypeError, "Integer"):
                self.call(num_warps=value)

    def test_wrong_device(self):
        self.meta.kv_flags.device = torch.device("cuda:1")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            self.call()

    def test_metadata_shapes(self):
        self.meta.kv_flags.shape = (3, 4)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.call()

    def test_metadata_alignment(self):
        self.meta.global_tokens = 64
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.call()

    def test_negative_input_stride(self):
        self.k.strides = (131072, -128, 256, 1)
        with self.assertRaisesRegex(ValueError, "Positive"):
            self.call()


if __name__ == "__main__":
    unittest.main()
