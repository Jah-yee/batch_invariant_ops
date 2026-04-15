"""Tests for batch-invariant operations."""

import pytest
import torch
from batch_invariant_ops import (
    set_batch_invariant_mode,
    mm_batch_invariant,
    addmm_batch_invariant,
    log_softmax,
    mean_kernel,
)


class TestBatchInvariant:
    """Test batch-invariant property: op(x[:1], y) == op(x, y)[:1]"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Enable batch-invariant mode for all tests."""
        with set_batch_invariant_mode(True):
            yield

    def test_mm_batch_invariant(self):
        """Test mm preserves batch invariance."""
        B, D = 16, 32
        a = torch.randn(B, D)
        b = torch.randn(D, D)

        # Single batch
        out1 = mm_batch_invariant(a[:1], b)
        # Full batch, sliced
        out2 = mm_batch_invariant(a, b)[:1]

        assert torch.allclose(out1, out2)

    def test_mm_output_shape(self):
        """Test mm output has correct shape."""
        a = torch.randn(8, 16)
        b = torch.randn(16, 32)
        result = mm_batch_invariant(a, b)
        assert result.shape == (8, 32)

    def test_addmm_batch_invariant(self):
        """Test addmm preserves batch invariance."""
        B, D = 16, 32
        bias = torch.randn(D)
        a = torch.randn(B, D)
        b = torch.randn(D, D)

        # Single batch
        out1 = addmm_batch_invariant(bias, a[:1], b)
        # Full batch, sliced
        out2 = addmm_batch_invariant(bias, a, b)[:1]

        assert torch.allclose(out1, out2)

    def test_addmm_output_shape(self):
        """Test addmm output has correct shape."""
        bias = torch.randn(32)
        a = torch.randn(8, 16)
        b = torch.randn(16, 32)
        result = addmm_batch_invariant(bias, a, b)
        assert result.shape == (8, 32)

    def test_log_softmax_batch_invariant(self):
        """Test log_softmax preserves batch invariance."""
        B, D = 16, 32
        x = torch.randn(B, D)

        # Single batch
        out1 = log_softmax(x[:1], dim=-1)
        # Full batch, sliced
        out2 = log_softmax(x, dim=-1)[:1]

        assert torch.allclose(out1, out2)

    def test_log_softmax_output_shape(self):
        """Test log_softmax output has correct shape."""
        x = torch.randn(8, 16)
        result = log_softmax(x, dim=-1)
        assert result.shape == x.shape

    def test_log_softmax_values(self):
        """Test log_softmax sums to 1 in probability space."""
        x = torch.randn(4, 8)
        result = log_softmax(x, dim=-1)
        probs = torch.exp(result)
        # Sum along the softmax dimension should be ~1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0]), atol=1e-5)

    def test_mean_kernel_batch_invariant(self):
        """Test mean preserves batch invariance."""
        B, M, N = 4, 8, 16
        x = torch.randn(B, M, N)

        # Single batch
        out1 = mean_kernel(x[:1], dim=1)
        # Full batch, sliced
        out2 = mean_kernel(x, dim=1)[:1]

        assert torch.allclose(out1, out2)

    def test_mean_kernel_output_shape(self):
        """Test mean_kernel output has correct shape."""
        x = torch.randn(4, 8, 16)
        result = mean_kernel(x, dim=1)
        assert result.shape == (4, 16)

    def test_mean_kernel_various_dims(self):
        """Test mean works along different dimensions."""
        x = torch.randn(4, 8, 16)

        # Along dim 1
        result1 = mean_kernel(x, dim=1)
        assert result1.shape == (4, 16)

        # Along dim 2
        result2 = mean_kernel(x, dim=2)
        assert result2.shape == (4, 8)

    def test_mean_kernel_empty_dimension(self):
        """Test mean_kernel handles edge case of empty dimension."""
        # This tests the fix for issue #15
        x = torch.randn(1, 0, 8)
        result = mean_kernel(x, dim=1)
        # Should not produce NaN
        assert not torch.isnan(result).any() if result.numel() > 0 else True