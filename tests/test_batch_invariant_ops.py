"""
Test suite for batch-invariant operations.

This test suite verifies that the batch-invariant property holds for the following operations:
- mm (matrix multiplication)
- addmm (matrix multiplication with bias)
- log_softmax
- mean

The batch-invariant property states that:
    op(x[:1], y) == op(x, y)[:1]

This means computing an operation on a single batch element should give the same result
as computing it on the full batch and then taking the first element.
"""

import pytest
import torch
from batch_invariant_ops import (
    set_batch_invariant_mode,
    matmul_persistent,
    addmm_batch_invariant,
    log_softmax,
    mean_dim,
)


def get_device():
    """Get the current accelerator device."""
    device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
    if device_type == "cpu":
        return "cpu"
    elif device_type in ("cuda", "xpu"):
        return device_type
    return "cpu"


DEVICE = get_device()
DTYPES = [torch.float32, torch.float16, torch.bfloat16] if DEVICE != "cpu" else [torch.float32]


class TestBatchInvariantMM:
    """Tests for mm (matrix multiplication) batch invariance."""

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_mm_batch_invariant_basic(self, dtype):
        """Test basic batch invariance for mm operation."""
        B, M, K, N = 4, 16, 32, 16
        
        x = torch.randn(B, M, K, dtype=dtype, device=DEVICE)
        y = torch.randn(K, N, dtype=dtype, device=DEVICE)
        
        # Single batch element
        out_single = torch.mm(x[:1], y)
        # Full batch, take first
        out_full = torch.mm(x, y)[:1]
        
        torch.testing.assert_close(out_single, out_full, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_mm_batch_invariant_large(self, dtype):
        """Test batch invariance with larger matrices."""
        B, M, K, N = 8, 128, 256, 128
        
        x = torch.randn(B, M, K, dtype=dtype, device=DEVICE)
        y = torch.randn(K, N, dtype=dtype, device=DEVICE)
        
        out_single = torch.mm(x[:1], y)
        out_full = torch.mm(x, y)[:1]
        
        torch.testing.assert_close(out_single, out_full, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_mm_batch_invariant_single_row(self, dtype):
        """Test with single row in batch dimension."""
        B, M, K, N = 1, 8, 16, 8
        
        x = torch.randn(B, M, K, dtype=dtype, device=DEVICE)
        y = torch.randn(K, N, dtype=dtype, device=DEVICE)
        
        out_single = torch.mm(x[:1], y)
        out_full = torch.mm(x, y)[:1]
        
        torch.testing.assert_close(out_single, out_full, atol=1e-3, rtol=1e-3)


class TestBatchInvariantAddMM:
    """Tests for addmm (matrix multiplication with bias) batch invariance."""

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_addmm_batch_invariant_basic(self, dtype):
        """Test basic batch invariance for addmm operation."""
        B, M, K, N = 4, 16, 32, 16
        
        x = torch.randn(B, M, K, dtype=dtype, device=DEVICE)
        y = torch.randn(K, N, dtype=dtype, device=DEVICE)
        bias = torch.randn(N, dtype=dtype, device=DEVICE)
        
        # Single batch element
        out_single = torch.addmm(bias, x[:1], y)
        # Full batch, take first
        out_full = torch.addmm(bias, x, y)[:1]
        
        torch.testing.assert_close(out_single, out_full, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_addmm_batch_invariant_large(self, dtype):
        """Test batch invariance with larger matrices."""
        B, M, K, N = 8, 128, 256, 128
        
        x = torch.randn(B, M, K, dtype=dtype, device=DEVICE)
        y = torch.randn(K, N, dtype=dtype, device=DEVICE)
        bias = torch.randn(N, dtype=dtype, device=DEVICE)
        
        out_single = torch.addmm(bias, x[:1], y)
        out_full = torch.addmm(bias, x, y)[:1]
        
        torch.testing.assert_close(out_single, out_full, atol=1e-3, rtol=1e-3)


class TestBatchInvariantLogSoftmax:
    """Tests for log_softmax batch invariance."""

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_log_softmax_batch_invariant_basic(self, dtype):
        """Test basic batch invariance for log_softmax operation."""
        B, S, V = 4, 16, 32
        
        x = torch.randn(B, S, V, dtype=dtype, device=DEVICE)
        
        # Single batch element
        out_single = torch.log_softmax(x[:1], dim=-1)
        # Full batch, take first
        out_full = torch.log_softmax(x, dim=-1)[:1]
        
        torch.testing.assert_close(out_single, out_full, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_log_softmax_batch_invariant_2d(self, dtype):
        """Test batch invariance for 2D tensors."""
        B, V = 4, 64
        
        x = torch.randn(B, V, dtype=dtype, device=DEVICE)
        
        out_single = torch.log_softmax(x[:1], dim=-1)
        out_full = torch.log_softmax(x, dim=-1)[:1]
        
        torch.testing.assert_close(out_single, out_full, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_log_softmax_batch_invariant_large(self, dtype):
        """Test batch invariance with larger sequences."""
        B, S, V = 8, 128, 512
        
        x = torch.randn(B, S, V, dtype=dtype, device=DEVICE)
        
        out_single = torch.log_softmax(x[:1], dim=-1)
        out_full = torch.log_softmax(x, dim=-1)[:1]
        
        torch.testing.assert_close(out_single, out_full, atol=1e-3, rtol=1e-3)


class TestBatchInvariantMean:
    """Tests for mean operation batch invariance."""

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_mean_batch_invariant_basic(self, dtype):
        """Test basic batch invariance for mean operation."""
        B, S, V = 4, 16, 32
        
        x = torch.randn(B, S, V, dtype=dtype, device=DEVICE)
        
        # Single batch element
        out_single = torch.mean(x[:1], dim=1)
        # Full batch, take first
        out_full = torch.mean(x, dim=1)[:1]
        
        torch.testing.assert_close(out_single, out_full, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_mean_batch_invariant_dim0(self, dtype):
        """Test batch invariance for mean along dim 0."""
        B, S, V = 4, 16, 32
        
        x = torch.randn(B, S, V, dtype=dtype, device=DEVICE)
        
        out_single = torch.mean(x[:1], dim=0)
        out_full = torch.mean(x, dim=0)[:1]
        
        torch.testing.assert_close(out_single, out_full, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_mean_batch_invariant_keepdim(self, dtype):
        """Test batch invariance for mean with keepdim=True."""
        B, S, V = 4, 16, 32
        
        x = torch.randn(B, S, V, dtype=dtype, device=DEVICE)
        
        out_single = torch.mean(x[:1], dim=1, keepdim=True)
        out_full = torch.mean(x, dim=1, keepdim=True)[:1]
        
        torch.testing.assert_close(out_single, out_full, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_mean_batch_invariant_large(self, dtype):
        """Test batch invariance with larger tensors."""
        B, S, V = 8, 64, 128
        
        x = torch.randn(B, S, V, dtype=dtype, device=DEVICE)
        
        out_single = torch.mean(x[:1], dim=1)
        out_full = torch.mean(x, dim=1)[:1]
        
        torch.testing.assert_close(out_single, out_full, atol=1e-3, rtol=1e-3)


class TestBatchInvariantMode:
    """Tests for batch invariant mode context manager."""

    def test_batch_invariant_mode_enable_disable(self):
        """Test enabling and disabling batch invariant mode."""
        from batch_invariant_ops import (
            is_batch_invariant_mode_enabled,
            enable_batch_invariant_mode,
            disable_batch_invariant_mode,
        )
        
        # Initially disabled
        assert not is_batch_invariant_mode_enabled()
        
        # Enable
        enable_batch_invariant_mode()
        assert is_batch_invariant_mode_enabled()
        
        # Disable
        disable_batch_invariant_mode()
        assert not is_batch_invariant_mode_enabled()

    def test_batch_invariant_mode_context_manager(self):
        """Test context manager for batch invariant mode."""
        from batch_invariant_ops import is_batch_invariant_mode_enabled
        
        # Initially disabled
        assert not is_batch_invariant_mode_enabled()
        
        # Enable via context manager
        with set_batch_invariant_mode(True):
            assert is_batch_invariant_mode_enabled()
        
        # Should be disabled after context
        assert not is_batch_invariant_mode_enabled()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
