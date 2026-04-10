"""Tests for training losses."""

import pytest
import torch

from infer.lib.train.losses import (
    MultiResolutionSTFTLoss,
    feature_loss,
    generator_loss,
    discriminator_loss,
)


class TestMultiResolutionSTFTLoss:
    @pytest.fixture
    def mrstft(self):
        return MultiResolutionSTFTLoss()

    def test_identical_signals_near_zero(self, mrstft):
        """同一信号の損失はゼロ近傍"""
        x = torch.randn(1, 1, 16000)
        loss = mrstft(x, x)
        assert loss.item() < 0.01

    def test_different_signals_positive(self, mrstft):
        """異なる信号の損失は正"""
        x = torch.randn(1, 1, 16000)
        y = torch.randn(1, 1, 16000)
        loss = mrstft(x, y)
        assert loss.item() > 0

    def test_gradient_flows(self, mrstft):
        """勾配が逆伝播可能であることを確認"""
        x = torch.randn(1, 1, 16000, requires_grad=True)
        y = torch.randn(1, 1, 16000)
        loss = mrstft(x, y)
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_output_is_scalar(self, mrstft):
        """出力はスカラーテンソル"""
        x = torch.randn(1, 1, 16000)
        y = torch.randn(1, 1, 16000)
        loss = mrstft(x, y)
        assert loss.dim() == 0

    def test_batch_dimension(self, mrstft):
        """バッチサイズ > 1 で動作"""
        x = torch.randn(4, 1, 16000)
        y = torch.randn(4, 1, 16000)
        loss = mrstft(x, y)
        assert loss.dim() == 0
        assert loss.item() > 0


class TestFeatureLoss:
    def test_identical_returns_zero(self):
        fmap = [[torch.randn(1, 32, 100)]]
        loss = feature_loss(fmap, fmap)
        # feature_loss detaches real, so identical inputs should give 0
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_different_returns_positive(self):
        fmap_r = [[torch.randn(1, 32, 100)]]
        fmap_g = [[torch.randn(1, 32, 100)]]
        loss = feature_loss(fmap_r, fmap_g)
        assert loss.item() > 0


class TestDiscriminatorLoss:
    def test_returns_tuple(self):
        real = [torch.ones(1, 1, 100)]
        fake = [torch.zeros(1, 1, 100)]
        loss, r_losses, g_losses = discriminator_loss(real, fake)
        assert isinstance(loss, torch.Tensor)
        assert len(r_losses) == 1
        assert len(g_losses) == 1

    def test_perfect_discrimination(self):
        real = [torch.ones(1, 1, 100)]
        fake = [torch.zeros(1, 1, 100)]
        loss, _, _ = discriminator_loss(real, fake)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)


class TestGeneratorLoss:
    def test_returns_tuple(self):
        outputs = [torch.ones(1, 1, 100)]
        loss, gen_losses = generator_loss(outputs)
        assert isinstance(loss, torch.Tensor)
        assert len(gen_losses) == 1

    def test_perfect_generation(self):
        outputs = [torch.ones(1, 1, 100)]
        loss, _ = generator_loss(outputs)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)
