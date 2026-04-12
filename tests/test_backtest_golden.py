"""Golden reference tests for backtest engine — schema extension and regression.

Phase 49: Signal schema golden reference tests.
Wave 0 stubs to be unskipped as implementation proceeds.
"""

import pytest


class TestSignalSchemaExtension:
    """Tests for OrderType enum and limit-order fields on Signal schema."""

    @pytest.mark.skip(reason="Wave 0 stub -- implementation pending")
    def test_signal_backward_compat(self):
        """Signal() without new fields succeeds, all new fields are None."""
        from poseidon.signals.schemas import Signal

        sig = Signal(
            symbol="BTCUSDT",
            market="crypto_perp",
            action="long",
            confidence=0.8,
        )
        assert sig.order_type is None
        assert sig.order_price is None
        assert sig.stop_loss_price is None
        assert sig.take_profit_price is None

    @pytest.mark.skip(reason="Wave 0 stub -- implementation pending")
    def test_signal_with_limit_order(self):
        """Signal with order_type=LIMIT, order_price=50000.0 succeeds."""
        from poseidon.signals.schemas import OrderType, Signal

        sig = Signal(
            symbol="BTCUSDT",
            market="crypto_perp",
            action="long",
            confidence=0.8,
            order_type=OrderType.LIMIT,
            order_price=50000.0,
        )
        assert sig.order_type == OrderType.LIMIT
        assert sig.order_price == 50000.0

    @pytest.mark.skip(reason="Wave 0 stub -- implementation pending")
    def test_limit_requires_order_price(self):
        """Signal with order_type=LIMIT without order_price raises ValidationError."""
        from pydantic import ValidationError

        from poseidon.signals.schemas import OrderType, Signal

        with pytest.raises(ValidationError, match="order_price"):
            Signal(
                symbol="BTCUSDT",
                market="crypto_perp",
                action="long",
                confidence=0.8,
                order_type=OrderType.LIMIT,
            )

    @pytest.mark.skip(reason="Wave 0 stub -- implementation pending")
    def test_market_order_no_price_required(self):
        """Signal with order_type=MARKET without order_price succeeds."""
        from poseidon.signals.schemas import OrderType, Signal

        sig = Signal(
            symbol="BTCUSDT",
            market="crypto_perp",
            action="long",
            confidence=0.8,
            order_type=OrderType.MARKET,
        )
        assert sig.order_type == OrderType.MARKET
        assert sig.order_price is None


class TestBacktestGolden:
    """Golden regression tests for BacktestRunner (placeholder for future tasks)."""

    @pytest.mark.skip(reason="Wave 0 stub -- placeholder for golden regression tests")
    def test_placeholder(self):
        """Placeholder for golden regression test suite."""
        pass
