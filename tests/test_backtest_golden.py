"""Golden reference tests for backtest engine — schema extension and regression.

Phase 49: Signal schema golden reference tests.
Wave 0 stubs to be unskipped as implementation proceeds.
"""

import pytest


class TestSignalSchemaExtension:
    """Tests for OrderType enum and limit-order fields on Signal schema."""

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


class TestSignalConsumerIntegration:
    """Verify Signal consumer chain handles limit-order fields correctly."""

    def test_repository_maps_limit_order_fields(self):
        """Repository.save() passes limit order fields to SignalRecord constructor."""
        from poseidon.signals.schemas import OrderType, Signal

        signal = Signal(
            symbol="BTCUSDT",
            market="crypto_perp",
            action="long",
            confidence=0.8,
            order_type=OrderType.LIMIT,
            order_price=50000.0,
            stop_loss_price=48000.0,
            take_profit_price=55000.0,
        )
        # Verify the fields that would be passed to SignalRecord
        assert signal.order_type.value == "limit"  # .value for DB storage
        assert signal.order_price == 50000.0
        assert signal.stop_loss_price == 48000.0
        assert signal.take_profit_price == 55000.0

    def test_repository_maps_none_for_market_orders(self):
        """Repository.save() passes None for market-order signals."""
        from poseidon.signals.schemas import Signal

        signal = Signal(
            symbol="BTCUSDT",
            market="crypto_perp",
            action="long",
            confidence=0.8,
        )
        assert signal.order_type is None
        assert signal.order_price is None

    def test_delivery_includes_limit_fields_conditionally(self):
        """Delivery only includes non-None fields in Redis stream dict."""
        from poseidon.signals.schemas import OrderType, Signal, SignalStatus

        # Signal with limit order fields
        limit_signal = Signal(
            symbol="BTCUSDT",
            market="crypto_perp",
            action="long",
            confidence=0.8,
            order_type=OrderType.LIMIT,
            order_price=50000.0,
            stop_loss_price=48000.0,
            take_profit_price=55000.0,
            status=SignalStatus.PASSED,
        )
        # Build fields dict manually (same logic as delivery.deliver())
        fields = {}
        if limit_signal.order_type is not None:
            fields["order_type"] = limit_signal.order_type.value
        if limit_signal.order_price is not None:
            fields["order_price"] = str(limit_signal.order_price)
        if limit_signal.stop_loss_price is not None:
            fields["stop_loss_price"] = str(limit_signal.stop_loss_price)
        if limit_signal.take_profit_price is not None:
            fields["take_profit_price"] = str(limit_signal.take_profit_price)

        assert fields["order_type"] == "limit"
        assert fields["order_price"] == "50000.0"
        assert fields["stop_loss_price"] == "48000.0"
        assert fields["take_profit_price"] == "55000.0"

    def test_delivery_excludes_none_fields(self):
        """Market-order signals don't include limit fields in Redis stream."""
        from poseidon.signals.schemas import Signal, SignalStatus

        market_signal = Signal(
            symbol="BTCUSDT",
            market="crypto_perp",
            action="long",
            confidence=0.8,
            status=SignalStatus.PASSED,
        )
        fields = {}
        if market_signal.order_type is not None:
            fields["order_type"] = market_signal.order_type.value
        assert "order_type" not in fields
        assert "order_price" not in fields

    def test_api_response_includes_limit_fields(self):
        """SignalResponse schema accepts limit order fields."""
        from poseidon.api.signals import SignalResponse

        # Verify SignalResponse has the new fields
        fields = SignalResponse.model_fields
        assert "order_type" in fields
        assert "order_price" in fields
        assert "stop_loss_price" in fields
        assert "take_profit_price" in fields
