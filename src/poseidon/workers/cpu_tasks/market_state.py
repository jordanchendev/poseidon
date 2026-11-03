"""Position tracker, latest-price, broker, and perp-adapter helpers shared by the CPU tasks."""

from poseidon.core.database import SessionLocal, db_session


def _build_tw_stock_broker(broker_cfg):
    """Build the existing TW stock execution adapter from broker config."""
    from poseidon.broker.paper_adapter import PaperBrokerAdapter

    if broker_cfg.mode != "live" or broker_cfg.execution_backend == "paper":
        broker = PaperBrokerAdapter(SessionLocal)
        broker.login()
        return broker

    if broker_cfg.execution_backend != "shioaji":
        raise ValueError(f"Unsupported TW stock execution backend: {broker_cfg.execution_backend}")

    from poseidon.broker.shioaji_adapter import ShioajiBrokerAdapter

    broker = ShioajiBrokerAdapter(
        api_key=broker_cfg.shioaji_api_key,
        secret_key=broker_cfg.shioaji_secret_key,
        simulation=broker_cfg.shioaji_simulation,
        ca_cert_path=broker_cfg.shioaji_ca_cert_path,
        ca_password=broker_cfg.shioaji_ca_password,
        person_id=broker_cfg.shioaji_person_id,
    )
    broker.login()
    return broker


def _build_position_tracker():
    """Build and initialize a PositionTracker."""
    from poseidon.strategies.portfolio.position_tracker import PositionTracker

    tracker = PositionTracker(SessionLocal)
    tracker.rebuild_from_db()
    return tracker


def _get_latest_prices(symbols: list[str]) -> dict[str, float]:
    """Get latest close prices from Thalassa for given symbols.

    Returns:
        Dict mapping symbol -> latest close price.
    """
    from poseidon.data.remote_repository import RemoteDataRepository

    repo = RemoteDataRepository.from_settings()
    prices: dict[str, float] = {}
    for sym in symbols:
        latest = repo.read_ohlcv(sym, "tw_stock", "1d")
        if not latest.empty:
            prices[sym] = float(latest["close"].iloc[-1])
    return prices


def _build_perp_adapter_from_db():
    """Build PerpPaperAdapter and populate positions from DB holdings.

    Each Celery task needs its own adapter instance because in-memory
    _positions dict is NOT shared across tasks (pitfall #1 from research).
    """
    from poseidon.broker.perp_paper_adapter import PerpPaperAdapter, PerpPosition, calc_liquidation_price
    from poseidon.models.portfolio_holding import PortfolioHoldingRecord

    adapter = PerpPaperAdapter(SessionLocal)
    with db_session() as session:
        perp_holdings = (
            session.query(PortfolioHoldingRecord)
            .filter(
                PortfolioHoldingRecord.closed == False,  # noqa: E712
                PortfolioHoldingRecord.market == "crypto_perp",
            )
            .all()
        )
        for h in perp_holdings:
            leverage = adapter._leverage_per_symbol.get(h.symbol, adapter._default_leverage)
            entry_price = h.entry_price or 0.0
            quantity = h.shares or 0.0
            side = h.side or "long"
            liq_price = calc_liquidation_price(entry_price, leverage, side)
            adapter._positions[h.symbol] = PerpPosition(
                symbol=h.symbol,
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                leverage=leverage,
                margin=entry_price * quantity / leverage,
                liquidation_price=liq_price,
            )

    return adapter


def _get_perp_mark_prices(symbols: list[str]) -> dict[str, float]:
    """Get latest perpetual mark prices from Thalassa."""
    from poseidon.data.remote_repository import RemoteDataRepository

    repo = RemoteDataRepository.from_settings()
    prices = {}
    for sym in symbols:
        latest = repo.read_latest_price(sym)
        if latest is not None:
            prices[sym] = float(latest)
    return prices
