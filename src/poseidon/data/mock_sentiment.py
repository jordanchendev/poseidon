"""Mock sentiment data generator for testing.

Generates pseudo-sentiment scores that correlate with price movements.
Positive when price rises, negative when drops, with random noise.
This allows downstream phases to test strategies that include sentiment features.
"""

import logging
import random
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy.orm import Session

from poseidon.data.repository import DataRepository

logger = logging.getLogger(__name__)

# Sentiment source types for mock data
MOCK_SOURCE_TYPES = ["news", "earnings", "podcast", "tweet"]

# Noise parameters
NOISE_STD = 0.2  # Standard deviation of random noise
CORRELATION_WEIGHT = 0.7  # How much price change influences sentiment


def generate_mock_sentiment(
    session: Session,
    symbol: str,
    market: str,
    interval: str = "1d",
    days: int = 30,
) -> int:
    """Generate mock sentiment scores correlated with price movements.

    For each trading day in the range:
    1. Calculate daily return from OHLCV close prices
    2. Map return to a base sentiment (-1.0 to 1.0)
    3. Add random noise
    4. Clamp to [-1.0, 1.0]
    5. Persist via write_sentiment()

    Args:
        session: SQLAlchemy session
        symbol: Symbol ID
        market: Market name
        interval: OHLCV interval to compute returns from
        days: Number of recent days to generate sentiment for

    Returns:
        Number of sentiment rows created.
    """
    repo = DataRepository(session)
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=days + 5)  # Extra buffer for return calculation

    df = repo.read_ohlcv(symbol, market, interval, start, end)
    if df.empty or len(df) < 2:
        logger.warning("Not enough OHLCV data for %s/%s to generate mock sentiment", market, symbol)
        return 0

    # Calculate daily returns
    df = df.sort_values("time")
    df["return"] = df["close"].pct_change()
    df = df.dropna(subset=["return"])

    # Take the most recent N days
    df = df.tail(days)

    count = 0
    for _, row in df.iterrows():
        # Map return to base sentiment: positive return -> positive sentiment
        # Typical daily return is -5% to +5%, map to -1.0 to 1.0
        base_sentiment = max(-1.0, min(1.0, row["return"] * 20))  # Scale factor

        # Apply correlation weight and add noise
        noise = random.gauss(0, NOISE_STD)
        sentiment_score = CORRELATION_WEIGHT * base_sentiment + (1 - CORRELATION_WEIGHT) * noise

        # Clamp to valid range
        sentiment_score = max(-1.0, min(1.0, sentiment_score))

        # Choose a random source type
        source_type = random.choice(MOCK_SOURCE_TYPES)

        repo.write_sentiment(symbol, market, source_type, round(sentiment_score, 4))
        count += 1

    session.commit()
    logger.info("Generated %d mock sentiment scores for %s/%s", count, market, symbol)
    return count
