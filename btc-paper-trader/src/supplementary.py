"""Supplementary data fetchers: order book depth and derivatives state.

Order book: Binance US spot (same venue as the live OHLCV).
Open interest + funding: Kraken Futures (Binance Futures blocked in US), one
ticker call for both.

This data is NOT used by the current model; it is accumulated for future model
iterations, and it cannot be backfilled. The fetchers return payload dicts
only — no timestamps, no storage. Keying, scheduling, gap marking, and atomic
parquet writes belong to src.archiver (hardening spec WS7).
"""

import json
import logging
import zlib

import numpy as np
import requests

logger = logging.getLogger(__name__)

# Kraken Futures funding rate is annualized absolute.
# Convert to Binance-equivalent per-8h rate: rate / (365.25 * 3)
_KRAKEN_ANNUAL_TO_8H = 1.0 / (365.25 * 3)


def fetch_orderbook_snapshot(
    symbol: str = "BTCUSDT",
    base_url: str = "https://api.binance.us",
    depth_limit: int = 1000,
) -> dict | None:
    """Fetch order book depth snapshot from Binance US.

    Returns dict with summary metrics and compressed raw levels,
    or None on failure.
    """
    url = f"{base_url}/api/v3/depth"
    params = {"symbol": symbol, "limit": depth_limit}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Order book fetch failed: {e}")
        return None

    bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
    asks = [(float(p), float(q)) for p, q in data.get("asks", [])]

    if not bids or not asks:
        logger.warning("Empty order book")
        return None

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid_price = (best_bid + best_ask) / 2
    spread_bps = (best_ask - best_bid) / mid_price * 10000

    # Compute volume within percentage bands from mid
    def volume_within_pct(levels, mid, pct, side):
        total = 0.0
        for price, qty in levels:
            if side == "bid" and price >= mid * (1 - pct / 100):
                total += qty
            elif side == "ask" and price <= mid * (1 + pct / 100):
                total += qty
        return total

    bid_vol_05 = volume_within_pct(bids, mid_price, 0.5, "bid")
    bid_vol_1 = volume_within_pct(bids, mid_price, 1.0, "bid")
    bid_vol_2 = volume_within_pct(bids, mid_price, 2.0, "bid")
    ask_vol_05 = volume_within_pct(asks, mid_price, 0.5, "ask")
    ask_vol_1 = volume_within_pct(asks, mid_price, 1.0, "ask")
    ask_vol_2 = volume_within_pct(asks, mid_price, 2.0, "ask")

    def imbalance(bid_v, ask_v):
        total = bid_v + ask_v
        return (bid_v - ask_v) / total if total > 0 else 0.0

    # Compress raw top-100 levels as JSON blob
    raw_top100 = {
        "bids": bids[:100],
        "asks": asks[:100],
    }
    raw_json = json.dumps(raw_top100, separators=(",", ":"))
    raw_compressed = zlib.compress(raw_json.encode(), level=6)

    return {
        "mid_price": mid_price,
        "spread_bps": spread_bps,
        "bid_volume_0_5pct": bid_vol_05,
        "bid_volume_1pct": bid_vol_1,
        "bid_volume_2pct": bid_vol_2,
        "ask_volume_0_5pct": ask_vol_05,
        "ask_volume_1pct": ask_vol_1,
        "ask_volume_2pct": ask_vol_2,
        "imbalance_0_5pct": imbalance(bid_vol_05, ask_vol_05),
        "imbalance_1pct": imbalance(bid_vol_1, ask_vol_1),
        "raw_levels": raw_compressed,
    }


def fetch_derivatives_snapshot(
    kraken_futures_url: str = "https://futures.kraken.com",
    kraken_symbol: str = "PF_XBTUSD",
) -> dict | None:
    """Fetch open interest and funding from one Kraken Futures ticker call.

    Funding was previously only captured indirectly (forward-filled into the
    OHLCV parquet by the pipeline); archiving it hourly at source keeps the
    non-backfillable record whole (WS7). Returns dict or None on failure.
    """
    url = f"{kraken_futures_url}/derivatives/api/v3/tickers/{kraken_symbol}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        ticker = resp.json().get("ticker", {})
    except Exception as e:
        logger.warning(f"Derivatives fetch failed: {e}")
        return None

    if not ticker:
        logger.warning("Empty Kraken ticker response")
        return None

    def _finite(field):
        try:
            v = float(ticker[field])
        except (KeyError, TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    oi = _finite("openInterest")
    if oi is None:
        logger.warning(f"Kraken openInterest not usable: {ticker.get('openInterest')!r}")
        return None
    mark = _finite("markPrice")
    annual = _finite("fundingRate")

    return {
        "open_interest": oi,
        "open_interest_usd": oi * mark if mark else 0.0,
        "funding_rate_annual": annual,
        "funding_rate_8h": annual * _KRAKEN_ANNUAL_TO_8H if annual is not None else None,
    }
