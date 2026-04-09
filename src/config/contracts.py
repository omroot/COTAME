"""
Loader for COT flat price contract mappings.

Usage:
    from src.config.contracts import get_contracts, get_cftc_codes, get_ice_names

    # All contracts for WTI
    contracts = get_contracts('CL')

    # Just the CFTC codes (for filtering disaggregated_combined.csv)
    codes = get_cftc_codes('CL')
    # -> ['067651', '067411', '06765I', ...]

    # ICE market names (for filtering ice_cot.csv)
    combined, futures = get_ice_names('CO')
    # -> ('ICE Brent Crude Futures and Options - ICE Futures Europe',
    #     'ICE Brent Crude Futures - ICE Futures Europe')
"""

import json
from pathlib import Path
from typing import Optional


_CONFIG_PATH = Path(__file__).parent / 'cot_flat_price_contracts.json'
_cache = None


def _load() -> dict:
    global _cache
    if _cache is None:
        with open(_CONFIG_PATH) as f:
            _cache = json.load(f)
    return _cache


def get_tickers() -> list[str]:
    """Return all available ticker symbols."""
    return list(_load().keys())


def get_contracts(ticker: str) -> dict:
    """Return the full config for a ticker (name + contracts list)."""
    data = _load()
    if ticker not in data:
        raise KeyError(f'Unknown ticker: {ticker}. Available: {list(data.keys())}')
    return data[ticker]


def get_cftc_codes(ticker: str) -> list[str]:
    """Return CFTC contract codes for a ticker (for filtering disaggregated report)."""
    contracts = get_contracts(ticker)['contracts']
    return [c['code'] for c in contracts if c['source'] == 'cftc' and c['code'] is not None]


def get_cftc_multipliers(ticker: str) -> dict[str, float]:
    """Return a mapping of CFTC code → position multiplier for a ticker.

    Contracts with no explicit multiplier default to 1.0.  Multipliers
    adjust for contract-size differences relative to the ticker's
    benchmark (e.g. E-Mini WTI is 500 bbl vs 1,000 bbl → 0.5).
    """
    contracts = get_contracts(ticker)['contracts']
    return {
        c['code']: c.get('multiplier', 1.0)
        for c in contracts
        if c['source'] == 'cftc' and c['code'] is not None
    }


def get_ice_names(ticker: str) -> tuple[Optional[str], Optional[str]]:
    """Return (combined_name, futures_name) for ICE contracts.

    Returns (None, None) if the ticker has no ICE component.
    """
    contracts = get_contracts(ticker)['contracts']
    for c in contracts:
        if c['source'] == 'ice':
            return (c.get('market_name_combined'), c.get('market_name_futures'))
    return (None, None)


def has_ice(ticker: str) -> bool:
    """Check if a ticker has an ICE Europe component."""
    return get_ice_names(ticker)[0] is not None
