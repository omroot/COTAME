"""
Consolidated configuration for the COTAME project.

Backwards-compatible: existing code using `import src.config as cfg`
or `from src.settings import Settings` continues to work.
"""

from src.config.paths import ROOT_DIR, OUTPUT_DIR, Settings, DEBUG
from src.config.tickers import FUTURES_TICKER_TO_NAME
