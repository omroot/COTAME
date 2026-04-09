"""Project paths and settings derived from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

try:
    load_dotenv()
except Exception as e:
    print(f"Error loading .env file : {e}")

DEBUG = str(os.getenv("DEBUG")).lower() in ['true']

DEBUG_ROOT_DIR = Path(os.getenv("DEBUG_ROOT_DIR"))
PROD_ROOT_DIR = Path(os.getenv("PROD_ROOT_DIR"))

if DEBUG:
    ROOT_DIR = DEBUG_ROOT_DIR
else:
    ROOT_DIR = PROD_ROOT_DIR

OUTPUT_DIR = ROOT_DIR / "cache" / "output"


class Settings:
    LOGS_DIR = ROOT_DIR / "logs"
    MODELS_DIR = ROOT_DIR / "cache" / "models"

    class historical:
        class paths:
            PRE_RAW_DATA_PATH = ROOT_DIR / 'cache' / 'pre_raw_data'
            RAW_DATA_PATH = ROOT_DIR / 'cache' / 'raw_data'
            PREPROCESSED_DATA_PATH = ROOT_DIR / 'cache' / 'preprocessed_data'

    class daily:
        class paths:
            PRE_RAW_DATA_PATH = ROOT_DIR / 'cache' / 'pre_raw_data'
            RAW_DATA_PATH = ROOT_DIR / 'cache' / 'raw_data'
            PREPROCESSED_DATA_PATH = ROOT_DIR / 'cache' / 'preprocessed_data'

    class loggers:
        DAILY = "daily"
        BACKFILL = "backfill"
