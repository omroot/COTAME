from pathlib import Path
import src.config as cfg


ROOT_DIR = cfg.ROOT_DIR

class Settings:
    LOGS_DIR = ROOT_DIR / "logs"
    MODELS_DIR = ROOT_DIR / "cache" / "models"
    class historical:
        class paths:
            PRE_RAW_DATA_PATH = ROOT_DIR / 'cache' / 'pre_raw_data'
            RAW_DATA_PATH = ROOT_DIR / 'cache' /  'raw_data'
            PREPROCESSED_DATA_PATH = ROOT_DIR / 'cache' / 'preprocessed_data'

    class daily:
        class paths:
            PRE_RAW_DATA_PATH = ROOT_DIR / 'cache' / 'pre_raw_data'
            RAW_DATA_PATH = ROOT_DIR / 'cache' /   'raw_data'
            PREPROCESSED_DATA_PATH = ROOT_DIR / 'cache' / 'preprocessed_data'
    class loggers:
        DAILY = "daily"
        BACKFILL = "backfill"