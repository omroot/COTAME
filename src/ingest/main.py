"""Orchestrates all data ingestion from pre-raw CSVs into unified raw databases."""

import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path

from src.ingest.price import ingest_price_data
from src.ingest.cot import ingest_cot_data, ingest_cot_data_from_cot_ingest
from src.ingest.volume import ingest_volume_data
from src.ingest.openinterest import ingest_openinterest_data
from src.settings import Settings

# cot-ingest repo paths
COT_INGEST_ROOT = Path('/Users/oualid/Documents/Projects/omroot_repos/cot-ingest/downloads')
CFTC_PATH = COT_INGEST_ROOT / 'cftc' / 'disaggregated_combined.csv'
ICE_PATH = COT_INGEST_ROOT / 'ice' / 'ice_cot.csv'


def ingest_all() -> None:
    """Run the full ingestion pipeline.

    Reads pre-raw CSV files from PRE_RAW_DATA_PATH, produces unified
    databases and writes them to RAW_DATA_PATH.

    Produces two COT files:
    - cot_legacy_db.csv: from local pre_raw (Commercial + ManagedMoney only)
    - cot_db.csv: from cot-ingest repo (all 5 disaggregated categories)
    """
    pre_raw_data_directory = Settings.historical.paths.PRE_RAW_DATA_PATH
    raw_data_directory = Settings.historical.paths.RAW_DATA_PATH

    price_data = ingest_price_data(pre_raw_data_directory)
    price_data.to_csv(raw_data_directory / 'prices_db.csv')

    cot_legacy_data = ingest_cot_data(pre_raw_data_directory)
    cot_legacy_data.to_csv(raw_data_directory / 'cot_legacy_db.csv')

    cot_data = ingest_cot_data_from_cot_ingest(cftc_path=CFTC_PATH, ice_path=ICE_PATH)
    cot_data.to_csv(raw_data_directory / 'cot_db.csv')

    volume_data = ingest_volume_data(pre_raw_data_directory)
    volume_data.to_csv(raw_data_directory / 'volume_db.csv')

    openinterest_data = ingest_openinterest_data(pre_raw_data_directory)
    openinterest_data.to_csv(raw_data_directory / 'openinterest_db.csv')


if __name__ == "__main__":
    ingest_all()
