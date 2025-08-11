
import warnings
# Disable all warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingest.price import ingest_price_data
from src.ingest.cot import ingest_cot_data
from src.ingest.volume import ingest_volume_data
from src.ingest.openinterest import ingest_openinterest_data    
from src.settings import Settings

def ingest_all()->None:


    settings = Settings()
    pre_raw_data_directory = settings.historical.paths.PRE_RAW_DATA_PATH
    raw_data_directory = settings.historical.paths.RAW_DATA_PATH
    price_data = ingest_price_data(pre_raw_data_directory)  
    price_data.to_csv(raw_data_directory / 'prices_db.csv')



    cot_data = ingest_cot_data(pre_raw_data_directory)
    cot_data.to_csv(raw_data_directory / 'cot_db.csv')

    volume_data = ingest_volume_data(pre_raw_data_directory)
    volume_data.to_csv(raw_data_directory / 'volume_db.csv')



    openinterest_data = ingest_openinterest_data(pre_raw_data_directory)
    openinterest_data.to_csv(raw_data_directory / 'openinterest_db.csv')



if __name__ == "__main__":
    ingest_all()


