import warnings
# Disable all warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

from src.preprocessing.base import FutureTicker
from src.preprocessing.prices import PricePanel
from src.preprocessing.volume import VolumePanel
from src.preprocessing.openinterest import OpenInterestPanel
from src.preprocessing.synthetic_spread import SyntheticSpreadBuilder, HedgeMethod
from src.preprocessing.cot import COTPanel
from src.preprocessing.dataset_builder import DataSetBuilder

from src.utils.io.read import RawDataReader
from src.utils.dates import get_nyse_business_dates
from src.settings import Settings


def preprocess_all(ticker: FutureTicker)->None:
    
    RAW_DATA_PATH = Settings.historical.paths.RAW_DATA_PATH
    PREPROCESSED_DATA_PATH = Settings.historical.paths.PREPROCESSED_DATA_PATH
    rdr = RawDataReader(raw_data_directory= RAW_DATA_PATH)
    all_prices_db = rdr.read_prices()


    prices_db = all_prices_db[all_prices_db['Name'] == ticker.value]
    prices_db['tradeDate'].dropna(inplace=True)
    prices_db.loc[:,'tradeDate'] = pd.to_datetime(prices_db['tradeDate']).dt.date

    prices_db.loc[:,'F1_RolledPrice'] = pd.to_numeric(prices_db['F1_RolledPrice'], errors='coerce')
    prices_db.loc[:,'F2_RolledPrice'] = pd.to_numeric(prices_db['F2_RolledPrice'], errors='coerce')
    prices_db.loc[:,'F3_RolledPrice'] = pd.to_numeric(prices_db['F3_RolledPrice'], errors='coerce')

    business_dates = get_nyse_business_dates(prices_db['tradeDate'].min(),
                                             prices_db['tradeDate'].max())
    prices_db = prices_db[prices_db['tradeDate'].isin(business_dates)]
    price_panel_builder = PricePanel()
    price_panel_builder.fit(dataset=prices_db)

    price_panel_builder.panel.to_csv(PREPROCESSED_DATA_PATH / f'{ticker.name}_prices_panel.csv', index=False)

    synthetic_spread_builder = SyntheticSpreadBuilder(method=HedgeMethod.OLS, windows=[10, 20])
    synthetic_spread_db = synthetic_spread_builder.compute(price_panel_builder.panel)
    synthetic_spread_db.to_csv(PREPROCESSED_DATA_PATH / f'{ticker.name}_synthetic_spread_db.csv', index=False)



    all_volume_db = rdr.read_volume()
    all_volume_db['F1_Volume'] = pd.to_numeric(all_volume_db['F1_Volume'], errors='coerce')
    all_volume_db['F2_Volume'] = pd.to_numeric(all_volume_db['F2_Volume'], errors='coerce')
    all_volume_db['F3_Volume'] = pd.to_numeric(all_volume_db['F3_Volume'], errors='coerce')
    all_volume_db['F1MinusF2_Volume'] = pd.to_numeric(all_volume_db['F1MinusF2_Volume'], errors='coerce')
    volume_db = all_volume_db[all_volume_db['Name'] == ticker.value]
    volume_db['tradeDate'] = pd.to_datetime(volume_db['tradeDate']).dt.date
    volume_db = volume_db[volume_db['tradeDate'].isin(business_dates)]

    volume_panel_builder = VolumePanel()
    volume_panel_builder.fit(dataset=volume_db)

    volume_panel_builder.panel.to_csv(PREPROCESSED_DATA_PATH / f'{ticker.name}_volume_panel.csv', index=False)


    all_openinterest_db = rdr.read_openinterest()
    all_openinterest_db['F1_OI'] = pd.to_numeric(all_openinterest_db['F1_OI'], errors='coerce')
    all_openinterest_db['F2_OI'] = pd.to_numeric(all_openinterest_db['F2_OI'], errors='coerce')
    all_openinterest_db['F3_OI'] = pd.to_numeric(all_openinterest_db['F3_OI'], errors='coerce')
    all_openinterest_db['AGG_OI'] = pd.to_numeric(all_openinterest_db['AGG_OI'], errors='coerce')
    all_openinterest_db['F1_OI_Minus_F2_OI'] =  all_openinterest_db['F1_OI'] - all_openinterest_db['F2_OI']

    openinterest_db = all_openinterest_db[all_openinterest_db['Name'] == ticker.value]
    openinterest_db['tradeDate'] = pd.to_datetime(openinterest_db['tradeDate']).dt.date
    openinterest_db = openinterest_db[openinterest_db['tradeDate'].isin(business_dates)]

    openinterest_panel_builder = OpenInterestPanel()
    openinterest_panel_builder.fit(dataset=openinterest_db)
    openinterest_panel_builder.panel.to_csv(PREPROCESSED_DATA_PATH / f'{ticker.name}_openinterest_panel.csv', index=False)


    all_cot_db = rdr.read_cot()
    all_cot_db.dropna(inplace=True)
    all_cot_db = all_cot_db[[
                    'tradeDate',
                    'Name',
                    'Commercial_NetPosition',
                    'CommercialLongPosition',
                    'CommercialShortPosition',
                    'ManagedMoney_NetPosition',
                    'ManagedMoney_LongPosition',
                    'ManagedMoney_ShortPosition'

    ]]
    cot_db = all_cot_db[all_cot_db['Name']== ticker.value]
    cot_panel_builder = COTPanel()
    cot_panel_builder.fit(dataset=cot_db)
    cot_panel_builder.panel.to_csv(PREPROCESSED_DATA_PATH / f'{ticker.name}_cot_panel.csv', index=False)



    dataset_builder = DataSetBuilder()
    dataset_builder.fit(cot_db=cot_panel_builder.panel,
                            synthetic_spread_db=synthetic_spread_db,
                            volume_db=volume_panel_builder.panel,
                            openinterest_db=openinterest_panel_builder.panel)

    dataset_builder.data.to_csv(PREPROCESSED_DATA_PATH / f'{ticker.name}_dataset.csv', index=False)







if __name__ == "__main__":
    preprocess_all(ticker=FutureTicker.WTI)
    preprocess_all(ticker=FutureTicker.BRENT)
    preprocess_all(ticker=FutureTicker.RBOB)
    preprocess_all(ticker=FutureTicker.HEATING_OIL)
