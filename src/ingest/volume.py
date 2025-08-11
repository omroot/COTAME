
from  pathlib import Path
import pandas as pd

def ingest_volume_data(pre_raw_data_directory: Path) -> pd.DataFrame:

    # Read volume data CSV files
    wti_vol_wo_sprd = pd.read_csv(pre_raw_data_directory / 'wti_vol.csv')
    wti_sprd_vol = pd.read_csv(pre_raw_data_directory / 'wti_spd_vol.csv')
    rbob_vol = pd.read_csv(pre_raw_data_directory / 'rbob_vol.csv')
    ho_vol = pd.read_csv(pre_raw_data_directory / 'ho_vol.csv')
    gasoil_vol = pd.read_csv(pre_raw_data_directory / 'gasoil_vol.csv')
    br_vol = pd.read_csv(pre_raw_data_directory / 'br_vol.csv')


    wti_vol_wo_sprd.columns=['tradeDate', 'CL1 Comdty', 'CL2 Comdty', 'CL3 Comdty']
    wti_sprd_vol.columns = ['tradeDate', 'S:CLCL 1-2 Comdty']
    wti_vol = pd.merge(wti_vol_wo_sprd,
                    wti_sprd_vol,
                    on = 'tradeDate',
                    how = 'left')


    # Add a 'Name' column to each DataFrame

    wti_vol['Name'] = 'CL'
    rbob_vol['Name'] = 'XB'
    ho_vol['Name'] = 'HO'
    gasoil_vol['Name'] = 'QS'
    br_vol['Name'] = 'CO'

    for df in [wti_vol  ,
                rbob_vol  ,
                ho_vol  ,
                gasoil_vol  ,
                br_vol ]:
        df.columns=['tradeDate',
                    'F1_Volume',
                    'F2_Volume',
                    'F3_Volume',
                    'F1MinusF2_Volume',
                'Name']
    volume_db = pd.concat([wti_vol  ,
                rbob_vol  ,
                ho_vol  ,
                gasoil_vol  ,
                br_vol ]).reset_index(drop = True)
    


    return     volume_db[['tradeDate',
             'Name',
                 'F1_Volume',
                 'F2_Volume',
                 'F3_Volume',
                 'F1MinusF2_Volume',
              ]] 