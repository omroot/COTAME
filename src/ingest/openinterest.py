
from pathlib import Path
import pandas as pd


def ingest_openinterest_data(pre_raw_data_directory: Path) -> pd.DataFrame:

    wti_oi = pd.read_csv(pre_raw_data_directory / 'wti_oi.csv')
    rbob_oi = pd.read_csv(pre_raw_data_directory / 'rbob_oi.csv')
    ho_oi = pd.read_csv(pre_raw_data_directory / 'ho_oi.csv')
    gasoil_oi = pd.read_csv(pre_raw_data_directory / 'gasoil_oi.csv')
    br_oi = pd.read_csv(pre_raw_data_directory / 'br_oi.csv')

    wti_oi['Name'] = 'CL'
    rbob_oi['Name'] = 'XB'
    ho_oi['Name'] = 'HO'
    gasoil_oi['Name'] = 'QS'
    br_oi['Name'] = 'CO'

    for df in [wti_oi  ,
            rbob_oi  ,
            ho_oi  ,
            gasoil_oi  ,
            br_oi ]:
        df.columns=[
                "tradeDate", 
        "F1_OI", "F1_AGG", 
        "F2_OI", "F2_AGG", 
        "F3_OI", "AGG_OI",
                'Name']
        df = df.iloc[1:].reset_index(drop=True)

    oi_db = pd.concat([wti_oi  ,
            rbob_oi  ,
            ho_oi  ,
            gasoil_oi  ,
            br_oi ]).reset_index(drop = True)
    oi_db = oi_db.iloc[1:].reset_index(drop=True)
    return oi_db[['tradeDate',
             'Name',
               'F1_OI',   'F2_OI',   'F3_OI', 'AGG_OI'
              ]]

