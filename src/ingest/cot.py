import pandas as pd 
from  pathlib import Path



def ingest_cot_data(pre_raw_data_directory: Path) -> pd.DataFrame:



    wti_price_cot = pd.read_csv(pre_raw_data_directory / 'wti_price_cot.csv')
    rbob_price_cot = pd.read_csv(pre_raw_data_directory / 'rbob_price_cot.csv')
    ho_price_cot = pd.read_csv(pre_raw_data_directory / 'ho_price_cot.csv')
    gasoil_price_cot = pd.read_csv(pre_raw_data_directory / 'gasoil_price_cot.csv')
    br_price_cot = pd.read_csv(pre_raw_data_directory / 'br_price_cot.csv')
    wti_price_cot['Name'] = 'CL'
    rbob_price_cot['Name'] = 'XB'
    ho_price_cot['Name'] = 'HO'
    gasoil_price_cot['Name'] = 'QS'
    br_price_cot['Name'] = 'CO'


    for df in [wti_price_cot  ,
                rbob_price_cot,  
                ho_price_cot , 
                gasoil_price_cot ,
                br_price_cot  ]:
        df.columns = [ 'tradeDate', 
                            'F1_Price',
                            'F2_Price', 
                            'F3_Price', 
                            'F1_RolledPrice', 
                            'F2_RolledPrice',
                                'F3_RolledPrice', 
                                'Commercial_NetPosition', 
                                'CommercialLongPosition', 
                                'CommercialShortPosition', 
                                'ManagedMoney_NetPosition',
                                'ManagedMoney_LongPosition', 
                                'ManagedMoney_ShortPosition',
                    'Name']
    dataset = pd.concat([wti_price_cot  ,
            rbob_price_cot,  
            ho_price_cot , 
            gasoil_price_cot ,
            br_price_cot  ]).reset_index(drop=True)
    cot_db = dataset[['tradeDate',
                        'Name',
                        'Commercial_NetPosition', 
                                'CommercialLongPosition', 
                                'CommercialShortPosition', 
                                'ManagedMoney_NetPosition',
                                'ManagedMoney_LongPosition', 
                                'ManagedMoney_ShortPosition']]
        
    return cot_db