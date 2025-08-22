
import pandas as pd


class DataSetBuilder:

    def __init__(self) -> None:
        self.data = pd.DataFrame()

    def fit(self,
            cot_db: pd.DataFrame,
            synthetic_spread_db: pd.DataFrame,
            volume_db: pd.DataFrame,
            openinterest_db: pd.DataFrame,
            ) -> None:

        cot_db['tradeDate'] = pd.to_datetime(cot_db['tradeDate']).dt.date
        synthetic_spread_db['SyntheticF1MinusF2_RolledPrice'] = (synthetic_spread_db['F1_RolledPrice'] - 
                            synthetic_spread_db['beta_ols_10'] * synthetic_spread_db['F2_RolledPrice']
                                )
        synthetic_spread_db['tradeDate'] = pd.to_datetime(synthetic_spread_db['tradeDate']).dt.date
        dataset = pd.merge(cot_db,
                            synthetic_spread_db[['tradeDate', 
                                                'F1_RolledPrice',
                                                'F2_RolledPrice',
                                                'F3_RolledPrice',
                                                'F1_RolledPrice_rolling_20D_volatility',
                                                'F2_RolledPrice_rolling_20D_volatility',
                                                'F3_RolledPrice_rolling_20D_volatility',
                                                'SyntheticF1MinusF2_RolledPrice']],
                            on = 'tradeDate',
                            how = 'left') 
        dataset[f'prior_report_SyntheticF1MinusF2_RolledPrice_change'] = (dataset['SyntheticF1MinusF2_RolledPrice']-
                                                                  dataset['SyntheticF1MinusF2_RolledPrice'].shift(1) )
        volume_db['tradeDate'] = pd.to_datetime(volume_db['tradeDate']).dt.date
        dataset = pd.merge(dataset, 
                            volume_db[[  'tradeDate', 
                                            'prior_cumulative_5D_F1_Volume',
                                            'prior_cumulative_5D_F2_Volume' ]],
                            on = 'tradeDate',
                            how = 'left')
        dataset['prior_cumulative_5D_F1MinusF2_Volume'] = dataset['prior_cumulative_5D_F1_Volume']-dataset['prior_cumulative_5D_F2_Volume']
        openinterest_db['tradeDate'] = pd.to_datetime(openinterest_db['tradeDate']).dt.date
        dataset = pd.merge(dataset, 
                            openinterest_db[['tradeDate',
                                                    'F1_OI',
                                                    'F2_OI',
                                                    'F3_OI',
                                                    'AGG_OI',
                                                    'prior_5D_F1_OI_change',
                                                    'prior_5D_F2_OI_change',
                                                    'prior_5D_AGG_OI_change'
                                                ]],
                                            on = 'tradeDate',
                                            how = 'left')
        dataset['prior_5D_F1MinusF2_openinterest_change'] = dataset['prior_5D_F1_OI_change']-dataset['prior_5D_F2_OI_change']
        for f in  ['Commercial_NetPosition',
                    'CommercialLongPosition',
                    'CommercialShortPosition',
                    'ManagedMoney_NetPosition',
                    'ManagedMoney_LongPosition',
                    'ManagedMoney_ShortPosition']:
            dataset[f'{f}_to_openinterest'] = dataset[f]/dataset['AGG_OI'] 
        dataset.sort_values(by = 'tradeDate', ascending = True, inplace = True)
        for feature_name in ['Commercial_NetPosition_to_openinterest',
                            'CommercialLongPosition_to_openinterest',
                            'CommercialShortPosition_to_openinterest',
                            'ManagedMoney_NetPosition_to_openinterest',
                            'ManagedMoney_LongPosition_to_openinterest',
                            'ManagedMoney_ShortPosition_to_openinterest']:
            dataset[f'{feature_name}_change'] = dataset[feature_name]- dataset[feature_name].shift(1)
            dataset[f'prior_report_{feature_name}_change'] = dataset[f'{feature_name}_change'].shift(1)
            dataset[f'forward_{feature_name}_change'] =  dataset[feature_name].shift(-1) - dataset[feature_name]
        

        for name in [ 'prior_cumulative_5D_F1_Volume',
                    'prior_cumulative_5D_F2_Volume',
                    'prior_cumulative_5D_F1MinusF2_Volume' ,
                    'F1_RolledPrice',
                    'F2_RolledPrice',
                    'F3_RolledPrice']:
            dataset[f'{name}_change'] = dataset[name] - dataset[name].shift(1)
            dataset[f'next_{name}_change'] =  dataset[name].shift(-1) - dataset[name] 
        self.data = dataset
