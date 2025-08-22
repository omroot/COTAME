
import pandas as pd

class COTPanel():

    def __init__(self) -> None:
        self.panel = None
    def fit(self, dataset: pd.DataFrame) -> None:
        dataset['tradeDate'] = pd.to_datetime(dataset['tradeDate'])
        dataset.sort_values(by = 'tradeDate', ascending = True, inplace = True)
        for feature_name in ['Commercial_NetPosition', 
                            'CommercialLongPosition', 
                            'CommercialShortPosition',
                            'ManagedMoney_NetPosition',
                            'ManagedMoney_LongPosition', 
                            'ManagedMoney_ShortPosition']:
            dataset[f'{feature_name}_change'] = dataset[feature_name]- dataset[feature_name].shift(1)
            dataset[f'prior_report_{feature_name}_change'] = dataset[f'{feature_name}_change'].shift(1)
            dataset[f'forward_report_{feature_name}_change'] = dataset[f'{feature_name}_change'].shift(-1)
            
            
        self.panel = dataset

