import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.model_selection._split import BaseCrossValidator

from sklearn.cluster import KMeans
from sklearn.feature_selection import SequentialFeatureSelector

from research.feature_selection.mda import MDA
from research.feature_selection.mrmr import MRMR
from research.covariance.CleanseMatrix import CleanseMatrix

class MasterSelector():

    def __init__(self,
                 cv: BaseCrossValidator,
                 relevance_model : BaseEstimator,
                 redundancy_model: BaseEstimator,
                 feature_names: list[str] = None,
                 response_name: str = None,
                 feature_similarity_method: str = 'pearson',
                 is_classification: bool = True,
                 redundancy_method: str = 'mrmr',
                 correlation_threshold: float = 0.5,
                 verbose: bool = False
                 ):

        self.cv = cv
        self.relevance_model = relevance_model
        self.redudancy_model = redundancy_model
        self.feature_names = feature_names
        self.response_name = response_name
        self.feature_similarity_method = feature_similarity_method
        self.is_classification = is_classification
        self.redundancy_method = redundancy_method
        self.correlation_threshold = correlation_threshold
        self.verbose = verbose
        self.number_of_feature_clusters: int = None
        self.feature_clusters: dict = None
        self.selected_feature_cluster_names : list = None
        self.selected_feature_names: list = None
        self.clustered_mda: pd.DataFrame = None
        self.feature_sfi: dict = None  # {cluster_name: pd.DataFrame with mean/std per feature}


    def fit(self, dataset: pd.DataFrame):
        feature_similarity_matrix = dataset[self.feature_names].corr(method=self.feature_similarity_method)
        q = int(dataset.shape[0] / len(self.feature_names))
        feature_similarity_matrix_cleanser = CleanseMatrix(use_shrinkage=True,
                                                     shrinkage_regularizer=0.1,
                                                     detone=True,
                                                     market_components_max_index=1,
                                                     grid_size=1000,
                                                     kernel='gaussian',
                                                     cv=2,
                                                     min_bandwidth_grid_exponent=-3,
                                                     max_bandwidth_grid_exponent=1,
                                                     bandwidth_grid_size=250,
                                                     initial_variance=0.5,
                                                     epsilon=1e-5,
                                                     min_q=q,
                                                     max_q=q,
                                                     q_grid_size=1,
                                                     verbose=True)


        feature_similarity_matrix_cleanser.fit(np.array(feature_similarity_matrix.astype(float)))
        self.number_of_feature_clusters = feature_similarity_matrix_cleanser.estimated_number_signal_factors
        km = KMeans(n_clusters=self.number_of_feature_clusters, n_init=10)
        km.fit(feature_similarity_matrix)
        self.feature_clusters = {f'C_{i}': np.array(self.feature_names)[np.where(km.labels_ == i)[0]].tolist()
                            for i in np.unique(km.labels_)}
        if self.verbose:
            print(f'Feature cluster elements:')
            for key, value in self.feature_clusters.items():
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
                print()

        clustered_mda_estimator = MDA(model=self.relevance_model,
                                      cv=self.cv,
                                      clusters=self.feature_clusters,
                                      is_classification=self.is_classification )
        clustered_mda_estimator.fit(X=dataset[self.feature_names], y= dataset[self.response_name])

        clustered_mda_estimator.mda.index = [f'{i}' for i in list(clustered_mda_estimator.mda.index ) ]
        clustered_mda = clustered_mda_estimator.mda.reset_index()
        clustered_mda.columns = ['Cluster', 'Clustered MDA', 'CMDAStd']
        if self.verbose:
            print(clustered_mda)
            plt.figure(figsize=(20, 5))
            plt.bar(clustered_mda['Cluster'],
                    clustered_mda['Clustered MDA'],
                    align='center',
                    alpha=0.5,
                    ecolor='black',
                    capsize=10)

            plt.errorbar(clustered_mda['Cluster'],
                         clustered_mda['Clustered MDA'],
                         yerr=clustered_mda['CMDAStd'],
                         fmt='none',
                         ecolor='black',
                         capsize=10)
            plt.show()
            plt.close()
        
        self.clustered_mda = clustered_mda.set_index('Cluster')

        self.selected_feature_cluster_names =  clustered_mda[clustered_mda['Clustered MDA'] > np.maximum(clustered_mda['Clustered MDA'].median(),0)][
            'Cluster'].unique().tolist()
        if self.verbose:
            print(f'Selected cluster names: {self.selected_feature_cluster_names}')
    
        purged_feature_clusters = {}
        self.feature_sfi = {}
        for cluster_name in self.selected_feature_cluster_names:
            cluster_features = self.feature_clusters[cluster_name]
            if len(cluster_features) > 1:
                if self.redundancy_method == 'mrmr':
                    mrmr = MRMR(model=self.redudancy_model,
                                cv=self.cv,
                                correlation_threshold=self.correlation_threshold,
                                similarity_method=self.feature_similarity_method,
                                is_classification=self.is_classification)
                    mrmr.fit(X=dataset[cluster_features], y=dataset[self.response_name])
                    purged_feature_clusters[cluster_name] = mrmr.selected_features
                    self.feature_sfi[cluster_name] = mrmr.sfi.importances
                elif self.redundancy_method == 'sfs':
                    sfs = SequentialFeatureSelector(estimator=self.redudancy_model,
                                                    cv=self.cv,
                                                    direction='backward')
                    sfs.fit(X=dataset[cluster_features], y=dataset[self.response_name])
                    purged_feature_clusters[cluster_name] = list(np.array(cluster_features)[sfs.get_support()])
            else:
                purged_feature_clusters[cluster_name] = cluster_features
        self.purged_feature_clusters =purged_feature_clusters
        if self.verbose:
            print(f'Purged cluster elements:')
            for key, value in self.purged_feature_clusters.items():
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
                print()

        purged_features = [feature for features in self.purged_feature_clusters.values() for feature in features]





        final_features = []
        for cluster_name, cluster_features in self.purged_feature_clusters.items():
            if cluster_name in self.selected_feature_cluster_names :
                final_features = final_features + cluster_features
        self.selected_feature_names = final_features
        if self.verbose:
            print(f'Final Selected Features: {self.selected_feature_names}')