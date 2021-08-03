from Experiment.test_utils import get_one_hot_encoded_features_data
from vulcanai.models.dnn import DenseNet

import torch
import numpy as np
from pandas import DataFrame
from sklearn.cluster import KMeans
MAX_TOP_K=3

"""
The function processes a dataset for the kmnn-method evaluation:
It applies k-means and replaces the original features with the ditances from the centroids 
"""
def get_kmeans_configuration(df_all, target_name, number_of_prototypes, differential_feature, real_prob_names=[]):
    all_features_names= list(df_all.drop([target_name], axis=1).columns.values)

    #  differential_feature_data   is list of one-hot encoded features, generated for the differential feature,
    #  each item is clusters_size tuple of (feature_index, value, name).
    differential_feature_data = get_one_hot_encoded_features_data(all_features_names, differential_feature,
                                                                  prefix_sep='_')
    differential_feature_names = [i[2] for i in differential_feature_data]

    all_prototype_feature_names = [name for name in all_features_names if
                                   name not in differential_feature_names ]
    # apply k-mean algorithm to the original dataframe and obtain the centroids
    kmeans = KMeans(n_clusters=number_of_prototypes).fit(df_all[all_prototype_feature_names])
    centroids = kmeans.cluster_centers_
    df_columns = ["dist centroid" + str(i) for i in range(len(centroids))] + differential_feature_names +real_prob_names+ [target_name]
    # update the differential_feature_data, since the number of features is modified in the new dataset
    new_differential_feature_indexes = list(range(len(centroids), len(centroids) + len(differential_feature_names)))
    for i, d in enumerate(differential_feature_data): #todo clumsy
        differential_feature_data[i]= [new_differential_feature_indexes[i],d[1],d[2]]
    df_list = []
    # create a new dataframe that includes the Euclidean distance from the centroids instead of the original features
    for index, row in df_all.iterrows():  # for each sample in the test set
        distances = np.linalg.norm(np.array(np.array(row[all_prototype_feature_names]) - np.array(centroids)), axis=1)
        differential_value = row[differential_feature_names]
        target = row[target_name]
        df_list += [np.concatenate([distances, differential_value.values,row[real_prob_names].values,  [target]])]
    new_df = DataFrame(df_list, columns=df_columns)



    return new_df, differential_feature_data

