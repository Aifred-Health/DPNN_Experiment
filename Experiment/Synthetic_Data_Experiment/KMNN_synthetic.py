
from Experiment.test_utils import get_one_hot_encoded_features_data
from vulcanai.models.dnn import DenseNet
from sklearn import preprocessing
from vulcanai.datasets.utils import *
MAX_TOP_K=3



def synthetic_k_means_df(df_all, target_name, real_prob_names, NUMBER_OF_PROTOTYPES=5,   differential_feature= "intervention", ):
    from pandas import DataFrame
    from sklearn.cluster import KMeans

    all_features_names= list(df_all.drop([target_name], axis=1).columns.values)
    # concat the data frames for the get_dummies function (to  make sure both data frames have the same dimensions)

    #  differential_feature_data   is list of one-hot encoded features, generated for the differential feature,
    #  each item is clusters_size tuple of (feature_index, value, name).
    differential_feature_data = get_one_hot_encoded_features_data(all_features_names, differential_feature,
                                                                  prefix_sep='_')
    differential_feature_names = [i[2] for i in differential_feature_data]

    all_prototype_feature_names = [name for name in all_features_names if
                                   name not in differential_feature_names ]
    kmeans = KMeans(n_clusters=NUMBER_OF_PROTOTYPES).fit(df_all[all_prototype_feature_names])
    centroids = kmeans.cluster_centers_
    df_columns = ["dist centroid" + str(i) for i in range(len(centroids))] + differential_feature_names +real_prob_names+ [target_name]
    new_differential_feature_indexes = list(range(len(centroids), len(centroids) + len(differential_feature_names)))
    for i, d in enumerate(differential_feature_data): #todo clumsy
        differential_feature_data[i]= [new_differential_feature_indexes[i],d[1],d[2]]
    df_list = []
    all_distances = []
    for index, row in df_all.iterrows():  # for each sample in the test set
        distances = np.linalg.norm(np.array(np.array(row[all_prototype_feature_names]) - np.array(centroids)), axis=1)
        all_distances.append(distances)
        differential_value = row[differential_feature_names]
        target = row[target_name]
        df_list += [np.concatenate([distances, differential_value.values,row[real_prob_names].values,  [target]])]
    new_df = DataFrame(df_list, columns=df_columns)



    dense_net_config = {
        'dense_units': [len(df_columns) - len(differential_feature_names)- len(real_prob_names)],
        'initializer': None,
        'bias_init': None,
        'norm': None,
        'dropout': [0.5],
    }

    remsn_model = DenseNet(
        name='trained_7_drug_combined_model',
        activation=torch.nn.ReLU(),
        input_networks=None,
        in_dim=(len(df_columns) - 1),
        config=dense_net_config,
        optim_spec={'name': 'Adam', 'lr': 0.001},
        num_classes=2,
        device='cpu'
    )
    return remsn_model, new_df, all_distances, differential_feature_data
