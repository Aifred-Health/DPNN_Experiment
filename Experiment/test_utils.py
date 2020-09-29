import math
from collections import defaultdict

from sklearn import preprocessing
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from vulcanai.models import DenseNet
from vulcanai.models.basenetwork import set_tensor_device

# calculates the " remission regret"- the difference between the remission
# probability of the  model's recommended treatment and the actual best probability
def ranking_order_distance(predicted_probs, real_probs):
    array = np.array(predicted_probs)
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return 1 / (len(real_probs) - ranks[np.argmax(real_probs)])

# calculate the distance in the ranking order
def ranking_distance(top_ranked_index, real_probs):
    return np.max(real_probs) - real_probs[top_ranked_index]

def create_data_loader(df, target_name, b_size=10):
    input = set_tensor_device(torch.Tensor(np.array(df.drop([target_name], axis=1))), device="cuda:0")
    target = torch.LongTensor(np.array(df[target_name]))
    dataset = TensorDataset(input, target)
    data_loader = DataLoader(dataset, batch_size=b_size)
    return data_loader



def get_differential_values_counts(df, differential_features):
    # returns an array with the number of samples from each (one hot encoded) feature from a lost
    return [len(df[df[feature] == 1]) for feature in differential_features]


# the function gets a data frame, a list of one-hot encoded features and  a threshold and returns a filtered data
# frame without values that are rare
def check_division_by_differential_features(df, sub_df, differential_features, tolerance_ratio=2):
    all_df_counts = get_differential_values_counts(df, differential_features)
    counts = get_differential_values_counts(sub_df, differential_features)

    for i, c in enumerate(counts):
        if (c > (all_df_counts[i] * (len(sub_df) / len(df)) * tolerance_ratio) or c < (
                all_df_counts[i] * (len(sub_df) / len(df))) / tolerance_ratio):
            return False
    return True


def check_division_by_target(df, sub_df,target_name,  tolerance_ratio=2):
    target_all_count = len(df[df[target_name] == 1])
    target_sub_df_count = len(sub_df[sub_df[target_name] == 1])
    if (target_sub_df_count > ((target_all_count * (len(sub_df) / len(df))) * tolerance_ratio)):
        return False
    if (target_sub_df_count < ((target_all_count * (len(sub_df) / len(df)) / tolerance_ratio))):
        return False
    return True

def _get_probs_new(network, loader, indexesAndValues):
    """Returns probability for each object within loader based on output
    from training neural network

    Parameters:
        network : vulcan.model
            training vulcan network
        loader : torch.dataloader
            dataloader containing validation set
        indexesAndValues : list
            includes clusters_size list of one-hot encoded features, each item is clusters_size tuple of (feature_index, value, name)
    Returns:
        dct_scores : dictionary
            dictionary of scores
    """

    dct_scores = defaultdict()
    for index in range(len(loader)):
        dct_scores[index] = {}
    for index in range(len(loader)):
        # Extract specific index from loader and create  DataLoader instance
        # to send to forward_pass
        input_loader = DataLoader(TensorDataset(loader.dataset[index][0]
                                                .unsqueeze(0),
                                                loader.dataset[index][1]
                                                .unsqueeze(0)))

        subj_prob = network.forward_pass(data_loader=input_loader,
                                         transform_outputs=False)

        # Iterate through other possible values and find probability of
        # positive label and add to dictionary
        # for current index.
        for [featureIndex, _, _] in indexesAndValues:
            loader.dataset[index][0][featureIndex] = 0
        # for each possible differential features
        for [featureIndex, featurValue, _] in indexesAndValues:
            loader.dataset[index][0][featureIndex] = 1
            input_loader = DataLoader(TensorDataset(loader
                                                    .dataset[index][0]
                                                    .unsqueeze(0),
                                                    loader
                                                    .dataset[index][1]
                                                    .unsqueeze(0)))
            subj_prob = network.forward_pass(data_loader=input_loader,
                                             ransform_outputs=False)
            subj_prob = subj_prob[0][1] * 100
            subj_prob = round(subj_prob, 2)
            dct_scores[index][featurValue] = subj_prob
            # set treatment back to 0  (all treatments should be zero at this point)
            loader.dataset[index][0][featureIndex] = 0
    return dct_scores

def get_remission_rate(probs, differential_feature_indexes, train_loader, max_top_k=1):
    remission_rate_from_recommended={}
    remission_recommended_drug = 0
    total_recommended_drug = 0
    for k in range(1,max_top_k+1):
        for key in probs:
            topk_indexes= np.argpartition(list(probs[key].values()), -k)[-k:]
            # if the recommended  differential feature is the actual differential feature in the dataset
            for index in topk_indexes:
                if train_loader.dataset[key][0][differential_feature_indexes[index]].cpu().detach().numpy()==1:
                    total_recommended_drug += 1
                    remission_recommended_drug += train_loader.dataset[key][1].cpu().detach().numpy()
                    break
        # print the average of the max prediction
        remission_rate_from_recommended[k] =remission_recommended_drug / total_recommended_drug  if total_recommended_drug>0  else 0
    if(max_top_k==1):
        return remission_rate_from_recommended[1]
    return remission_rate_from_recommended

"""
Given a data frame with one hot encoded data and a categorical feature name, return the list of  corresponding binary features in the
data frame. each item is a tuple of (feature_index, feature value, feature )
the function assumes that the original_feature_name was a name of a column in the original data frame
"""
def get_dummy_features_indexes(df, original_feature_name,
                               prefix_sep='__'):
    return [(i, col.split(prefix_sep)[1], col) for i, col in enumerate(df.columns) if
            prefix_sep in col and col.split(prefix_sep)[0] == original_feature_name]



def get_one_hot_encoded_features_data(columns, original_feature_name,
                                      prefix_sep='_'):
    differential_feature_data = []
    for i, col in enumerate(columns):
        if prefix_sep in col and col.split(prefix_sep)[0] == original_feature_name:
            differential_feature_data.append((i, col.split(prefix_sep)[1], col))
    return differential_feature_data

def create_dense_net_model(input_dim,inner_layers, model_name):

    dense_net_config = {
        'dense_units': inner_layers,
        'initializer': None,
        'bias_init': None,
        'norm': None,
        'dropout': [0.5],
    }

    dense_net_model = DenseNet(
        name=model_name,
        activation=torch.nn.ReLU(),
        input_networks=None,
        in_dim=(input_dim),
        config=dense_net_config,
        optim_spec={'name': 'Adam', 'lr': 0.001},
        num_classes=2,
        device='cpu'
    )

    return dense_net_model