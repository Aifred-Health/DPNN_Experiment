from sklearn.model_selection import StratifiedKFold

from Experiment.Kmean import get_kmeans_configuration
from Experiment.test_utils import *
from Experiment.test_utils import _get_probs_new
import pandas as pd
import numpy as np
import torch
from vulcanai.models import DenseNet
from vulcanai.models.differentialpnn import DifferentialPrototypeNet

MAX_TOP_K=3

"""
# load the synthetic data
# the synthetic data includes: 1) a set of features describing the patient 2) an intervention ( the differential feature) 3) The outcome of the intervention 
and the probability of the outcome of all the counter-factuals.  see "generate_synthetic_data.py" documentation 
"""

target_name = 'target'
differential_feature = 'intervention'
pd_all = pd.read_csv("synthetic_data.csv")
NUMBER_OF_PROTOTYPES=5 # This is the number of prototypes that was used for generating the data (see "generate_synthetic_data.py")

all_data_loader= create_data_loader(pd_all, target_name)
k_fold=5

# initialize the metrics for synthetic data evaluation
all_aucs={}  # AUC
all_rrs= {}  # Reciprocal-rank
all_rls={}   # Remission loss

pd_all = pd_all.sample(frac=1).reset_index(drop=True)
# create a k-fold division of the data
kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=2)

all_features_names = list(pd_all.drop([target_name], axis=1).columns.values)

#  differential_feature_data   is list of one-hot encoded features, generated for the differential feature,
#  each item is clusters_size tuple of (feature_index, value, name).
differential_feature_data = get_dummy_features_indexes(pd_all.drop([target_name], axis=1), differential_feature,
                                                       prefix_sep='_')
differential_feature_names = [i[2] for i in differential_feature_data]
real_probs_names= [name for name in all_features_names if name.startswith('real') ]
# find the number of features describing the patient
df_features_number = len(all_features_names) - len(differential_feature_names)- len(real_probs_names)


# set configurations of the models
dpnn_net_config = {
    'encoder_units': [14],
    'classifier_units': [16],
    'num_prototypes': NUMBER_OF_PROTOTYPES,
    'dropout': [0.5],
    'excluded_feature_indexes': [],
    'differential_feature_data': differential_feature_data
}

models_loss_coefficients= {
    'lambda_class': 1,
    'lambda_ae': 0.01,
    'lambda_regularizer1': 0,
    'lambda_regularizer2': 0,
    'lambda_regularizer3': 0.05,
    'differential_loss_balance':0.85,
    'differential_feature_ratio': None
}



number_of_nodes= len(all_features_names) - len(real_probs_names)
nn_model=create_dense_net_model(number_of_nodes, [number_of_nodes], "nn")

proto_net= DifferentialPrototypeNet(
                name='prototype_model',
                in_dim=(len(all_features_names) - len(real_probs_names),),
                config=dpnn_net_config,
                num_classes=2,
                optim_spec={'name': 'Adam', 'lr': 0.0001},
                loss_coefficients=models_loss_coefficients
            )

b_size = 10

for model_name in ["nn", "kmean", "proto" ]:
    if model_name not in all_aucs:
        all_aucs[model_name]=[]
        all_rrs[model_name]=[]
        all_rls[model_name]=[]

    print("model name: " + model_name)
    # for each fold- use fold as test set and the rest as train set
    for fold in range(k_fold):
        result = next(kf.split(np.array(pd_all.drop([target_name], axis=1)),np.array(pd_all[target_name])))
        train_df_original = pd_all.iloc[result[0]]
        test_df_original = pd_all.iloc[result[1]]
        train_df= train_df_original.drop(real_probs_names, axis=1)
        test_df= test_df_original.drop( real_probs_names, axis=1)
        b_size = 10
        # create a dataloader without the features of the  real probabilities
        train_loader = create_data_loader(train_df, target_name)
        test_loader =create_data_loader(test_df, target_name)


        # set the model and the
        if(model_name== "proto"):
            new_net = proto_net

        elif (model_name == "kmean"):
            # get all parameters for the kmean evaluation
            kmean_df, differential_feature_data = get_kmeans_configuration(
                pd_all, target_name, NUMBER_OF_PROTOTYPES, differential_feature ,real_probs_names)

            number_of_nodes= len(kmean_df.columns.values) - 1
            new_net = create_dense_net_model(number_of_nodes, [number_of_nodes], "kmnn-net")
            differential_feature_indexes = [i[0] for i in differential_feature_data]
            train_df_original = kmean_df.iloc[result[0]]
            test_df_original = kmean_df.iloc[result[1]]
            test_df= test_df_original
            train_loader = create_data_loader(train_df_original, target_name, b_size)
            test_loader = create_data_loader(test_df_original, target_name, b_size)

        elif (model_name == "nn"):
            new_net = nn_model

        new_net.fit(train_loader=train_loader, val_loader=test_loader, epochs=90, \
                        valid_interv=4, plot=False, save_path='.')

        # create DataLoader for testing and test
        results_dict= new_net.run_test(data_loader=test_loader, plot=False, \
                               save_path='.', \
                               transform_outputs=False)
        all_aucs[model_name].append(results_dict["macro_auc"])

        # create a new test loader for single fast forward and obtain all proabibilities for all treatments
        test_loader= create_data_loader(test_df,target_name,1)
        probs = _get_probs_new(new_net,test_loader , differential_feature_data)


        remission_loss=[]

        for index,key in enumerate(probs):
            real_probs= test_df_original[real_probs_names].values[index]
            remission_loss.append( ranking_distance(np.argmax(list(probs[key].values())),real_probs))
        all_rls[model_name].append(np.average(np.array(remission_loss)))
        rank_distances=[]
        for index,key in enumerate(probs):
            real_probs= test_df_original[real_probs_names].values[index]
            rank_distances.append( ranking_order_distance(list(probs[key].values()),real_probs))
        all_rrs[model_name].append(np.average(np.array(rank_distances)))


    print(all_aucs)
    print(all_rrs)
    print(all_rls)
