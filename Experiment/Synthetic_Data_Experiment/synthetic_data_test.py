from sklearn.model_selection import StratifiedKFold
from Experiment.Synthetic_Data_Experiment.KMNN_synthetic import synthetic_k_means_df
from Experiment.test_utils import *
from Experiment.test_utils import _get_probs_new
import pandas as pd
import numpy as np
import torch
from vulcanai.models import DenseNet
from vulcanai.models.differentialpnn import DifferentialPrototypeNet

MAX_TOP_K=3
EPOCHS=20

target_name = 'target'
differential_feature = 'intervention'
pd_all = pd.read_csv("synthetic_data.csv")
NUMBER_OF_PROTOTYPES=5 # This is the number of prototypes that will be used in both kmnn and dpnn algorithm

all_data_loader= create_data_loader(pd_all, target_name)

k_fold=5


all_aucs={}
all_sens={}
all_spec={}
all_rrs= {}
all_rls={}


pd_all = pd_all.sample(frac=1).reset_index(drop=True)
kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=2)

# the metrics- the s
k_all = [[] for i in range(MAX_TOP_K)]


all_features_names = list(pd_all.drop([target_name], axis=1).columns.values)

#  differential_feature_data   is list of one-hot encoded features, generated for the differential feature,
#  each item is clusters_size tuple of (feature_index, value, name).
differential_feature_data = get_dummy_features_indexes(pd_all.drop([target_name], axis=1), differential_feature,
                                                       prefix_sep='_')
differential_feature_names = [i[2] for i in differential_feature_data]
real_probs_names= [name for name in all_features_names if name.startswith('real') ]
df_features_number = len(all_features_names) - len(differential_feature_names)- len(real_probs_names)


dpnn_net_config = {
    'encoder_units': [12],
    'classifier_units': [12],
    'num_prototypes': NUMBER_OF_PROTOTYPES,
    'dropout': [0.5],
    'excluded_feature_indexes': [],
    'differential_feature_data': differential_feature_data
}



models_loss_coefficients= {
    'lambda_class': 1,
    'lambda_ae': 0.01,
    'lambda_regularizer1': 0.001,
    'lambda_regularizer2': 0.001,
    'lambda_regularizer3': 0.005,
    'differential_loss_balance':0.85
}


dense_net_config = {
    'dense_units': [len(all_features_names)- len(real_probs_names)],
    #'dense_units': [(int(len(all_features_names) / 2) - len(real_probs_names))],
    'initializer': None,
    'bias_init': None,
    'norm': None,
    'dropout': [0.5],
}


b_size = 10

for run_config in [[all_data_loader, "nn"],[all_data_loader, "kmean"],[all_data_loader, "proto"]]: #,
    all_data_loader = run_config[0]
    new_net_name = run_config[1]
    if new_net_name not in all_aucs:
        all_aucs[new_net_name]=[]
        all_sens[new_net_name]=[]
        all_spec[new_net_name]=[]
        all_rrs[new_net_name]=[]
        all_rls[new_net_name]=[]

    print("name----"+ new_net_name)

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


        # split data to training and test
        if(new_net_name== "proto"):
            new_net = DifferentialPrototypeNet(
                name='prototype_model',
                in_dim=(len(all_features_names) - len(real_probs_names),),
                config=dpnn_net_config,
                num_classes=2,
                optim_spec={'name': 'Adam', 'lr': 0.0001},
                loss_coefficients=models_loss_coefficients
            )

        elif (new_net_name == "kmean"):
            new_net, kmean_df, all_distances, differential_feature_data = synthetic_k_means_df(
                pd_all, target_name, real_probs_names)
            differential_feature_indexes = [i[0] for i in differential_feature_data]
            train_df_original = kmean_df.iloc[result[0]]
            test_df_original = kmean_df.iloc[result[1]]
            test_df= test_df_original
            train_loader = create_data_loader(train_df_original, target_name, b_size)
            test_loader = create_data_loader(test_df_original, target_name, b_size)

        elif (new_net_name == "nn"):
            new_net = DenseNet(
                name='trained_7_drug_combined_model',
                activation=torch.nn.ReLU(),
                input_networks=None,
                in_dim=(len(all_features_names) - len(real_probs_names)),
                config=dense_net_config,
                optim_spec={'name': 'Adam', 'lr': 0.0001},
                num_classes=2,
                device='cpu'
            )

        new_net.fit(train_loader=train_loader, val_loader=test_loader, epochs=EPOCHS, \
                        valid_interv=4, plot=False, save_path='.')

        # create DataLoader for testing and test
        results_dict= new_net.run_test(data_loader=test_loader, plot=False, \
                               save_path='.', \
                               transform_outputs=False)
        all_aucs[new_net_name].append(results_dict["macro_auc"])
        all_spec[new_net_name].append(results_dict["macro_specificity"])
        all_sens[new_net_name].append(results_dict["macro_sensitivity"])
        # create a new test loader for single fast forward and obtain all proabibilities for all treatments
        test_loader= create_data_loader(test_df,target_name,1)
        probs = _get_probs_new(new_net,test_loader , differential_feature_data)


        distances_from_optimal=[]

        for index,key in enumerate(probs):
            real_probs= test_df_original[real_probs_names].values[index]
            distances_from_optimal.append( ranking_distance(np.argmax(list(probs[key].values())),real_probs))
        all_rls[new_net_name].append(np.average(np.array(distances_from_optimal)))

        print(np.average(np.array(distances_from_optimal)))
        distances_from_optimal2=[]
        for index,key in enumerate(probs):
            real_probs= test_df_original[real_probs_names].values[index]
            distances_from_optimal2.append( ranking_order_distance(list(probs[key].values()),real_probs))
        all_rrs[new_net_name].append(np.average(np.array(distances_from_optimal2)))
        print("MRR:")
        print(np.average(np.array(distances_from_optimal2)))
        #print(new_net_name)
        print("sensitivity")
        print(all_sens)
        print("macro_specificity")
        print(all_spec)
        max_values=[]
        for key in probs:
             # print the top k "best" treatments  for each patient
             max_values.append(max(probs[key].values()))
    print(all_aucs)
    print(all_rrs)
    print(all_rls)
    print("sensitivity")
    print(all_sens)
    print("macro_specificity")
    print(all_spec)