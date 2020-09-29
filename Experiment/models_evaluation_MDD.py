from Experiment.Kmean import get_kmeans_configuration
from Experiment.test_utils import _get_probs_new
from vulcanai.models.dnn import DenseNet
from vulcanai.models.differentialpnn import DifferentialPrototypeNet
from sklearn.model_selection import train_test_split
import  numpy as np
from Experiment.test_utils import  create_dense_net_model, get_dummy_features_indexes, create_data_loader, _get_probs_new



"""
This functions tests various models for a given data set.
The dataset should include:
 1) a differential feature= a feature that describes the treatment received by the patient
 2) A binary target feature= indicates the outcome after the treatment
 
 In this function, the model splits the data and evaluates the performance of three models:
 1) DPNN   2) NN (Vulcan's original dense net)  3) KMNN- a neural network based with kmean for feature processing. 
"""


def test_models(df_all, differential_feature, target_name, inner_layers, number_of_prototypes=6):
    # the metrics for the real workd evaluation
    all_aucs = {}
    all_rrs = {}

    all_features_names = list(df_all.drop([target_name], axis=1).columns.values)
    for g in range(13):
        df_all = df_all.sample(frac=1).reset_index(drop=True)

    #  differential_feature_data  is list of one-hot encoded features, generated for the differential feature,
    #  each item is clusters_size tuple of (feature_index, value, name).
    differential_feature_data_df_all = get_dummy_features_indexes(df_all.drop([target_name], axis=1),
                                                                  differential_feature,
                                                                  prefix_sep='_')
    models_loss_coefficients = {
        'lambda_class': 1,
        'lambda_ae': 0.01,
        'lambda_regularizer1': 0.01,
        'lambda_regularizer2': 0.01,
        'lambda_regularizer3': 0.06,
        'differential_loss_balance': 0.95
    }


    # iterate over all models, and evaluate each one separately
    for model_name in ["proto", "nn", "kmean"]:
        all_aucs[model_name] = []
        all_rrs[model_name] = []

        # get  parameters for kmnn net
        kmean_df, kmean_differential_feature_data = get_kmeans_configuration(
            df_all, target_name, number_of_prototypes, differential_feature)

        # set parameters for dpnn net
        dpnn_net_config = {
            'encoder_units': [19],
            'classifier_units': [11],
            'num_prototypes': number_of_prototypes,
            'dropout': [0.5, ],
            'excluded_feature_indexes': [],
            'differential_feature_data': differential_feature_data_df_all,
            'differential_feature_ratio': None
        }

        proto_net = DifferentialPrototypeNet(
            name='prototype_model',
            in_dim=(len(all_features_names),),
            config=dpnn_net_config,
            num_classes=2,
            optim_spec={'name': 'Adam', 'lr': 0.0001},
            loss_coefficients=models_loss_coefficients,
            # early_stopping="best_validation_error",
            # early_stopping_patience=2
        )
        input_dimension= len(all_features_names)
        nn_model = create_dense_net_model(input_dimension, inner_layers, "nn")


        train_df_original, test_df_original = train_test_split(df_all, test_size=0.2)

        train_loader = create_data_loader(train_df_original, target_name)
        test_loader = create_data_loader(test_df_original, target_name)
        differential_feature_data = differential_feature_data_df_all
        print("model name=  " + model_name)

        # set the network and all the parameters according to model
        if (model_name == "proto"):
            new_net = proto_net
        elif (model_name == "nn"):
            new_net = nn_model
        elif (model_name == "kmean"):
            differential_feature_data = kmean_differential_feature_data
            features_num= len(kmean_df.columns.values)-1
            new_net = create_dense_net_model(features_num,[features_num], "kmnn-net")
            train_df_original, test_df_original = train_test_split(kmean_df, test_size=0.2)
            train_loader = create_data_loader(train_df_original, target_name)
            test_loader = create_data_loader(test_df_original, target_name)
        new_net.fit(train_loader=train_loader, val_loader=test_loader, epochs=70, \
                    valid_interv=4)

        differential_feature_indexes = [i[0] for i in differential_feature_data]
        test_loader = create_data_loader(test_df_original, target_name, 1)

        results_dict = new_net.run_test(data_loader=test_loader, plot=False, \
                                        save_path='.', \
                                        transform_outputs=False)

        all_aucs[model_name].append(results_dict["macro_auc"])

        # test_loader = DataLoader(val_dataset, batch_size=1)
        probs = _get_probs_new(new_net, test_loader, differential_feature_data)

        # get the remission rate within the patient who recieved the model's "recommended" drug
        remission_recommended_drug = 0
        total_recommended_drug = 0
        for key in probs:
            best_index = np.argmax(list(probs[key].values()))
            # if the recommended  differential feature is the actual differential feature in the dataset
            if train_loader.dataset[key][0][differential_feature_indexes[best_index]].cpu().detach().numpy() == 1:
                total_recommended_drug += 1
                # adds one if the desired outcome was received (meaning that remission= true)
                remission_recommended_drug += train_loader.dataset[key][1].cpu().detach().numpy()
        # calculate and print the remission rate
        remission_rate_from_recommended = remission_recommended_drug / total_recommended_drug if total_recommended_drug > 0 else 0
        all_rrs[model_name].append(remission_rate_from_recommended)
    # print all metrics' values
    print(all_rrs)
    print(all_aucs)
