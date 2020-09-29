import numpy as np
from Experiment.LPAD import  LPA
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score

"""
This functions tests the the latent profile analysis approach for treatment selection
The function gets a dataset for the evaluation. 
The dataset should include:
 1) a differential feature= a feature that describes the treatment received by the patient
 2) A binary target feature= indicates the outcome after the treatment
 
 LPA, latent profile analysis- finds clusters of data in the latent space 
"""
def evaluate_lpa(pd_all, all_features,differential_feature_names,target_name):
    aucs={}
    rrs={}
    NUMBER_OF_CLUSTERS=3
    aucs[NUMBER_OF_CLUSTERS] = []
    rrs[NUMBER_OF_CLUSTERS] = []
    # The list of metrics results
    for k in range(10):
        lpa_model = LPA(n_components=NUMBER_OF_CLUSTERS, tol=10e-5, max_iter=60)
        fold_k=5

        pd_all = pd_all.sample(frac=1).reset_index(drop=True)
        kf = StratifiedKFold(n_splits=fold_k, shuffle=True, random_state=2)

        for i in range(fold_k):  # for each fold
            # divide the dataset into train and test set according to fold
            result = next(
                kf.split(np.array(pd_all.drop([target_name], axis=1)), np.array(pd_all[target_name])))
            pd_train = pd_all.iloc[result[0]]
            pd_test = pd_all.iloc[result[1]]

            all_data_columns= list(pd_train.columns)
            differential_feature_indexes= [ all_data_columns.index(diff) for diff in differential_feature_names]
            all_features_names = all_features[:-1] # except target
            # create DataLoader for training and train, the training does not consider the treatments assigned
            # we drop the treatment features so that the clustering will not be based on it
            train_input= np.array(pd_train.drop(['remsn']+differential_feature_names, axis=1))
            train_target = np.array(pd_train['remsn'])
            # fit and find clusters
            lpa_model.fit(train_input)

            # create DataLoader for training and train
            test_input= np.array(pd_test.drop(['remsn']+differential_feature_names, axis=1))
            test_target = np.array(pd_test['remsn'])

            #remission_dict is a dictionary with cluster numbers as keys. the values is the distribution of targets for each intervention
            # so each value is a dictionary, the interventions are keys and the value is the list of all  remission
            remission_dict= {}
            for i in range(NUMBER_OF_CLUSTERS):
                remission_dict[i]={}

            # get predictions of clusters
            predict = lpa_model.predict(train_input)

            #  fill remission_dict- for each sample, get the associated cluster and the intervention and add the target to
            # the appropiate list in the dictionary
            for i,d in enumerate(pd_train.values):
                cluster_prediction= predict[i]
                # get the number of the actual differential feature in the sample
                differential_feature_order= np.argmax([d[ind] for ind in differential_feature_indexes])
                if differential_feature_order not in remission_dict[cluster_prediction]:
                    remission_dict[cluster_prediction][differential_feature_order]=[]
                remission_dict[cluster_prediction][differential_feature_order].append(train_target[i])

            # find the best intervention for each cluster
            best_interventions=[]
            rem_rate_dict= {}  # contains the remission rates of for each cluster and intervention
            for c in range(NUMBER_OF_CLUSTERS): # for each cluster
                # save the remission rate for each treatment feature in cluster
                remission_rate = np.zeros(len(differential_feature_names))
                for i in range(len(differential_feature_names)):
                    remission_rate[i]= 0 if i not in  remission_dict[c] or  len(remission_dict[c][i])<5 else np.sum(remission_dict[c][i]) / len(remission_dict[c][i])
                # for each index i add the intervention with the highest remission rate
                best_interventions.append(remission_rate.argsort()[-1:][::-1] )
                rem_rate_dict[c]= remission_rate

            # predict on test data set
            predict_test = lpa_model.predict(test_input)

            # The remission cluster_prediction is the remission rate for the received treatment in the associated cluster
            remission_predictions= []
            total_as_recommended=0.0
            total_remission_in_recommended= 0.0

            # get the remission rate for the patients who received the recommended drug
            # and obtain the expected remission rate for each patient according t his assocaiated cluste
            for i,d in enumerate(pd_test.values):
                cluster_prediction= predict_test[i]
                # get the differential feature number
                differential_feature_order= np.argmax([d[ind] for ind in differential_feature_indexes])
                # if the recommended drug is the same as the actual received drug in the dataset
                if(differential_feature_order in best_interventions[predict_test[i]]):
                    total_as_recommended+=1
                    total_remission_in_recommended+= test_target[i]
                remission_predictions.append(rem_rate_dict[cluster_prediction][differential_feature_order])
            remission_rate_from_recommended= total_remission_in_recommended/ total_as_recommended if total_as_recommended>0 else 0
            rrs[NUMBER_OF_CLUSTERS].append(remission_rate_from_recommended)
            auc=   roc_auc_score( np.array(pd_test[target_name]),remission_predictions)
            aucs[NUMBER_OF_CLUSTERS].append(auc)
        print(rrs)
        print(aucs)
