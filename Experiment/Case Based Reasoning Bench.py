

from scipy import spatial
from scipy import special
from sklearn.metrics import roc_auc_score, recall_score,  confusion_matrix
from sklearn.model_selection import StratifiedKFold

from examples.examples_utils import *
from Experiment.test_utils import get_dummy_features_indexes

pd_train  = pd.read_csv( "")  # fill your train-data file name here
pd_test = pd.read_csv("") # fill your test-data file name here




target_name = 'remsn'
differential_feature = 'drug'
Neighbors_K=5




def calculate_cosine_similarity(vector1, vector2):
    return (1 - spatial.distance.cosine(vector1, vector2))


def get_closeset_neighbors(user, users):
    dictionary={}
    for index, neighbor in  enumerate(users.values):
        dictionary[index]= calculate_cosine_similarity(user,neighbor)
    return   dictionary

def subset_of_list(list, subset_indexes):
    new_list=[]
    for i in subset_indexes:
        new_list.append(list[i])
    return  new_list

aucs= []
senss=[]
specs=[]
rrs= []

for k in range(20):
    fold_k=7
    for i in range(5):
        pd_all = pd_all.sample(frac=1).reset_index(drop=True)
    kf = StratifiedKFold(n_splits=fold_k, shuffle=True, random_state=2)

    for i in range(1):  # for each fold
        # divide the dataset into train and test set according to fold
        result = next(
            kf.split(np.array(pd_all.drop([target_name], axis=1)), np.array(pd_all[target_name])))
        pd_train = pd_all.iloc[result[0]]
        pd_test = pd_all.iloc[result[1]]
        probs={}
        for index, row in pd_test.iterrows():  # for each sample in the test set
            probs[index] = []
            for differential_feature in differential_features:  # for each possible treatment
                treatment_df= pd_train[pd_train[differential_feature] == 1]  # choose only the samples that received the current treatemtn
                user= row[[f for f in all_features_new_model if f not in differential_features+ [target_name ]]]._values
                users= treatment_df[[f for f in all_features_new_model if f not in differential_features + [target_name]]]
                # choose the top-k similar samples
                neighbors_dictionary= get_closeset_neighbors(user, users)
                sorted_neighbors_dictionary = sorted(neighbors_dictionary.items(), key=lambda kv: kv[1], reverse=True)[:Neighbors_K]
                sorted_neighbors_dictionary= [list(i) for i in sorted_neighbors_dictionary]
                # normalize the similarities (with softmax- so that they will sum up to 1 )
                similarities = np.array([i[1] for i in sorted_neighbors_dictionary])
                similarities= special.softmax(similarities)  # softmax
                for ind, key in enumerate(sorted_neighbors_dictionary):
                    sorted_neighbors_dictionary[ind][1]= similarities[ind]
                # predict according to neighbors and similarities
                counterfactual_prediction=0
                for i,item in enumerate(sorted_neighbors_dictionary):
                    counterfactual_prediction+= item[1] * int(list(treatment_df['remsn'])[item[0]])
                probs[index].append(counterfactual_prediction)

        print("here")
        #proto_net.get_differential_predictions(probs, train_loader)
        max_values = []
        remission_recommended_drug = 0
        total_recommended_drug = 0
        differential_feature_data = get_dummy_features_indexes(pd_test[all_features_new_model].drop([target_name], axis=1), 'drug',
                                                               prefix_sep='_')
        differential_feature_indexes = [i[0] for i in differential_feature_data]

        k=1
        predictions=[]
        for key in probs:
            actual_treatment_index=    subset_of_list(pd_test.loc[key], differential_feature_indexes).index(1)
            predictions.append( probs[key][actual_treatment_index])
            topk_indexes= np.argpartition(list(probs[key]), -k)[-k:]
            # if the recommended  differential feature is the actual differential feature in the dataset
            for index in topk_indexes:
                if pd_test.loc[key][differential_feature_indexes[index]]==1:
                    total_recommended_drug += 1
                    remission_recommended_drug += pd_test[target_name][key]
                    break
            # print the top k "best" treatments  for each patient
            print(np.array2string(np.argsort(-np.array(list(probs[key])))[0:k]))
            max_values.append(max(probs[key]))
        # print(max_values)

        print("****************************REMISSION RATE OF PATIENTS WHO RECEIVED RECOMMENDED DRUG********")

        rr= remission_recommended_drug / float(total_recommended_drug)
        # print the remission rate within recommended drug

        print(rr)
        rrs.append(rr)

        print(total_recommended_drug)
        print("****************************")

        auc=    roc_auc_score( np.array(pd_test[target_name]),[round(x) for x in predictions]) # todo maybe round the predictions for more accurate predictions
        print(auc)
        aucs.append(auc)

        tn, fp, fn, tp = confusion_matrix(np.round(np.array(pd_test[target_name])),np.round(np.array(predictions))).ravel()

        spec = tn / float(tn + fp)
        sens = tp / float(tp + fn)

        print("specs and sen")
        senss.append(sens)
        print(senss)
        specs.append(spec)
        print(specs)


        print("AUC")
        print(auc)


print(aucs)
print(rrs)
print(senss)
print(specs)