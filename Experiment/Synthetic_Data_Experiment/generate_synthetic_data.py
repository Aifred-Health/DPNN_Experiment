import numpy as np
import enum
from scipy.special import softmax
from scipy import stats

NUMBER_OF_PROTOTYPES= 5

# constants
PROTOTYPE_DISTRIBUTION= True  # If true- the samples are generated with
NUMBER_OF_LATENT_FEATURES=10
NUMBER_OF_NON_LATENT_FEATURES= 20

# determines the portion of features that are "noise"- meaning features that are not relevant for the cluster_prediction
NOISE_FEATURES_RATIO= 0.2

NUMBER_OF_SAMPLES=10000
NUMBER_OF_CLASSIFICATIONS= 2
POSSIBLE_INTERVENTIONS=["0", "1", "2", "3"]

# inline funtions
RELU= lambda x: np.maximum(x,0)
EXP= lambda x: np.exp(x)

class Function_types(enum.Enum):
    LINEAR=1
    NON_LINEAR=2

PRED_FUN_TYPE= Function_types.NON_LINEAR


"""

"""
class Syntethic_Data_Generator():
    """A class for generating data that describes sample, an intervention and the outcome.
    A sample includes is a list of describing features
    An intervention is one element from a finite list of possible intervention (POSSIBLE_INTERVENTIONS)
    An outcome- is the result following the intervention. Here we consider binary case (desired outcome and non-desired)

    The data is generated through  the following process:


    First, generate the following parameters:
    1) Randomly generate a list of prototypes, each sample includes a list of d_1 features
    2) randomly generate cluster_prediction functions, that map a pair of (sample, intervention) to outcome
    3) In addition- generate "decoder function" that transform the samples from the latent space (that is important for  )


    Then, generate the data samples (the rows). For each sample d
    1. First, generate the sample randomly according to the distribution of one of the possible proptypes. Each sample is represneted by d_1 features
    2. Decode the samples into the non-latent space
    3. Add noise features.

    The data is supposed to simulate the data available from clinical experiment, that includes discription of
    patients, a prescribed medicine (intervention)
    """
    def __init__(self):
        self.number_of_noise_features= int(NUMBER_OF_NON_LATENT_FEATURES *NOISE_FEATURES_RATIO)
        self.decoder= self.initialize_decoder()
        self.latent_feature_distribution_dict= self.initialize_feature_distribution()
        self.samples = self.sample_generator()
        self.decoded_samples= self.decode()
        self.prediction_function= self._initialize_prediction_function(PRED_FUN_TYPE)
        self.classification, self.differential_predictions= self.classify()
        self.print_to_csv()


    #prototypes_classification = np.zeros((NUMBER_OF_PROTOTYPES, len(DIFFERENTIAL_FEATURES), NUMBER_OF_CLASSIFICATIONS))
    def initialize_decoder(self, function_type= Function_types.LINEAR):
        return np.random.rand(NUMBER_OF_LATENT_FEATURES, NUMBER_OF_NON_LATENT_FEATURES-self.number_of_noise_features)


    # initialize parameters (mean and std) for the distribution of the latent features
    def initialize_feature_distribution(self):
        latent_feature_distribution_dict = {}
        if(not PROTOTYPE_DISTRIBUTION):
            for i in range(NUMBER_OF_LATENT_FEATURES):
                latent_feature_distribution_dict[i] = {}
                latent_feature_distribution_dict[i]["mean"] = 0
                latent_feature_distribution_dict[i]["std"] = 1
        else:
            for i in range(NUMBER_OF_PROTOTYPES):
                latent_feature_distribution_dict[i]= {}
                latent_feature_distribution_dict[i] = [np.random.normal(0,10,NUMBER_OF_LATENT_FEATURES), np.abs(np.random.normal(0,2))]

        return latent_feature_distribution_dict

    def _initialize_prediction_function(self,function_type= Function_types.LINEAR):
        prediction_functions= {}
        for i in range(len(POSSIBLE_INTERVENTIONS)):
            if function_type == Function_types.LINEAR:
                prediction_functions[POSSIBLE_INTERVENTIONS[i]]= np.random.rand(NUMBER_OF_LATENT_FEATURES, NUMBER_OF_CLASSIFICATIONS)
            elif (function_type== Function_types.NON_LINEAR):
                prediction_functions[POSSIBLE_INTERVENTIONS[i]] = [None, None]
                prediction_functions[POSSIBLE_INTERVENTIONS[i]][0]= np.random.rand(NUMBER_OF_LATENT_FEATURES, int(NUMBER_OF_LATENT_FEATURES / 2))
                prediction_functions[POSSIBLE_INTERVENTIONS[i]][1]= np.random.rand(int(NUMBER_OF_LATENT_FEATURES / 2), NUMBER_OF_CLASSIFICATIONS)

        return prediction_functions

    # create a array of random samples
    def sample_generator(self):
        samples= np.zeros((NUMBER_OF_SAMPLES, NUMBER_OF_LATENT_FEATURES))
        for i in range(NUMBER_OF_SAMPLES):
            if(not PROTOTYPE_DISTRIBUTION):
                for j in range(NUMBER_OF_LATENT_FEATURES):
                    samples[i][j] = np.random.normal(self.latent_feature_distribution_dict[j]["mean"], self.latent_feature_distribution_dict[j]["std"], 1)[0]
            else:
                samples[i]= self.latent_feature_distribution_dict[i%NUMBER_OF_PROTOTYPES][0] +  np.random.normal(0, self.latent_feature_distribution_dict[i%NUMBER_OF_PROTOTYPES][1], NUMBER_OF_LATENT_FEATURES)
        return samples

    # convert the latent variables to non-hidden features, and add noise
    def decode(self):
        #decoded_samples = np.zeros((NUMBER_OF_SAMPLES, NUMBER_OF_NON_LATENT_FEATURES-self.number_of_noise_features))
        #noise_features= np.zeros((NUMBER_OF_SAMPLES, self.number_of_noise_features))
        full_samples= np.zeros((NUMBER_OF_SAMPLES, NUMBER_OF_NON_LATENT_FEATURES))
        for i in range(NUMBER_OF_SAMPLES):
            # decode with decoder function and add noise
            decoded_sample =np.dot(self.samples[i], self.decoder)*np.dot(self.samples[i], self.decoder) +np.random.normal(0,1,NUMBER_OF_NON_LATENT_FEATURES-self.number_of_noise_features)
            noise_feature= np.random.normal(0,1,self.number_of_noise_features)
            full_samples[i]= np.append(decoded_sample, noise_feature)
        return full_samples
    # def prototype_classifier():
    #     for i in range(NUMBER_OF_PROTOTYPES):
    #         for j in range(len(DIFFERENTIAL_FEATURES)):
    #             prototypes_classification[i][j] = np.random.dirichlet(np.ones(NUMBER_OF_CLASSIFICATIONS), size=1)
    #     print(prototypes_classification)

    def classify(self):
        classifications= np.zeros((NUMBER_OF_SAMPLES, NUMBER_OF_CLASSIFICATIONS))
        differential_predictions= np.zeros((NUMBER_OF_SAMPLES, len(POSSIBLE_INTERVENTIONS)))
        for i in range(NUMBER_OF_SAMPLES):
            sample = self.samples[i]
            #sample= EXP(self.samples[i])  #todo- try non linear classification functions
            #exp_sample = RELU(exp_sample)
            # if(i%2== 1):
            #     exp_sample= EXP(self.samples[i])
            #     classifications[i] = [0,1] if np.argmax(exp_sample) >0   else [1,0]
            # else:
            #     classifications[i] = [0,1] if np.sum(sample) >0  else [1,0]
            if(PRED_FUN_TYPE== Function_types.LINEAR):
                mult = np.dot(sample, self.prediction_function[ POSSIBLE_INTERVENTIONS[i % len(POSSIBLE_INTERVENTIONS)]][0])
            elif(PRED_FUN_TYPE== Function_types.NON_LINEAR):
                mult = np.dot(sample, self.prediction_function[ POSSIBLE_INTERVENTIONS[i % len(POSSIBLE_INTERVENTIONS)]][0])
                mult= RELU(mult)
                mult = np.dot(mult, self.prediction_function[ POSSIBLE_INTERVENTIONS[i % len(POSSIBLE_INTERVENTIONS)]][1])

            # for j in range(1,len(self.prediction_function[POSSIBLE_INTERVENTIONS[0]])):
            #     mult = np.dot(mult,
            #                   self.prediction_function[POSSIBLE_INTERVENTIONS[i % len(POSSIBLE_INTERVENTIONS)]][j])
            classifications[i] =mult
            #classifications[i]= np.dot(classifications[i], self.prediction_function[ POSSIBLE_INTERVENTIONS[i % len(POSSIBLE_INTERVENTIONS)]][1])
            classifications[i]= softmax(classifications[i])
            for j in range(len(POSSIBLE_INTERVENTIONS)):
                if (PRED_FUN_TYPE == Function_types.LINEAR):
                    mult = np.dot(sample,
                                  self.prediction_function[POSSIBLE_INTERVENTIONS[i % len(POSSIBLE_INTERVENTIONS)]][0])
                elif (PRED_FUN_TYPE == Function_types.NON_LINEAR):
                    mult = np.dot(sample,
                                  self.prediction_function[POSSIBLE_INTERVENTIONS[j]][0])
                    mult= RELU(mult)
                    mult = np.dot(mult,
                                  self.prediction_function[POSSIBLE_INTERVENTIONS[j]][1])
                differential_predictions[i][j] = softmax(mult)[0]
        return classifications, differential_predictions

    # print all features and labels to a csv files (csv dataset)
    def print_to_csv(self):
        filename= "pdata.csv" if PROTOTYPE_DISTRIBUTION else "data.csv"
        with open(filename, "w") as f:
            feature_names = ""
            for i in range(NUMBER_OF_NON_LATENT_FEATURES):
                # regular features names will be in the format  featureI (I is their )
                feature_names += "feature" + str(i) + ","
            for i in range(len(POSSIBLE_INTERVENTIONS)):
                feature_names += "intervention_" + str(POSSIBLE_INTERVENTIONS[i]) + ","
            for i in range(len(POSSIBLE_INTERVENTIONS)):
                feature_names += "real_pred_inter" + str(POSSIBLE_INTERVENTIONS[i]) + ","
            f.write(feature_names+"target" + "\n")

            for i in range(NUMBER_OF_SAMPLES):
                bin_class = 0 if self.classification[i][1] > self.classification[i][0] else 1
                intervention_np_array= np.zeros(len(POSSIBLE_INTERVENTIONS))
                intervention_np_array[i % len(POSSIBLE_INTERVENTIONS)]=1
                sample_and_intervenation= np.append(np.around(self.decoded_samples[i], decimals=1), intervention_np_array)
                sample_and_intervenation= np.append(sample_and_intervenation,np.around(self.differential_predictions[i], decimals=2))
                row_string = ','.join(map(str, sample_and_intervenation)) + "," + str(bin_class)
                f.write(row_string+"\n")



    def test(self):
        print(self.decoded_samples)


data_generator= Syntethic_Data_Generator()




