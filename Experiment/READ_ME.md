This repository includes the cose that were used in order to obtain the results reported in: "Treatment selection using prototyping in latent-space withapplication to depression treatment"

The MDD experiment files are all in the main folder (\Experiment) and the synthetic data experiment, including the generation of the synthetic data are in a separate inner file (\Experiment\Synthetic_Data_Experiment). In addition, the CFRNET implementation was taken from https://github.com/clinicalml/cfrnet and was modified in order to fit our datasets. We zipped the modified implemention and included it here (CFRNET zip file). 

The implementaion of the DPNN model is currently unavavilabe due to privacy issues. The DNN model that is used in both experiments (called Vulcan in the paper) is available in https://github.com/Aifred-Health/Vulcan .
 
**Running Requirements and Environments

All the code except for the CFRnet files run with Ptyhon 3.6 and requires torch , numpy, pandas and sklearn libraries.
CFRNet runs on Python 2.6 and  requires, in addition tensorflow in addtion to numpy, pandas and sklearn libraries.
