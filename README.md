# Negation Cue Detection, TM group 4
This project has been developed as a part of the Applied Text Mining 1: Methods course at VU Amsterdam in January 2022.

### Authors
YC Roderick Li, Shuyi Shen, Panagiota Tselenti and Lahorka Nikolovski.


### Repository structure


#### ./annotated_documents
This directory contains the 4 sets of annotations of negation cues for Assignment 2, each with 10 documents.


#### ./data
Data files needed to run the experiments are specified in `./data/README.md`. We suggest placing them in this directory. Otherwise, adjust the paths in `./code/config.json`.


#### ./code
This directory contains all the code developed in the course of the project. The scripts should be run in the order specified below to ensure experimental results are reproduced. The paths to files needed to run the experiments are specified in the `config.json` file. Be sure to adjust the paths if needed.


##### Step 0: Requirements
Install the required Python modules specified in `requirements.txt`. This can be done by running `pip install -r requirements.txt`.


##### Step 1: Classification experiments
Run the `all_experiments.py` script. This script will run all the steps needed to reproduce experiments described in the report. It will take care of preprocessing the data files, extracting features, and building and evaluating the baseline system and 5 machine learning systems used in the experiments.  Results of prediction for each system will be written to a file in `./data/predictions`.

Alternativelly, different experimental steps can also be conducted one by one as described below. Running the scripts without passing the arguments will run on the paths specified in the config.json file. As a result of running the scripts, new .txt files containing preprocessed data will be created in the `/predictions` directory:

###### 1A: Preprocessing
Run the `preprocessing.py` script.

The script can be run from the command line, passing one or more arguments specifying the path(s) to the .txt file(s) to be preprocessed:
`python preprocessing.py [path_to_file] ... [path_to_file]`

###### 1B: Feature extraction
Run the `feature_extraction.py` script.

The script can be run from the command line, passing one or more arguments specifying the path(s) to the preprocessed .txt file(s) obtained in the last step:
`python feature_extraction.py [path_to_file] ... [path_to_file]`

###### 1C: Running the experiments
Run the `SVM.py` script to train, test and evaluate the SVM classifiers.

Run the `CRF.py` script to train, test and evaluate the CRF classifiers.

Run the `mlp_classifier.py` script to train, test and evaluate the MLP classifiers.

These scripts can all be run from the command line, passing two arguments specifying the paths to the .txt files obtained in 1B:
`python file_name_script path_to_train_dataset_with_features_extracted path_to_test_dataset_with_features_extracted`


##### Step 2: Feature ablation
For the ablation study, run the `feature_ablation.py` script.

This script can be run from the command line, passing two arguments specifying the paths to the preprocessed .txt files obtained in the feature extraction step and features to be used when training the classifier:
`python feature_ablation.py path_to_train_dataset_with_features_extracted path_to_test_dataset_with_features_extracted feature1 [feature2] ... [feature9]`

Alternatively, running the script without passing the arguments will run analysis on the paths specified in the `config.json` file.


##### Step 3: Evaluate final system on the test set
Run the `final_evaluation.py` script. This script will evaluate the final system on the combined cardboard and circle test datasets. The results of prediction will be written to a file in `./data/predictions`. 


###### Bonus: Exploratory Data Analysis notebook
EDA.ipynb was used the explore the training and development datasets.
