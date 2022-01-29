# Negation Cue Detection, TM group 4
This project has been developed as a part of the Applied Text Mining 1: Methods course at VU Amsterdam in January 2022.

### Authors
YC Roderick Li, Shuyi Shen, Panagiota Tselenti and Lahorka Nikolovski.

### Repository structure

#### ./annotated_documents

This directory contains the 4 sets of annotations of negation cues for Assignment 2, each with 10 documents.

#### ./data
Data files should be placed in this directory to ensure everything works as intended.
Files used for this project are specified below and can be obtained through the course Canvas page.
1. SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt
2. SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt

#### ./code
This directory contains all the code. The scripts should be run in the order specified below to ensure experimental results are reproduced.

##### Step 1.

Run the `processing.py` script.

The script can be run from the command line, passing one or more arguments specifying the path(s) to the .txt file(s) to be preprocessed:

`python preprocessing.py [path_to_file] ... [path_to_file]`

Alternatively, running the script as is will preprocess all the files needed for the experiment. In that case, make sure the files are in the data directory and that you execute the script from the code directory, or that you adjust the paths.

As a results of running the script, for every preprocessed file a new .txt file containing preprocessed data will be created in the same directory as the original file.

##### Step 2.
Run the `feature_extraction.py` script.

The script can be run from the command line, passing one or more arguments specifying the path(s) to the preprocessed .txt file(s) obtained in the last step:

`python feature_extraction.py [path_to_file] ... [path_to_file]`

Alternatively, running the script as is will create all feature files needed for the experiment. In that case, make sure the files are in the data directory and that you execute the script from the code directory, or that you adjust the paths.

As a result of running the script, .txt file(s) containing the features will be created in the same directory that stores the original files(s).


##### Step 3.
Run the `SVM.py` script to train, test and evaluate the SVM classifiers.

Run the `CRF.py` script to train, test and evaluate the CRF classifiers.

Run the `mlp_classifier.py` script to train, test and evaluate the MLP classifiers. In order to run this script, `word2vec` model `GoogleNews-vectors-negative300.bin` needs to be placed in the `../data/` directory. It can be obtained via https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz.

These scripts can all be run from the command line, passing two arguments specifying the paths to the preprocessed .txt files obtained in step 2:

`python file_name_script path_to_preprocessed_train_dataset path_to_preprocessed_test_dataset

Alternatively, running the scripts without passing the arguments will run the experiments as described in the report. In that case, make sure the files are in the data directory and that you execute the script from the code directory, or that you adjust the paths.

As a result of running the scripts, .txt files containing the predictions will be created in the same directory that stores the original files.

For the ablation study, run the `feature_ablation.py` script.

This script can be run from the command line, passing two arguments specifying the paths to the preprocessed .txt files obtained in step 2 and features to be used when training the classifier:

`python feature_ablation.py path_to_preprocessed_train_dataset path_to_preprocessed_test_dataset feature1 [feature2] ... [feature9]`

Alternatively, running the script without passing the arguments will run the ablation analysis as described in the report. In that case, make sure the files are in the data directory and that you execute the script from the code directory, or that you adjust the paths.

###### Bonus: Exploratory Data Analysis notebook
EDA.ipynb was used the explore the training and development datasets.
