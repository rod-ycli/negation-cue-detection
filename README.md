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
Run the `SVM.py` script to train, test and evaluate SVM classifiers.

Run the `CRF.py` script to train, test and evaluate CRF classifiers.

Run the `mlp_classifier.py` script to train, test and evaluate MLP classifiers. In order to run this script, `word2vec` model `GoogleNews-vectors-negative300.bin` need to be put in the `../data/` directory.

Thees scripts can all be run from the command line, passing one or more arguments specifying the path(s) to the preprocessed .txt file(s) obtained in the last step:

`python [file_name] [path_to_file] ... [path_to_file]`

Alternatively, running the scripts as are will create results and evaluations of the experiments. In that case, make sure the files are in the data directory and that you execute the script from the code directory, or that you adjust the paths.

As a result of running the scripts, .txt files containing the predictions will be created in the same directory that stores the original files(s).

For the ablation study, run the `feature_ablation.py` script.

This script can be run from the command line, passing one or more arguments specifying the path(s) to the preprocessed .txt file(s) obtained and specified features:

`python [file_name] [path_to_file] [path_to_file] [feature] ([feature])...`

Alternatively, running the scripts as are will create evaluations of the experiments. In that case, make sure the files are in the data directory and that you execute the script from the code directory, or that you adjust the paths.

###### Bonus: Exploratory Data Analysis notebook
EDA.ipynb was used the explore the training and development datasets.
