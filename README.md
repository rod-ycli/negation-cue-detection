# Negation Cue Detection, TM group 4
This project has been developed as a part of the Applied Text Mining 1: Methods course at VU Amsterdam in January 2022.

### Authors
YC Roderick Li, Shuyi Shen, Panagiota Tselenti and Lahorka Nikolovski.

### Repository structure

#### ./annotated_documents

This directory contains the 4 sets of annotations of negation cues for Assignment 2, each with 10 documents.

#### ./data
Data files should be placed in this directory to ensure all the code works as intended.
Files used for this project are specified below and can be obtained through the course Canvas page.
1. SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt
2. SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt

#### ./code
This directory contains all the code. The scripts should be run in the order specified below to ensure experimental 
results are reproduced.

##### Step 1.

Run the `processing.py` script.

The script can be run from the command line, passing one or more arguments specifying the path(s) to the .txt file(s) 
to be preprocessed:

`python preprocessing.py [path_to_file] ... [path_to_file]`

Alternatively, running the script will automatically preprocess all the files used in the experiment and specified 
under ./data below. In that case, make sure the files are in the data directory and that you execute the script from 
the code directory, or adjust the paths.

As a results of running the script, for every preprocessed file a new file containing preprocessed data will be 
generated in the same directory.



