import gensim
import numpy as np
import pandas as pd
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from utils import evaluate_classifier, write_predictions_to_file, CONFIG


# parts of the code are inspired by code available at https://github.com/cltl/ma-ml4nlp-labs/tree/main/code/assignment3


def extract_word_embedding(token, word_embedding_model):
    """
    Function that returns the 300-dimension dense representation 
    of the token otherwise '0' if the token does not exist in 
    the embedding dictionary.
    
    :param token: the token
    :param word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
  
    """

    if token in word_embedding_model:
        vector = word_embedding_model[token]
    else:
        vector = [0]*300
    return vector


def combine_embeddings(file, word_embedding_model):
    """
    Extract dense representation for lemmas, previous lemmas and next lemmas and 
    using word embeddings, concatenate them, and return a list containing 
    combined embeddings 
    
    :param file: a pandas dataframe
    :param word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    """
    lemmas = file['lemma']
    prev_lemmas = file['prev_lemma']
    next_lemmas = file['next_lemma']
    
    concatenate_result = []
    for lemma, prev_lemma, next_lemma in zip(lemmas, prev_lemmas, next_lemmas):

        # Extract embeddings for all lemmas features

        lemma_embedding = extract_word_embedding(lemma, word_embedding_model)
        prev_lemma_embedding = extract_word_embedding(prev_lemma, word_embedding_model)
        next_lemma_embedding = extract_word_embedding(next_lemma, word_embedding_model)
        
        # Concatenate the embeddings
        concatenate_result.append(np.concatenate((lemma_embedding, prev_lemma_embedding, next_lemma_embedding)))

    return concatenate_result


def make_sparse_features(data, selected_features):
    
    """
    Convert selected traditional features into one-hot encoding
    and return a sparse representation 
    :param data: a pandas dataframe
    :param selected_features: a list that contains the header names of 
     the traditional features
    """
 
    sparse_features = []
    for i in range(len(data)):

        # Prepare feature dictionary for each sample
        
        feature_dict = {}

        # Add feature values to dictionary
        for feature in selected_features:
            value = data[feature][i]
            feature_dict[feature] = value

        # Append all sample feature dictionaries
        sparse_features.append(feature_dict)

    return sparse_features


def combine_features(sparse_features, embeddings):
    """
    Combines one-hot encoding (traditional features) with word embedding 
    (current, previous, and next lemmas) 
    
    :param sparse_features: sparse representations of traditional features
    :param embeddings: word embeddings of the lemmas
    """
    # Prepare containers
    combined_vectors = []
    sparse = np.array(sparse_features.toarray())

    #  Combines sparse (one-hot-encoded) and dense (e.g. word embeddings) features into a combined feature set.
    for idx, vector in enumerate(sparse):
        combined_vector = np.concatenate((vector, embeddings[idx]))
        combined_vectors.append(combined_vector)
    
    return combined_vectors


def train_mlp_classifier(x_train, y_train):
    
    """
    Train a multi-layer classifier using default setting for 
    solver, alpha and activation function. Hidden layer size 
    and random state were changed to optimize the model performance
    """

    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=500, random_state=2)
    clf.fit(x_train, y_train)
    return clf


def load_data_embeddings(input_file, test_file, embedding_model_path):
    
    """
    load in the worb emdding model and return training and test 
    data/ labels for running MLP classifier
    
    :param input_file: path to the training data
    :param test_file: path to the test data
    :param embedding_model_path: path to the 300-dimnesion embedding model
    """
    print('Loading word embedding model...')
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format(
        embedding_model_path, binary=True)
    print('Done loading word embedding model')
    
    # read in the training data using pandas
    training = pd.read_csv(input_file, encoding='utf-8', sep='\t', keep_default_na=False,     
                           quotechar='\\', skip_blank_lines=False)
    
    training_labels = training['gold_label']
    
    # read in the test data using pandas
    test = pd.read_csv(test_file, encoding='utf-8', sep='\t', keep_default_na=False,     
                       quotechar='\\', skip_blank_lines=False)
    
    test_labels = test['gold_label']

    return training, training_labels, test, test_labels, embedding_model


def run_classifier(training, training_labels, test, word_embedding_model, selected_features):
    """
    Function that runs MLP classifier on a training and test file and return predicted labels
    for evaluation 
    

    :param training: training data returned from "load_data_embeddings" function
    :param training_labels: training labels returned from "load_data_embeddings" function
    :param test: test data returned from "load_data_embeddings" function
    :param word_embedding_model: path to the 300-dimnesion embedding model
    :param selected_features: a list of selected features 
    """
    # Extract embeddings for lemmas, previous lemmas, and next lemmas from taining data.
    embeddings = combine_embeddings(training, word_embedding_model)

    # Convert the traditional features into one-hot encoding for training data
    sparse_features = make_sparse_features(training, selected_features)
    vec = DictVectorizer()
    sparse_vectors = vec.fit_transform(sparse_features)

    # Combine dense and sparse representation
    training_data = combine_features(sparse_vectors, embeddings)

    # Train network
    print("Training classifier...")
    clf = train_mlp_classifier(training_data, training_labels)

    print("Done training classifier")

    # Extract embeddings for lemmas, previous lemmas, next lemmas from test data
    embeddings = combine_embeddings(test, word_embedding_model)

    # Convert the traditional features into one-hot encoding for test data
    sparse_features = make_sparse_features(test, selected_features)
    sparse_vectors = vec.transform(sparse_features)

    test_data = combine_features(sparse_vectors, embeddings)

    return clf, test_data


def evaluation(test_labels, prediction):
    """
    Function that prints out a confusion matrix with 
    precision, recall and f-score
    
    :param test_labels: gold labels
    :param prediction: predicted labels
    """
     
    metrics = classification_report(test_labels, prediction, digits=3)
    print(metrics)

    # Confusion matrix
    data = {'Gold': test_labels, 'Predicted': prediction}
    df = pd.DataFrame(data, columns=['Gold', 'Predicted'])

    confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
    print(confusion_matrix)
    print()


def main(paths=None) -> None:
    """run all the functions and return evaluation reports for different combination of features"""
    if not paths:
        paths = sys.argv[1:]

    if not paths:
        paths = [CONFIG['train_path'].replace('.txt', '_features.txt'),
                 CONFIG['dev_path'].replace('.txt', '_features.txt')]
   
    training_data_path, test_data_path = paths

    embedding_model_path = CONFIG['embedding_model_path']
    # Load data and the embedding model
    training, training_labels, test, test_labels, word_embedding_model = load_data_embeddings(training_data_path,
                                                                                              test_data_path,
                                                                                              embedding_model_path)
    
    # prepare the container for ablation analysis 
    available_features = ['pos_category', 'is_single_cue', 'has_affix', 'affix', 'base_is_word', 'base']

    # add all the traditional features
    selected_features = [available_features]

    # run the system using all features
    name = 'system5_MLP'
    clf, test_data = run_classifier(training, training_labels, test, word_embedding_model, selected_features[0])
    predictions = clf.predict(test_data)
    feature_names = ['lemma', 'prev_lemma', 'next_lemma'] + selected_features[0]
    evaluate_classifier(predictions, test_labels, feature_names, name)
    write_predictions_to_file(test_data_path, feature_names, predictions, name)

    # # Uncomment for ablation: remove traditional features one by one
    # for features in available_features:
    #     selected_features.append([f for f in available_features if (f != features)])
    #
    # for features in selected_features[1:]:
    #     clf, test_data = run_classifier(training, training_labels, test, word_embedding_model, features)
    #
    #     # Make prediction
    #     predictions = clf.predict(test_data)
    #
    #     # Print evaluation
    #     feature_names = ['lemma', 'prev_lemma', 'next_lemma'] + features
    #     evaluate_classifier(predictions, test_labels, feature_names, 'MLP')


if __name__ == '__main__':
    main()
