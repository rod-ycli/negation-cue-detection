import gensim
import numpy as np
import pandas as pd
import sys
import os
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV


def extract_word_embedding(token, word_embedding_model):

    if token in word_embedding_model:
        vector = word_embedding_model[token]
    else:
        vector = [0]*300
    return vector


def combine_embeddings(file, word_embedding_model):

    lemmas = file['lemma']
    prev_lemmas= file['prev_lemma']
    next_lemmas = file['next_lemma']
    
    concatenate_result = []
    for lemma, prev_lemma, next_lemma in zip(lemmas, prev_lemmas, next_lemmas ):

        # Extract embeddings for all token features
        lemma_embedding = extract_word_embedding(lemma, word_embedding_model)
        prev_lemma_embedding = extract_word_embedding(prev_lemma, word_embedding_model)
        next_lemma_embedding = extract_word_embedding(next_lemma, word_embedding_model)
        
        # Concatenate the embeddings
        concatenate_result.append(np.concatenate((lemma_embedding, prev_lemma_embedding, next_lemma_embedding)))

    return concatenate_result


def make_sparse_features(data, selected_features):
 
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
   
    # Prepare containers
    combined_vectors = []
    sparse = np.array(sparse_features.toarray())

    #  Combines sparse (one-hot-encoded) and dense (e.g. word embeddings) features into a combined feature set.
    for idx, vector in enumerate(sparse):
        combined_vector = np.concatenate((vector, embeddings[idx]))
        combined_vectors.append(combined_vector)
    
    return combined_vectors


def train_mlp_classifier(x_train, y_train):
    
#     clf = MLPClassifier()
#     parameters = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.0001, 0.05],
#     'learning_rate': ['constant','adaptive']}

#     grid = GridSearchCV(estimator=clf, param_grid=parameters, cv=5, scoring='f1_macro')

#     print("Running cross validation, this will take a while and you might get some Convergence Warnings")

#     grid.fit(train_features_vectorized, train_labels)

#     print(f'Done! Best parameters: {grid.best_params_}')

#     return grid.best_estimator_
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=500, random_state=42)
    clf.fit(x_train, y_train)
    return clf


def load_data_embeddings(input_file, test_file, embedding_model_path):

    print('Loading word embedding model...')
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format(
        embedding_model_path, binary=True)
    print('Done loading word embedding model')

    training = pd.read_csv(input_file, encoding='utf-8', sep='\t', keep_default_na=False,     
                             quotechar='\\', skip_blank_lines=False)
    
    training_labels = training['gold_label']

    test = pd.read_csv(test_file, encoding='utf-8', sep='\t', keep_default_na=False,     
                             quotechar='\\', skip_blank_lines=False)
    
    test_labels = test['gold_label']

    return training, training_labels, test, test_labels, embedding_model

def run_classifier(training, training_labels, test, word_embedding_model, selected_features):


    # Extract embeddings for token, prev_token and next_token
    embeddings = combine_embeddings(training, word_embedding_model)

    # Extract and vectorize one-hot features
    sparse_features = make_sparse_features(training, selected_features)
    vec = DictVectorizer()
    sparse_vectors = vec.fit_transform(sparse_features)

    # Combine both kind of features into training data
    training_data = combine_features(sparse_vectors, embeddings)

    # Train network
    print("Training classifier...")
    clf = train_mlp_classifier(training_data, training_labels)
    
#     clf = MLPClassifier()
#     parameters = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.0001, 0.05],
#     'learning_rate': ['constant','adaptive']}

#     grid = GridSearchCV(estimator=clf, param_grid=parameters, cv=5, scoring='f1_macro')

#     print("Running cross validation, this will take a while and you might get some Convergence Warnings")

#     grid.fit(training_data, training_labels)

#     print(f'Done! Best parameters: {grid.best_params_}')

#     classifier = grid.best_estimator_
    
    
    print("Done training classifier")

    # Extract embeddings for token, prev_token and next_token from test data
    embeddings = combine_embeddings(test, word_embedding_model)

    # Extract and vectorize one-hot features for test data
    sparse_features = make_sparse_features(test, selected_features)
    sparse_vectors = vec.transform(sparse_features)

    test_data = combine_features(sparse_vectors, embeddings)

    return clf, test_data


def evaluation(test_labels, prediction):
  
    metrics = classification_report(test_labels, prediction, digits=3)
    print(metrics)

    # Confusion matrix
    data = {'Gold': test_labels, 'Predicted': prediction}
    df = pd.DataFrame(data, columns=['Gold', 'Predicted'])

    confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
    print(confusion_matrix)
    print()


def main() -> None:
    
    paths = sys.argv[1:]

    if not paths:
        paths = ['../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2_features.txt',
                 '../data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2_features.txt']
   
    training_data_path = paths[0]
    test_data_path = paths[1]
    
    embedding_model_path = '../data/GoogleNews-vectors-negative300.bin'
    # Load data and the embedding model
    training, training_labels, test, test_labels, word_embedding_model = load_data_embeddings(training_data_path, test_data_path, embedding_model_path)

    selected_features =  ['pos_category', 'is_single_cue', 'has_affix', 'affix','base_is_word', 'base']

    # Train classifiers

    clf, test_data = run_classifier(training, training_labels, test, word_embedding_model, selected_features)
    
    # Make prediction
    prediction = clf.predict(test_data)

    # Print evaluation
    print('-------------------------------------------------------')
    print("Evaluation of MLP system with the following sparse features:")
    print(selected_features)
    evaluation(test_labels, prediction)


if __name__ == '__main__':
    main()
