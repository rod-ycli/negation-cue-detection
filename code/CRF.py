import csv
import sys
import sklearn_crfsuite
from sklearn.model_selection import GridSearchCV
import pandas as pd
from SVM import evaluate_classifier, write_predictions_to_file
from sklearn.metrics import make_scorer
from sklearn_crfsuite import metrics

# based on https://github.com/cltl/ma-ml4nlp-labs/blob/main/code/assignment3/CRF.py


def token2features(sentence, i, selected_features):

    features = {'bias': 1.0}

    for feature in selected_features:
        value = sentence[i][feature]

        if value and value not in ['eos'' bos']:  # if there is a value for this feature and it is not 'eos' or 'bos'
            features[feature] = value

    if i == 0:
        features['BOS'] = True
    elif i == len(sentence) - 1:
        features['EOS'] = True

    return features


def sent2features(sent, selected_features):

    return [token2features(sent, i, selected_features) for i in range(len(sent))]


def sent2labels(sent):

    return [d['gold_label'] for d in sent]


def extract_sents_from_file(file_path):

    sents = []
    current_sent = []

    with open(file_path, 'r', encoding='utf8') as infile:
        reader = csv.DictReader(infile, restval='', delimiter='\t', quotechar='\\')
        for row in reader:
            if row['next_lemma'] == 'eos':
                current_sent.append(row)
                sents.append(current_sent)
                current_sent = []
            else:
                current_sent.append(row)

    return sents


def train_crf_model(X_train, y_train):

    # # parameters provided in the original script actually give the best performance on our test set

    # crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
    #                            c1=0.1,
    #                            c2=0.1,
    #                            max_iterations=100,
    #                            all_possible_transitions=True)
    # crf.fit(X_train, y_train)
    #
    # return crf

    classifier = sklearn_crfsuite.CRF()  # use the default parameters

    classifier.fit(X_train, y_train)

    return classifier


def train_crf_model_using_cross_validation(X_train, y_train):

    print("Running cross validation, this will take a while and you should get a FutureWarning")

    classifier = sklearn_crfsuite.CRF()

    # possible parameters: https://sklearn-crfsuite.readthedocs.io/en/latest/api.html#module-sklearn_crfsuite

    parameters = {'algorithm': ['lbfgs'],
                  # 'algorithm': ['lbfgs', 'l2sgd'],
                  'c1': [0, 0.01, 0.1],
                  'c2': [0.01, 0.1, 1],
                  'max_iterations': [100],
                  # 'max_iterations': [100, 1000],
                  'all_possible_transitions': ['True', 'False']}

    # we can't use the same scoring as for SVM because labels are represented differently > as a list of lists, 1 list
    # of labels per sentence
    # https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#hyperparameter-optimization

    f1_scorer = make_scorer(metrics.flat_f1_score, average='macro')

    grid = GridSearchCV(estimator=classifier, param_grid=parameters, cv=5, scoring=f1_scorer)

    try:
        grid.fit(X_train, y_train)

    except AttributeError:
        # https://github.com/TeamHG-Memex/sklearn-crfsuite/issues/60
        print("You have to use sklearn version lower than 0.24 to be able to run cross-validation.")
        print("You can install it by typing \'pip install -U 'scikit-learn<0.24'\' into your terminal.")

    else:
        print(f'Done! Best parameters: {grid.best_params_}')
        print(f'Best result on the training set: {round(grid.best_score_, 3)} macro avg f1-score')
        return grid.best_estimator_


def create_crf_model(trainingfile, selected_features, cross_validation=False):

    train_sents = extract_sents_from_file(trainingfile)
    X_train = [sent2features(s, selected_features) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    if cross_validation:
        crf = train_crf_model_using_cross_validation(X_train, y_train)
    else:
        crf = train_crf_model(X_train, y_train)

    return crf


def run_crf_model(crf, evaluationfile, selected_features):

    test_sents = extract_sents_from_file(evaluationfile)
    X_test = [sent2features(s, selected_features) for s in test_sents]
    y_pred = crf.predict(X_test)

    return y_pred


def train_and_run_crf_model(trainingfile, evaluationfile, selected_features, cross_validation=False):

    crf = create_crf_model(trainingfile, selected_features, cross_validation)

    pred_labels = run_crf_model(crf, evaluationfile, selected_features)

    predictions = []
    for pred in pred_labels:
        predictions += pred

    return predictions


def run_and_evaluate_a_crf_system(train_path, test_path, selected_features, name, cross_validation=False):

    predictions = train_and_run_crf_model(train_path, test_path, selected_features, cross_validation)

    if cross_validation:
        print(f"Running {name.replace('_', ' ')} with best parameters")
    else:
        print(f"Running {name.replace('_', ' ')}")

    write_predictions_to_file(test_path, selected_features, predictions, name)

    df = pd.read_csv(test_path, encoding='utf-8', sep='\t', keep_default_na=False,
                     quotechar='\\', skip_blank_lines=False)

    gold_labels = df['gold_label'].to_list()
    evaluate_classifier(predictions, gold_labels, selected_features, 'CRF')


def main():

    paths = sys.argv[1:]

    if not paths:
        paths = ['../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2_features.txt',
                 '../data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2_features.txt']

    train_path = paths[0]
    test_path = paths[1]

    # use the full set of features
    name = "CRF_full"
    selected_features = ['lemma', 'prev_lemma', 'next_lemma', 'pos_category', 'is_single_cue', 'has_affix', 'affix',
                         'base_is_word', 'base']
    run_and_evaluate_a_crf_system(train_path, test_path, selected_features, name)

    # implement basic cross-validation in combination with the system using all features
    run_and_evaluate_a_crf_system(train_path, test_path, selected_features, name, cross_validation=True)


if __name__ == '__main__':
    main()
