from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import csv
import sys
from tabulate import tabulate


# parts of the code are inspired by code available at https://github.com/cltl/ma-ml4nlp-labs/tree/main/code/assignment2

def extract_features_and_labels(file_path, selected_features):

    features = []
    labels = []

    with open(file_path, 'r', encoding='utf8') as infile:
        # restval specifies value to be used for missing values
        reader = csv.DictReader(infile, restval='', delimiter='\t', quotechar='\\')
        for row in reader:
            feature_dict = {}
            for feature_name in selected_features:
                if row[feature_name]:  # if there is a value for this feature
                    feature_dict[feature_name] = row[feature_name]
            features.append(feature_dict)
            labels.append(row['gold_label'])

    return features, labels


def create_classifier(train_features, train_labels):

    classifier = LinearSVC()
    vec = DictVectorizer()
    train_features_vectorized = vec.fit_transform(train_features)
    classifier.fit(train_features_vectorized, train_labels)
        
    return classifier, vec


def select_classifier_using_cross_validation(train_features, train_labels):

    classifier = LinearSVC()
    vec = DictVectorizer()
    train_features_vectorized = vec.fit_transform(train_features)

    # define parameters we want to try out
    # for possibilities, see
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
    parameters = {'loss': ['hinge', 'squared_hinge'],
                  'C': [0.8, 0.9, 1],
                  'tol': [1e-5, 1e-4],
                  'max_iter': [1000, 2000]}

    grid = GridSearchCV(estimator=classifier, param_grid=parameters, cv=5, scoring='f1_macro')

    print("Running cross validation, this will take a while and you might get some Convergence Warnings.")

    grid.fit(train_features_vectorized, train_labels)

    print(f'Done! Best parameters: {grid.best_params_}, f1_macro: '
          f'{round(grid.score(train_features_vectorized, train_labels), 3)}')

    classifier = grid.best_estimator_

    return classifier, vec


def get_predicted_and_gold_labels(test_path, vectorizer, classifier, selected_features):

    # we use the same function as above (guarantees features have the same name and form)
    test_features, gold_labels = extract_features_and_labels(test_path, selected_features)
    
    # we need to use the same fitting as before, so now we only transform the current features according to this
    # mapping (using only transform)
    test_features_vectorized = vectorizer.transform(test_features)
    predictions = classifier.predict(test_features_vectorized)

    return predictions, gold_labels


def print_confusion_matrix(predictions, gold_labels):

    labels = sorted(set(gold_labels))
    cf_matrix = confusion_matrix(gold_labels, predictions, labels=labels)
    # transform confusion matrix into a dataframe
    df_cf_matrix = pd.DataFrame(cf_matrix, index=labels, columns=labels)

    print(tabulate(df_cf_matrix, headers='keys', tablefmt='psql'))


def print_precision_recall_f1_score(predictions, gold_labels, digits=3):

    # get the report in dictionary form
    report = classification_report(gold_labels, predictions, zero_division=0, output_dict=True)
    # remove unwanted metrics
    report.pop('accuracy')
    report.pop('weighted avg')
    # transform dictionary into a dataframe and round the results
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.round(digits)
    df_report['support'] = df_report['support'].astype(int)

    print(tabulate(df_report, headers='keys', tablefmt='psql'))


def run_classifier_and_write_file(train_path, test_path, selected_features, cross_validation=False, name=''):

    train_features, train_labels = extract_features_and_labels(train_path, selected_features)
    
    if cross_validation:
        classifier, vectorizer = select_classifier_using_cross_validation(train_features, train_labels)
        print('Result using the best parameters:')

    else:
        classifier, vectorizer = create_classifier(train_features, train_labels)

    predictions, gold_labels = get_predicted_and_gold_labels(test_path, vectorizer, classifier, selected_features)

    test_data = pd.read_csv(test_path, encoding='utf-8', sep='\t', keep_default_na=False,
                             quotechar='\\', skip_blank_lines=False)
    pred_keys = ['book', 'sent_num', 'token_num'] + selected_features + ['gold_label']
    pred_dict = dict()
    for key in pred_keys:
        pred_dict[key] = test_data[key]
    pred_dict.update({'pred': predictions})
    columns = pred_dict.keys()
    pred_df = pd.DataFrame(pred_dict, columns=columns)
    pred_df.to_csv(test_path.replace('_features.txt', f'_pred{name}.txt'), sep='\t', index=False)

    print()
    print('----> ' + 'SVM' + ' with ' + ' , '.join(selected_features) + ' as features <----')
    print_confusion_matrix(predictions, gold_labels)
    print_precision_recall_f1_score(predictions, gold_labels)


def main() -> None:
    
    paths = sys.argv[1:]

    if not paths:
        paths = ['../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2_features.txt',
                 '../data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2_features.txt']

    train_path = paths[0]
    test_path = paths[1]

    available_features = ['token', 'lemma', 'prev_lemma', 'next_lemma', 'pos_category', 'is_single_cue', 'has_affix',
                          'affix', 'base_is_word', 'base']

    # implement basic cross-validation in combination with the baseline system
    run_classifier(train_path, test_path, selected_features, cross_validation=True, name='_baseline')

    # use all features
    selected_features = ['lemma', 'prev_lemma', 'next_lemma', 'pos_category', 'is_single_cue', 'has_affix', 'affix',
                         'base_is_word', 'base']

    # implement basic cross-validation in combination with the system using all features
    run_classifier(train_path, test_path, selected_features, cross_validation=True)


if __name__ == '__main__':
    main()
