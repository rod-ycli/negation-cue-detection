from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import csv
import sys
from tabulate import tabulate


# parts of the code are inspired by code available at https://github.com/cltl/ma-ml4nlp-labs/tree/main/code/assignment2

def extract_features_and_labels(file_path, selected_features):

    data = []
    targets = []

    with open(file_path, 'r', encoding='utf8') as infile:
        # restval specifies value to be used for missing values
        reader = csv.DictReader(infile, restval='', delimiter='\t', quotechar='\\')
        for row in reader:
            feature_dict = {}
            for feature_name in selected_features:
                if row[feature_name]:  # if there is a value for this feature
                    feature_dict[feature_name] = row[feature_name]
            data.append(feature_dict)
            targets.append(row['gold_label'])

    return data, targets


def create_classifier(train_features, train_targets):

    classifier = svm.LinearSVC()
    vec = DictVectorizer()
    tokens_vectorized = vec.fit_transform(train_features)
    classifier.fit(tokens_vectorized, train_targets)
        
    return classifier, vec


def get_predicted_and_gold_labels(test_path, vectorizer, classifier, selected_features):

    # we use the same function as above (guarantees features have the same name and form)
    features, gold_labels = extract_features_and_labels(test_path, selected_features)
    
    # we need to use the same fitting as before, so now we only transform the current features according to this
    # mapping (using only transform)
    test_features_vectorized = vectorizer.transform(features)
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
    # transform dictionary into a dataframe and round the values
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.round(digits)
    df_report['support'] = df_report['support'].astype(int)

    print(tabulate(df_report, headers='keys', tablefmt='psql'))


def run_classifier(train_path, test_path, selected_features):

    feature_values, labels = extract_features_and_labels(train_path, selected_features)
    
    classifier, vectorizer = create_classifier(feature_values, labels)
    predictions, gold_labels = get_predicted_and_gold_labels(test_path, vectorizer, classifier, selected_features)
    print()
    print('----> ' + 'SVM' + ' with ' + ' , '.join(selected_features) + ' as features <----')
    print_confusion_matrix(predictions, gold_labels)
    print_precision_recall_f1_score(predictions, gold_labels)

    return predictions


def main() -> None:
    
    paths = sys.argv[1:]

    if not paths:
        paths = ['../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2_features.txt',
                 '../data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2_features.txt']

    train_path = paths[0]
    test_path = paths[1]

    available_features = ['token', 'lemma', 'prev_lemma', 'next_lemma', 'pos_category', 'is_single_cue', 'has_affix',
                          'affix', 'base_is_word', 'base']
    # baseline
    selected_features = ['token']
    run_classifier(train_path, test_path, selected_features)

    # use all features
    selected_features = ['lemma', 'prev_lemma', 'next_lemma', 'pos_category', 'is_single_cue', 'has_affix', 'affix',
                         'base_is_word', 'base']

    run_classifier(train_path, test_path, selected_features)


if __name__ == '__main__':
    main()
