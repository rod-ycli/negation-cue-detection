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

    classifier = LinearSVC(random_state=42)
    vec = DictVectorizer()
    train_features_vectorized = vec.fit_transform(train_features)
    classifier.fit(train_features_vectorized, train_labels)
        
    return classifier, vec


def create_classifier_using_cross_validation(train_features, train_labels):

    classifier = LinearSVC(random_state=42)
    vec = DictVectorizer()
    train_features_vectorized = vec.fit_transform(train_features)

    # define parameters we want to try out
    # for possibilities, see
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC

    parameters = {'loss': [hinge], ['squared_hinge'],
                  'C': [0.6, 0.7, 0.8],
                  'tol': [0.01, 0.03],
                  'max_iter': [1500, 2000]}

    grid = GridSearchCV(estimator=classifier, param_grid=parameters, cv=5, scoring='f1_macro')

    print("Running cross validation, this will take a while and you might get some Convergence Warnings")

    grid.fit(train_features_vectorized, train_labels)

    print(f'Done! Best parameters: {grid.best_params_}')
    print(f'Best result on the training set: {round(grid.best_score_, 3)} macro avg f1-score')

    return grid.best_estimator_, vec


def get_predicted_and_gold_labels(test_path, vectorizer, classifier, selected_features):

    # we use the same function as above (guarantees features have the same name and form)
    test_features, gold_labels = extract_features_and_labels(test_path, selected_features)
    
    # we need to use the same fitting as before, so now we only transform the current features according to this
    # mapping (using only transform)
    test_features_vectorized = vectorizer.transform(test_features)
    print(f'Using vectors of dimensionality {test_features_vectorized.shape[1]}')
    predictions = classifier.predict(test_features_vectorized)

    return predictions, gold_labels


def generate_confusion_matrix(predictions, gold_labels):

    labels = sorted(set(gold_labels))
    cf_matrix = confusion_matrix(gold_labels, predictions, labels=labels)
    # transform confusion matrix into a dataframe
    df_cf_matrix = pd.DataFrame(cf_matrix, index=labels, columns=labels)

    return df_cf_matrix


def calculate_precision_recall_f1_score(predictions, gold_labels, digits=3):

    # get the report in dictionary form
    report = classification_report(gold_labels, predictions, zero_division=0, output_dict=True)
    # remove unwanted metrics
    report.pop('accuracy')
    report.pop('weighted avg')
    # transform dictionary into a dataframe and round the results
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.round(digits)
    df_report['support'] = df_report['support'].astype(int)

    return df_report


def evaluate_classifier(predictions, gold_labels, selected_features, name='SVM'):

    print(f"Evaluating {name} with {', '.join(selected_features)} as features:")

    cf_matrix = generate_confusion_matrix(predictions, gold_labels)
    report = calculate_precision_recall_f1_score(predictions, gold_labels)

    print(tabulate(cf_matrix, headers='keys', tablefmt='psql'))
    # print(cf_matrix.to_latex())  # print and paste to Overleaf

    print(tabulate(report, headers='keys', tablefmt='psql'))
    # print(report.to_latex())  # print and paste to Overleaf


def run_classifier_and_return_predictions_and_gold(train_path, test_path, selected_features, cross_validation=False):

    train_features, train_labels = extract_features_and_labels(train_path, selected_features)

    if cross_validation:
        classifier, vectorizer = create_classifier_using_cross_validation(train_features, train_labels)

    else:
        classifier, vectorizer = create_classifier(train_features, train_labels)

    predictions, gold_labels = get_predicted_and_gold_labels(test_path, vectorizer, classifier, selected_features)

    return predictions, gold_labels


def write_predictions_to_file(test_path, selected_features, predictions, name):

    df = pd.read_csv(test_path, encoding='utf-8', sep='\t', keep_default_na=False,
                     quotechar='\\', skip_blank_lines=False)
    pred_keys = ['book', 'sent_num', 'token_num'] + selected_features + ['gold_label']
    pred_dict = dict()
    for key in pred_keys:
        pred_dict[key] = df[key]
    pred_dict.update({'pred': predictions})
    columns = pred_dict.keys()
    pred_df = pd.DataFrame(pred_dict, columns=columns)
    pred_df.to_csv(test_path.replace('_features.txt', f'_pred_{name}.txt'), sep='\t', index=False)


def run_and_evaluate_a_system(train_path, test_path, selected_features, name, cross_validation=False):

    predictions, gold_labels = run_classifier_and_return_predictions_and_gold(train_path, test_path, selected_features,
                                                                              cross_validation)
    if cross_validation:
        print(f"Running {name.replace('_', ' ')} with best parameters")
    else:
        print(f"Running {name.replace('_', ' ')}")
    write_predictions_to_file(test_path, selected_features, predictions, name)
    evaluate_classifier(predictions, gold_labels, selected_features)


def main() -> None:
    
    paths = sys.argv[1:]

    if not paths:
        paths = ['../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2_features.txt',
                 '../data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2_features.txt']

    train_path = paths[0]
    test_path = paths[1]

    # run baseline system
    name = "baseline_SVM"
    selected_features = ['token']
    run_and_evaluate_a_system(train_path, test_path, selected_features, name)

    # use the full set of features
    name = "full_SVM"
    selected_features = ['lemma', 'prev_lemma', 'next_lemma', 'pos_category', 'is_single_cue', 'has_affix', 'affix',
                         'base_is_word', 'base']

    run_and_evaluate_a_system(train_path, test_path, selected_features, name)

    # implement basic cross-validation in combination with the system using all features
    run_and_evaluate_a_system(train_path, test_path, selected_features, name, cross_validation=True)


if __name__ == '__main__':
    main()
