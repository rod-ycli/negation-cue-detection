from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
import pandas as pd
import os
import json


with open('config.json', 'r', encoding='utf8') as config_file:
    CONFIG = json.load(config_file)


def generate_confusion_matrix(predictions, gold_labels):
    """Generate a confusion matrix."""

    labels = sorted(set(gold_labels))
    cf_matrix = confusion_matrix(gold_labels, predictions, labels=labels)
    # transform confusion matrix into a dataframe
    df_cf_matrix = pd.DataFrame(cf_matrix, index=labels, columns=labels)

    return df_cf_matrix


def calculate_precision_recall_f1_score(predictions, gold_labels, digits=3):
    """Calculate evaluation metrics."""

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


def evaluate_classifier(predictions, gold_labels, selected_features, name):
    """Produce full evaluation of classifier."""

    print(f"Evaluating {name.replace('_', ' ')} with {', '.join(selected_features)} as features:")

    cf_matrix = generate_confusion_matrix(predictions, gold_labels)
    report = calculate_precision_recall_f1_score(predictions, gold_labels)

    print(tabulate(cf_matrix, headers='keys', tablefmt='psql'))
    # print(cf_matrix.to_latex())  # print and paste to Overleaf

    print(tabulate(report, headers='keys', tablefmt='psql'))
    # print(report.to_latex())  # print and paste to Overleaf


def write_predictions_to_file(test_path, selected_features, predictions, name):
    """Write predictions from classifier to file."""

    df = pd.read_csv(test_path, encoding='utf-8', sep='\t', keep_default_na=False,
                     quotechar='\\', skip_blank_lines=False)
    pred_keys = ['book', 'sent_num', 'token_num'] + selected_features + ['gold_label']
    pred_dict = dict()
    for key in pred_keys:
        pred_dict[key] = df[key]
    pred_dict.update({'pred': predictions})
    columns = pred_dict.keys()
    pred_df = pd.DataFrame(pred_dict, columns=columns)
    pred_dir = os.path.dirname(test_path) + '/predictions/'  # save predictions to a separate directory
    if not os.path.isdir(pred_dir):
        os.mkdir(pred_dir)
    out_path = pred_dir + os.path.basename(test_path).replace('_features.txt', f'_pred_{name}.txt')
    pred_df.to_csv(out_path, sep='\t', index=False)
