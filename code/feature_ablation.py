import sys
from SVM import run_classifier_and_return_predictions_and_gold, evaluate_classifier
from CRF import train_and_run_crf_model
import pandas as pd
from utils import CONFIG


def main(paths=None) -> None:
    """Run ablation analysis for SVM, CRF, MLP classifiers, evaluating on excluding features one by one
     and in specific combinations."""

    if not paths:
        paths = sys.argv[1:3]

    # Feature combination can be specified in the command line
    feature_combination = sys.argv[3:]

    if not paths:
        paths = [CONFIG['train_path'].replace('.txt', '_features.txt'),
                 CONFIG['dev_path'].replace('.txt', '_features.txt')]

    train_path, test_path = paths

    available_features = ['lemma', 'prev_lemma', 'next_lemma', 'pos_category',
                          'is_single_cue', 'has_affix', 'affix', 'base_is_word', 'base']

    combinations = []

    if not feature_combination:
        # By default, this script will run on two parts:
        # First part: exclude features one by one
        # Second part: add selected combinations

        combinations.append(available_features)

        # Exclude features one by one
        for target in available_features:
            combinations.append([f for f in available_features if (f != target)])

        # Add selected combinations
        # unwanted_features = ['base', 'is_single_cue', 'pos_category', 'prev_lemma']
        unwanted_combs = [['base', 'is_single_cue'], ['base', 'pos_category'], ['base', 'prev_lemma'],
                          ['base', 'is_single_cue', 'pos_category'], ['base', 'is_single_cue', 'prev_lemma'],
                          ['base', 'pos_category', 'prev_lemma'],
                          ['base', 'is_single_cue', 'pos_category', 'prev_lemma']]

        for comb in unwanted_combs:
            combinations.append([f for f in available_features if (f not in comb)])

    else:
        combinations.append(feature_combination)

    # CRF ablation
    df = pd.read_csv(test_path, encoding='utf-8', sep='\t', keep_default_na=False,
                     quotechar='\\', skip_blank_lines=False)
    gold_labels = df['gold_label'].to_list()
    for comb in combinations:
        predictions = train_and_run_crf_model(train_path, test_path, comb, custom=True)
        evaluate_classifier(predictions, gold_labels, comb, 'CRF')

    # SVM ablation
    for comb in combinations:
        predictions, gold = run_classifier_and_return_predictions_and_gold(train_path, test_path, comb)
        evaluate_classifier(predictions, gold, comb, 'SVM')


if __name__ == '__main__':
    main()
