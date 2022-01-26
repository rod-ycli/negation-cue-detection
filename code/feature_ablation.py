import sys
from SVM import run_classifier_and_return_predictions_and_gold, evaluate_classifier
from CRF import run_and_evaluate_a_crf_system


def main() -> None:

    # Feature: lemma, previous_lemma, next_lemma, pos_category, ...
    # ... is_single_cue, has_affix, affix, base_is_word, base

    paths = sys.argv[1:3]
    # Feature combination can be specified in the command line
    feature_combination = sys.argv[3:]

    if not paths:
        paths = ['../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2_features.txt',
                 '../data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2_features.txt']

    train_path = paths[0]
    test_path = paths[1]

    available_features = ['lemma', 'prev_lemma', 'next_lemma', 'pos_category',
                          'is_single_cue', 'has_affix', 'affix', 'base_is_word', 'base']

    combinations = []
    if not feature_combination:
        # By default, this script will run on two parts:
        # First part: exclude features one by one
        # Second part: add selected combinations

        # Exclude features one by one
        # for target in available_features:
        #     combinations.append([f for f in available_features if (f != target)])

        # Add selected combinations
        combinations.append(['lemma', 'prev_lemma', 'pos_category', 'has_affix', 'affix', 'base_is_word', 'base'])
        combinations.append(['lemma', 'prev_lemma', 'next_lemma', 'has_affix', 'affix', 'base_is_word', 'base'])
        combinations.append(['lemma', 'prev_lemma', 'next_lemma', 'pos_category', 'has_affix', 'affix', 'base_is_word'])
        combinations.append(['lemma', 'prev_lemma', 'next_lemma', 'has_affix', 'affix', 'base_is_word'])
        combinations.append(['lemma', 'prev_lemma', 'is_single_cue', 'has_affix', 'affix', 'base_is_word', 'base'])
        combinations.append(['lemma', 'prev_lemma', 'is_single_cue', 'pos_category', 'has_affix', 'affix', 'base_is_word'])
        combinations.append(['lemma', 'prev_lemma', 'is_single_cue', 'has_affix', 'base_is_word'])
        combinations.append(['lemma', 'prev_lemma', 'next_lemma', 'has_affix', 'base_is_word'])
    else:
        combinations.append(feature_combination)

    for comb in combinations:
        predictions, gold = run_classifier_and_return_predictions_and_gold(train_path, test_path, comb)
        evaluate_classifier(predictions, gold, comb)

    # CRF ablation
    # for comb in combinations:
    #     run_and_evaluate_a_crf_system(train_path, test_path, comb, name='CRF')


if __name__ == '__main__':
    main()
