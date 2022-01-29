import sys
# from SVM import run_classifier_and_return_predictions_and_gold, evaluate_classifier
from CRF import run_and_evaluate_a_crf_system
# from mlp_classifier import load_data_embeddings, run_classifier, evaluation


def main() -> None:
    """Run ablation analysis for SVM, CRF, MLP classifiers, evaluating on excluding features one by one
     and in specific combinations."""

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

        # combinations.append(['lemma', 'prev_lemma', 'pos_category', 'has_affix', 'affix', 'base_is_word', 'base'])
        # combinations.append(['lemma', 'prev_lemma', 'next_lemma', 'has_affix', 'affix', 'base_is_word', 'base'])
        # combinations.append(['lemma', 'prev_lemma', 'next_lemma', 'pos_category', 'has_affix', 'affix', 'base_is_word'])
        # combinations.append(['lemma', 'prev_lemma', 'next_lemma', 'has_affix', 'affix', 'base_is_word'])
        # combinations.append(['lemma', 'prev_lemma', 'is_single_cue', 'has_affix', 'affix', 'base_is_word', 'base'])
        # combinations.append(['lemma', 'prev_lemma', 'is_single_cue', 'pos_category', 'has_affix', 'affix', 'base_is_word'])
        # combinations.append(['lemma', 'prev_lemma', 'is_single_cue', 'has_affix', 'base_is_word'])
        # combinations.append(['lemma', 'prev_lemma', 'next_lemma', 'has_affix', 'base_is_word'])
    else:
        combinations.append(feature_combination)

    ### SVM ablation
    # for comb in combinations:
    #     predictions, gold = run_classifier_and_return_predictions_and_gold(train_path, test_path, comb)
    #     evaluate_classifier(predictions, gold, comb)

    ### CRF ablation
    for comb in combinations:
        run_and_evaluate_a_crf_system(train_path, test_path, comb, name='CRF')

    ### MLP ablation
    # embedding_model_path = '../data/GoogleNews-vectors-negative300.bin'
    # # Load data and the embedding model
    # training, training_labels, test, test_labels, word_embedding_model = load_data_embeddings(train_path,
    #                                                                                          test_path,
    #                                                                                          embedding_model_path)
    #
    # for comb in combinations:
    #     # Train classifiers
    #     clf, test_data = run_classifier(training, training_labels, test, word_embedding_model, comb)
    #
    #     # Make prediction
    #     prediction = clf.predict(test_data)
    #
    #     # Print evaluation
    #     print('-------------------------------------------------------')
    #     print("Evaluation of MLP system with the following sparse features:")
    #     print(comb)
    #     evaluation(test_labels, prediction)


if __name__ == '__main__':
    main()
