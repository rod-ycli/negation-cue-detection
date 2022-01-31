from CRF import run_and_evaluate_a_crf_system


# validate on the development data

train_path = '../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2_features.txt'
dev_path = '../data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2_features.txt'

selected_features = ['lemma', 'prev_lemma', 'next_lemma', 'pos_category', 'is_single_cue', 'has_affix', 'affix', 'base_is_word']

name = "best_CRF"

run_and_evaluate_a_crf_system(train_path, dev_path, selected_features, name, custom=True)
