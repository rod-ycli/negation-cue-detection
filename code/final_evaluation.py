import csv
import os
from CRF import run_and_evaluate_a_crf_system
from preprocessing import main as preprocess
from feature_extraction import main as extract_features
from utils import CONFIG


def combine_files(infile_paths, outfile_path):
    with open(outfile_path, 'w',  newline='', encoding='utf8') as outfile:
        filewriter = csv.writer(outfile, delimiter='\t', quotechar='\\')
        i = 1
        for infile_path in infile_paths:
            with open(infile_path, 'r', encoding='utf8') as infile:
                filereader = csv.reader(infile, delimiter='\t', quotechar='\\')
                for row in filereader:
                    filewriter.writerow(row)
                if i < len(infile_paths):
                    filewriter.writerow([])
                i += 1


def main():
    # evaluate System 4 on test data
    selected_features = ['lemma', 'prev_lemma', 'next_lemma', 'pos_category', 'is_single_cue', 'has_affix', 'affix',
                         'base_is_word', 'base']
    train_path = CONFIG['train_path'].replace('.txt', '_features.txt')
    dir_name = os.path.dirname(train_path)
    # used combined test files
    test_path = dir_name + '/SEM-2012-SharedTask-CD-SCO-test_features.txt'

    if not os.path.exists(test_path):
        test_paths = [CONFIG['test_path_cardboard'], CONFIG['test_path_circle']]
        combined_test_path = dir_name + '/SEM-2012-SharedTask-CD-SCO-test.txt'
        combine_files(test_paths, combined_test_path)
        preprocess([combined_test_path])
        preprocessed_combined_test_path = combined_test_path.replace('.txt', '_preprocessed.txt')
        extract_features([preprocessed_combined_test_path])
        # test_path = preprocessed_combined_test_path.replace('_preprocessed.txt', '_features.txt')

    run_and_evaluate_a_crf_system(train_path, test_path, selected_features, 'system4_CRF',
                                  custom=True)

    # train System 4 on a combined train and dev dataset
    train_path = dir_name + '/SEM-2012-SharedTask-CD-SCO-training-and-dev-simple.v2_features.txt'
    if not os.path.exists(train_path):
        train_paths = [CONFIG['train_path'], CONFIG['dev_path']]
        combined_train_path = dir_name + '/SEM-2012-SharedTask-CD-SCO-training-and-dev-simple.v2.txt'
        combine_files(train_paths, combined_train_path)
        preprocess([combined_train_path])
        preprocessed_combined_train_path = combined_train_path.replace('.txt', '_preprocessed.txt')
        extract_features([preprocessed_combined_train_path])
        # train_path = preprocessed_combined_train_path.replace('_preprocessed.txt', '_features.txt')

    name = "system4_CRF_trained_on_train_and_dev"
    run_and_evaluate_a_crf_system(train_path, test_path, selected_features, name,
                                  cross_validation=True)


if __name__ == '__main__':
    main()
