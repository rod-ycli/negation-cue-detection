import sys
from utils import CONFIG
from preprocessing import main as preprocess
from feature_extraction import main as extract_features
from SVM import main as run_svm_systems
from CRF import main as run_crf_systems
from mlp_classifier import main as run_mlp_system


def main():
    paths = sys.argv[1:]

    if not paths:  # if no paths are passed to the function through the command line
        paths = [CONFIG['train_path'], CONFIG['dev_path']]

    train_path, dev_path = paths
    preprocess(paths)
    extract_features([train_path.replace('.txt', '_preprocessed.txt'), dev_path.replace('.txt', '_preprocessed.txt')])
    train_path, dev_path = train_path.replace('.txt', '_features.txt'), dev_path.replace('.txt', '_features.txt')

    # SVM
    run_svm_systems([train_path, dev_path])
    # CRF
    run_crf_systems([train_path, dev_path])
    # MLP
    run_mlp_system([train_path, dev_path])
    # ablation ?


if __name__ == '__main__':
    main()
