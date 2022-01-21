import os
import sys
import pandas as pd
from nltk.corpus import brown, gutenberg
from nltk.stem import PorterStemmer


def generate_pos_category(pos_tags):
    """
    Assign part-of-speech tags to six categories as follows: ADJ, NN, ADV, VB, PRO, and OTH.
    :param pos_tags: list with pos_tags
    :return: list with POS tags categories
    """
    pos_list = []

    adj = ['JJ', 'JJR', 'JJS']
    nn = ['NN', 'NNS', 'NNP', 'NNPS']
    adv = ['RB', 'RBR', 'RBS']
    vb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    pro = ['PRP', 'PRP$']
   
    for pos_tag in pos_tags:
        if not pos_tag:  # if we had an empty row in our data set
            pos_list.append('')
        elif pos_tag in adj:
            pos_list.append('ADJ')   
        elif pos_tag in nn:
            pos_list.append('NN')  
        elif pos_tag in adv:
            pos_list.append('ADV')  
        elif pos_tag in pro:
            pos_list.append('PRO')  
        elif pos_tag in vb:
            pos_list.append('VERB')  
        else:
            pos_list.append('OTH')  
        
    return pos_list


def extract_previous_and_next(elements):
    """
    Extract previous and preceding token or lemma from a list of tokens or lemmas
    :param elements: list with tokens or lemmas
    :return: list with previous tokens/lemmas, list with next tokens/lemmas
    """
    position_index = 0

    prev_tokens = []
    next_tokens = []

    bos_previous, eos_next = False, False  # flags to tell us where sentences end

    for i in range(len(elements)):

        prev_index = (position_index - 1)
        next_index = (position_index + 1)
        
        # previous token
        if prev_index < 0:
            previous_token = "bos"

        else:
            previous_token = elements[prev_index]
            if previous_token == '':  # if there's an empty line before this token
                previous_token = "bos"
                bos_previous = True

            if bos_previous:
                prev_tokens[position_index - 1] = ''
                bos_previous = False

        prev_tokens.append(previous_token)
            
        # next token
        if eos_next:
            next_token = ''
            eos_next = False

        elif next_index < len(elements):
            next_token = elements[next_index]
            if next_token == '':  # if there's an empty line after this token
                next_token = 'eos'
                eos_next = True
        else: 
            next_token = "eos"

        next_tokens.append(next_token)
            
        # moving to next token in list
        position_index += 1
    
    return prev_tokens, next_tokens


def get_affixal_and_base_features(tokens: list, neg_prefix, neg_suffix, vocab) -> tuple:
    """
    Extract affixal and base features for each token, i.e. has_affix, affix, stem_is_word and stem.
    :return: tuple of four lists of the features.
    """
    has_affix = []
    affix = []
    stem_is_word = []
    stem = []

    ps = PorterStemmer()

    for token in tokens:

        if not token:  # empty line
            has_affix_val, affix_val, stem_is_word_val, stem_val = '', '', '', ''

        # Assign values if the token doesn't have the affixes
        else:
            has_affix_val = 0
            affix_val = ""
            stem_is_word_val = 0
            stem_val = ""

        # Check if the token does have the affixes; if it does, the values will be changed
            for prefix in neg_prefix:
                # If the token starts with one of the prefixes
                if token.startswith(prefix) and token != prefix:
                    # has_affix has value 1
                    has_affix_val = 1
                    # Feature 'affix' captures the prefix
                    affix_val = prefix
                    # If the base_stem is also in our vocab set
                    base_stem = ps.stem(token.replace(prefix, "", 1))
                    if base_stem in vocab:
                        # stem_is_word has value 1
                        stem_is_word_val = 1
                        # Feature 'stem' captures the base
                        stem_val = base_stem
                    break

            if not has_affix_val:  # if token doesn't have a prefix
                # Check if the token ends with one of the suffixes
                for suffix in neg_suffix:
                    if token.endswith(suffix) and token != suffix:
                        has_affix_val = 1
                        affix_val = suffix
                        # Check if the stem of the base appears in the vocab set
                        base_stem = ps.stem(token.replace(suffix, "", 1))
                        if base_stem in vocab:
                            stem_is_word_val = 1
                            stem_val = base_stem
                        break

        # Appending the values to the lists
        has_affix.append(has_affix_val)
        affix.append(affix_val)
        stem_is_word.append(stem_is_word_val)
        stem.append(stem_val)

    return has_affix, affix, stem_is_word, stem


def write_features(input_file, neg_prefix, neg_suffix, vocab):
    """
    Generate a new file containing extracted features.
    :param input_file: the path to preprocessed file
    """
    # Prepare output file
    output_file = input_file.replace('_preprocessed.txt', '_features.txt')

    # Read in preprocessed file
    input_data = pd.read_csv(input_file, encoding='utf-8', sep='\t', header=None, keep_default_na=False,     
                             quotechar='\\', skip_blank_lines=False)

    books = input_data[0]
    sent_num = input_data[1]
    token_num = input_data[2]
    tokens = input_data[3].astype('str').apply(lambda x: x.lower())
    lemmas = input_data[4].astype('str').apply(lambda x: x.lower())
    pos_tags = input_data[5]
    labels = input_data[6]

    # Defining header names
    feature_names = ["book",
                    "sent_num",
                    "token_num",      
                    "token",
                    "lemma",
                    "pos_tag",
                    "pos_category",     
                    "prev_token",
                    "next_token",
                    "prev_lemma",
                    "next_lemma",
                    "has_affix",
                    "affix",
                    "stem_is_word",
                    "stem",
                    "gold_label"]

    # Extracting features
    prev_tokens, next_tokens = extract_previous_and_next(tokens)
    
    pos_category = generate_pos_category(pos_tags)

    prev_lemmas, next_lemmas = extract_previous_and_next(lemmas)

    has_affix, affix, stem_is_word, stem = get_affixal_and_base_features(tokens, neg_prefix, neg_suffix, vocab)

    # Defining feature values for writing to output file
    features_dict = {'book': books, 'sent_num': sent_num, 'token_num': token_num,
                     'token': tokens, 'pos_tag': pos_tags,'pos_category': pos_category, 'lemma': lemmas,
                     'prev_token': prev_tokens, 'next_token': next_tokens,   
                     'prev_lemma': prev_lemmas, 'next_lemma': next_lemmas,
                     'has_affix': has_affix, 'affix': affix,
                     'stem_is_word': stem_is_word, 'stem': stem,
                     'gold_label': labels}

    features_df = pd.DataFrame(features_dict, columns=feature_names)

    # Writing features to file

    # features_df.to_csv(output_file, sep='\t', index=False)  # uncomment & comment the lines below to keep empty lines

    # If we want to get rid of empty lines, we use the next two lines

    features_df_clean = features_df[features_df['book'] != '']  # drop empty rows
    features_df_clean.to_csv(output_file, sep='\t', index=False)


def main() -> None:
    """Extend the features for preprocessed file and save a features-added version."""
    paths = sys.argv[1:]

    if not paths:
        paths = ['../data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2_preprocessed.txt',
                 '../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2_preprocessed.txt']

    # Create negation prefix set and suffix set
    # Negation affixes are acquired from the training and dev data sets
    neg_prefix = {"dis", "im", "in", "ir", "un"}
    neg_suffix = {"less", "lessness", "lessly"}

    # Create corpus vocab set that unions the two biggest corpora in NLTK.
    # Words are stemmed
    ps = PorterStemmer()
    print("Building vocab set. This will take a while.")
    vocab = {ps.stem(word.lower()) for word in gutenberg.words()} | {ps.stem(word.lower()) for word in brown.words()}
    print("Done")

    for path in paths:
        print(f'Extracting features from {os.path.basename(path)}')
        write_features(path, neg_prefix, neg_suffix, vocab)
    

if __name__ == '__main__':
    main()
