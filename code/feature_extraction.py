import os
import string
import sys
import pandas as pd
from nltk.corpus import gutenberg
from typing import List, Tuple
from utils import CONFIG


def generate_pos_category(pos_tags: List[str]) -> List[str]:
    """
    Assign part-of-speech tags to seven categories as follows: ADJ, NN, ADV, VB, PRO, PUNCT and OTH.
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
        if not pos_tag:  # if we have an empty row
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
        elif pos_tag[0] in string.punctuation:
            pos_list.append('PUNCT') 
        else:
            pos_list.append('OTH')  
        
    return pos_list


def extract_previous_and_next(elements: List[str]) -> Tuple[List[str], List[str]]:
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
        
                            
def get_negation_features(lemmas: list, single_neg_cues: set, neg_prefix: set, neg_suffix: set, vocab: set) \
        -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """Extract negation features for each lemma, i.e. is_neg_cue, has_affix, affix, base_is_word and base."""
    neg_cues = []           
    has_affix = []
    affix = []
    base_is_word = []
    base = []

    for lemma in lemmas:

        if not lemma:  # empty line
            is_neg_cue, has_affix_val, affix_val, base_is_word_val, base_val = '', '', '', '', ''
         
        # Assign values for the case that the token doesn't have the affixes
        else:
            is_neg_cue = 0
            has_affix_val = 0
            affix_val = ""
            base_is_word_val = 0
            base_val = ""
            
            if lemma in single_neg_cues:  # checking if lemma is found in list of collected single-word negation cues
                is_neg_cue = 1
                
            if not is_neg_cue:  # if not, we continue checking for affixal values

                # Check if the base does have the affixes
                for suffix in neg_suffix:
                    # If the token ends with one of the suffixes
                    if lemma.endswith(suffix):
                        # Get the base
                        base_lemma = lemma.replace(suffix, "")
                        if len(base_lemma) > 2:
                            # has_affix has value 1
                            has_affix_val = 1
                            # Feature 'affix' captures the prefix
                            affix_val = suffix
                            # If the base is also in our vocab set
                            if base_lemma in vocab:
                                # base_is_word has value 1
                                base_is_word_val = 1
                                # Feature 'base' captures the base
                                base_val = base_lemma
                            break

                if not has_affix_val:  # if lemma doesn't have a suffix
                    # Check if it starts with one of the prefixes
                    for prefix in neg_prefix:
                        # If the lemma starts with one of the prefixes
                        if lemma.startswith(prefix):
                            # Get the base
                            base_lemma = lemma.replace(prefix, "", 1)
                            if len(base_lemma) > 3:
                                # has_affix has value 1
                                has_affix_val = 1
                                # Feature 'affix' captures the prefix
                                affix_val = prefix
                                # If the base is also in our vocab set
                                if base_lemma in vocab:
                                    # base_is_word has value 1
                                    base_is_word_val = 1
                                    # Feature 'base' captures the base
                                    base_val = base_lemma
                                break

        # Appending the values to the lists
        neg_cues.append(is_neg_cue)
        has_affix.append(has_affix_val)
        affix.append(affix_val)
        base_is_word.append(base_is_word_val)
        base.append(base_val)

    return neg_cues, has_affix, affix, base_is_word, base


def write_features(input_file: str, output_file: str, neg_cues_set: set, neg_prefix: set, neg_suffix: set,
                   vocab: set) -> None:
    """Generate a new file containing extracted features."""

    # Read in preprocessed file
    input_data = pd.read_csv(input_file, encoding='utf-8', sep='\t', header=None, keep_default_na=False,     
                             quotechar='\\', skip_blank_lines=False)

    # Extracting features and labels
    tokens = input_data[3].astype('str').apply(lambda x: x.lower())
    lemmas = input_data[4].astype('str').apply(lambda x: x.lower())
    pos_tags = input_data[5]
    labels = input_data[6]

    # Extracting additional features
    # prev_tokens, next_tokens = extract_previous_and_next(tokens)
    prev_lemmas, next_lemmas = extract_previous_and_next(lemmas)
    pos_categories = generate_pos_category(pos_tags)  

    neg_cues, has_affix, affix, base_is_word, base = get_negation_features(lemmas, neg_cues_set, neg_prefix, neg_suffix,
                                                                           vocab)

    # Defining feature values for writing to output file
    features_dict = {'book': input_data[0], 'sent_num': input_data[1], 'token_num': input_data[2],
                     'token': tokens,  # 'prev_token': prev_tokens, 'next_token': next_tokens,
                     'lemma': lemmas, 'prev_lemma': prev_lemmas, 'next_lemma': next_lemmas,
                     # 'pos_tag': pos_tags,
                     'pos_category': pos_categories,
                     'is_single_cue': neg_cues,
                     'has_affix': has_affix, 'affix': affix,
                     'base_is_word': base_is_word, 'base': base,
                     'gold_label': labels}

    # Defining header names
    feature_names = features_dict.keys()

    features_df = pd.DataFrame(features_dict, columns=feature_names)
    # Remove empty rows
    features_df_clean = features_df[features_df['token'] != '']

    # Writing features to file
    features_df_clean.to_csv(output_file, sep='\t', index=False)


def main(paths=None) -> None:
    """Extend the features for preprocessed file and save a features-added version."""
    if not paths:  # if no paths are passed to the function
        paths = sys.argv[1:]

    if not paths:  # if no paths are passed to the function through the command line
        paths = [CONFIG['train_path'].replace('.txt', '_preprocessed.txt'),
                 CONFIG['dev_path'].replace('.txt', '_preprocessed.txt')]
        
    single_neg_cues_file = CONFIG['single_neg_cues_file']
    neg_cues_set = set()
        
    with open(single_neg_cues_file, 'r', encoding='utf8') as infile:
        for line in infile:
            neg_cues_set.add(line.strip())    

    # Create negation prefix set and suffix set
    # Negation affixes are acquired from the training and dev data sets
    # Common negation prefix "non-" is manually added
    neg_prefix = {"dis", "im", "in", "ir", "un", "non"}
    neg_suffix = {"less", "lessness", "lessly"}

    # Create corpus vocab set from the Gutenberg corpus in NLTK
    print("Building vocab set...")
    vocab = {word.lower() for word in gutenberg.words()}
    print("Done")

    for path in paths:
        # Prepare output file
        output_file = path.replace('_preprocessed.txt', '_features.txt')
        print(f'Extracting features from {os.path.basename(path)}')
        write_features(path, output_file, neg_cues_set, neg_prefix, neg_suffix, vocab)


if __name__ == '__main__':
    main()
