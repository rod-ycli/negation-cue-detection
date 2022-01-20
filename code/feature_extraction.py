import sys
import pandas as pd
<<<<<<< HEAD
import csv
import os
from typing import List, Dict
import nltk
=======
from nltk.corpus import brown, gutenberg
from nltk.stem import PorterStemmer

# Create negation prefix set and suffix set
# Negation affixes are acquired from the training and dev data sets
neg_prefix = {"dis", "im", "in", "ir", "un"}
neg_suffix = {"less", "lessness", "lessly"}

# Create corpus vocab set that unions the two biggest corpora in NLTK.
# Words are stemmed
ps = PorterStemmer()
print("Building vocab set...")
vocab = {ps.stem(word.lower()) for word in gutenberg.words()} | {ps.stem(word.lower()) for word in brown.words()}
print("Done")
>>>>>>> 0112e70ddb18de773b47f0f0a3caae7046132878

def pos_category_extraction(postag):
    """
    Function to extract part-of-speech tags 
    and assign POS tags to six categories as follows:
    ADJ, NN, ADV, VB, PRO, and OTH .
    :param tokens: list with tokens
    :return: list with POS tags categories
    """
    pos_list = []
    adj = ['JJ','JJR','JJS']
    nn = ['NN','NNS','NNP','NNPS']
    adv = ['RB','RBR','RBS']
    vb = ['VB','VBD','VBG','VBN','VBP','VBZ']
    pro = ['PRP','PRP$']
   
    for pos_tag in postag:
        if pos_tag in adj:
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

def previous_and_next_token_extraction(tokens):
    """
    Function to extract previous and preceding token and lemma
    from tokens and lemma list obtained from preprocessing.py.
    :param tokens: list with tokens or lemmas
    :return: list with previous tokens/lemmas, list with next tokens/lemas
    """
    position_index = 0

    prev_tokens = []
    next_tokens = []
    
    for i in range(len(tokens)):

        prev_index = (position_index - 1)
        next_index = (position_index + 1)
        
        #previous token
        if prev_index < 0:
            previous_token = "bos"
        else: 
            previous_token = tokens[prev_index]

        prev_tokens.append(previous_token)
            
        #next token
        if next_index < len(tokens):
            next_token = tokens[next_index]
        else: 
            next_token = "eos"

        next_tokens.append(next_token)
            
        #moving to next token in list 
        position_index += 1
    
    return prev_tokens, next_tokens

<<<<<<< HEAD
def new_function(x):
    
    if  x[3]:
        pass
    else:
        
        x[3] = 'EOS'
        x[4] = 'EOS'
    return x 
=======
def get_affixal_and_base_features(tokens: list) -> tuple:
    """This function extracts affixal and base features for each token, i.e. has_affix, affix, stem_is_word and stem.
    :return: tuple of four lists of the features."""
    has_affix = []
    affix = []
    stem_is_word = []
    stem = []

    for token in tokens:
        for prefix in neg_prefix:
            # If the token starts with one of the prefixes
            if token.startswith(prefix) and token!=prefix:
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
                else:
                    stem_is_word_val = 0
                    stem_val = ""
                break
            else:
                # Check if the token ends with one of the suffixes
                for suffix in neg_suffix:
                    if token.endswith(suffix) and token!=suffix:
                        has_affix_val = 1
                        affix_val = suffix
                        # Check if the stem of the base appears in the vocab set
                        base_stem = ps.stem(token.replace(suffix, "", 1))
                        if base_stem in vocab:
                            stem_is_word_val = 1
                            stem_val = base_stem
                        else:
                            stem_is_word_val = 0
                            stem_val = ""
                        break
                    # Assign values if the token doesn't have the affixes
                    else:
                        has_affix_val = 0
                        affix_val = ""
                        stem_is_word_val = 0
                        stem_val = ""

        # Appending the values to the lists
        has_affix.append(has_affix_val)
        affix.append(affix_val)
        stem_is_word.append(stem_is_word_val)
        stem.append(stem_val)

    return has_affix, affix, stem_is_word, stem
>>>>>>> 0112e70ddb18de773b47f0f0a3caae7046132878

def write_features(input_file):
    """
    Function to generate a new file containing extended features:
    pos_category, preceding and next tokens as well as lemmas.
    
    :param input_file: the path to preprocessed file
    :return: a new file extended with previous and next tokens/lemmas, 
    as well as postag category.    
    """
    # Prepare output file
    output_file = input_file.replace('_preprocessed.txt', '-features.txt')

    # Read in preprocessed file
    input_data = pd.read_csv(input_file, encoding='utf-8', sep='\t', header=None, keep_default_na=False,     
                             quotechar='\\', skip_blank_lines=False)\
   

    input_data = input_data.apply(lambda x: new_function(x), axis=1)
    
    books = input_data.iloc[:, 0]
    sent_num = input_data.iloc[:, 1]
    token_num = input_data.iloc[:, 2]
<<<<<<< HEAD
    tokens = input_data.iloc[:, 3].astype('str').apply(lambda x: x.lower())
 
    lemmas = input_data.iloc[:, 4].astype('str').apply(lambda x: x.lower())
   
=======
    tokens = input_data.iloc[:, 3]
    tokens = [str(token.lower()) for token in tokens]
    lemmas = input_data.iloc[:, 4]
    lemmas = [str(lemma.lower()) for lemma in lemmas]
>>>>>>> 0112e70ddb18de773b47f0f0a3caae7046132878
    pos_tags = input_data.iloc[:, 5]
    labels = input_data.iloc[:, -1]
    # Defining header names
    feature_names = ["books",
                    "sent_num",
                    "token_num",      
                    "token",
                    "lemma",
                    "pos_tag",
                    "pos_category",     
                    "prev_token",
                    "next_token",
                    "prev_lemmas",
                    "next_lemmas",
                    "has_affix",
                    "affix",
                    "stem_is_word",
                    "stem",
                    "gold_label"]

  
    
    prev_tokens, next_tokens = previous_and_next_token_extraction(tokens)    
    
    pos_category = pos_category_extraction(pos_tags)

    prev_lemmas, next_lemmas = previous_and_next_token_extraction(lemmas)

    has_affix, affix, stem_is_word, stem = get_affixal_and_base_features(tokens)

    # Defining feature names for writing to output file 
    features_dict = {'books':books, 'sent_num':sent_num, 'token_num':token_num,
                     'token': tokens, 'pos_tag': pos_tags,'pos_category': pos_category,'lemma': lemmas, 
                     'prev_token': prev_tokens, 'next_token': next_tokens,   
                     'prev_lemmas':prev_lemmas,'next_lemmas':next_lemmas,
                     'has_affix': has_affix, 'affix': affix,
                     'stem_is_word': stem_is_word, 'stem': stem,
                     'gold_label': labels}

    features_df = pd.DataFrame(features_dict, columns=feature_names)
    
    

    # Writing features names to file 
    features_df.to_csv(output_file, sep='\t', index=False)


def main() -> None:
    """Extend the features for preprocessed file and save a features-added version of it."""
    paths = sys.argv[1:]

    if not paths:
        paths = ['../data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2_preprocessed.txt',
         '../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2_preprocessed.txt']

    for path in paths:
        write_features(path)
    

if __name__ == '__main__':
    main()
