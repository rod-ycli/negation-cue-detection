import sklearn
import csv
import sys
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn import metrics
import pandas as pd
from sklearn.metrics import classification_report

def token2features(sentence, i):

    token = sentence[i][0]
    prev_token = sentence[i][1]
    next_token = sentence[i][2]
    lemma = sentence[i][3]
    prev_lemma = sentence[i][4]
    next_lemma = sentence[i][5]
    pos_tag = sentence[i][6]
    pos_category = sentence[i][7]
    has_affix = sentence[i][8]
    affix = sentence[i][9]
    stem_is_word = sentence[i][10]
    stem = sentence[i][11]
    
    features = {
        'bias': 1.0,
        'token': token.lower(),
        'prev_token': prev_token, 
        'next_token':next_token,
        'lemma':lemma,
        'prev_lemma':prev_lemma,
        'next_lemma':next_lemma,
        'pos_tag':pos_tag,
        'pos_category':pos_category,
        'has_affix':has_affix,
        'affix':affix,
        'stem_is_word':stem_is_word,
        'stem':stem
    }
    if i == 0:
        features['BOS'] = True
    elif i == len(sentence) -1:
        features['EOS'] = True
    
    
    return features

def sent2features(sent):
    return [token2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    #if you added features to your input file, make sure to add them here as well.
    #print([ner for token, postag, chunklabel, label, pretoken, nexttoken, prepos,nextpos, capital, ner in sent])
    return [gold_label for token, prev_token, next_token, lemma, prev_lemma, next_lemma, 
            pos_tag, pos_category, has_affix, affix, stem_is_word, stem, gold_label in sent]

def sent2tokens(sent):
    return [token for token, prev_token, next_token, lemma, prev_lemma, next_lemma, 
            pos_tag, pos_category, has_affix, affix, stem_is_word, stem, gold_label  in sent]
    
    
def extract_sents_from_conll(inputfile):
    sents = []
    current_sent = []
    
    i = 0
    with open(inputfile, 'r') as my_conll:
        for line in my_conll:
            row = line.strip("\n").split('\t')
            
            if row[1] == 'bos' and i!=1:
                 sents.append(current_sent)
                 current_sent = []
            else:
                current_sent.append(tuple(row))
            i+=1
    
    return sents


def train_crf_model(X_train, y_train):

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    
    return crf

def create_crf_model(trainingfile):

    train_sents = extract_sents_from_conll(trainingfile)
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    crf = train_crf_model(X_train, y_train)
    
    return crf


def run_crf_model(crf, evaluationfile):

    test_sents = extract_sents_from_conll(evaluationfile)
    X_test = [sent2features(s) for s in test_sents]
    y_pred = crf.predict(X_test)
    
    return y_pred, X_test

def write_out_evaluation(eval_data, pred_labels, outputfile):

    outfile = open(outputfile, 'w')
    
    for evalsents, predsents in zip(eval_data, pred_labels):
        for data, pred in zip(evalsents, predsents):
            outfile.write(data.get('token') + "\t" + pred + "\n")

def train_and_run_crf_model(trainingfile, evaluationfile, outputfile):

    crf = create_crf_model(trainingfile)
    pred_labels, eval_data = run_crf_model(crf, evaluationfile)
    write_out_evaluation(eval_data, pred_labels, outputfile)
    
    
def print_confusion_matrix(predictions, goldlabels):
    '''
    Function that prints out a confusion matrix
    
    :param predictions: predicted labels
    :param goldlabels: gold standard labels
    :type predictions, goldlabels: list of strings
    '''
    
    #based on example from https://datatofish.com/confusion-matrix-python/ 
    data = {'Gold':    goldlabels, 'Predicted': predictions    }
    df = pd.DataFrame(data, columns=['Gold','Predicted'])

    confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
    print (confusion_matrix)

  

def print_precision_recall_fscore(predictions, goldlabels):
    '''
    Function that prints out precision, recall and f-score
    
    :param predictions: predicted output by classifier
    :param goldlabels: original gold labels
    :type predictions, goldlabels: list of strings
    '''
    
    precision = metrics.precision_score(y_true=goldlabels,
                        y_pred=predictions,
                        average='macro')

    recall = metrics.recall_score(y_true=goldlabels,
                     y_pred=predictions,
                     average='macro')


    fscore = metrics.f1_score(y_true=goldlabels,
                 y_pred=predictions,
                 average='macro')

    print('P:', precision, 'R:', recall, 'F1:', fscore)
    
def crf_annotation(inputfile,annotationcolumn):
    conll_input = pd.read_csv(inputfile, sep='\t', quoting=csv.QUOTE_NONE)
    annotations = conll_input[annotationcolumn].tolist()
    return annotations
    
    
def main() -> None:
   
    paths = sys.argv[1:]

    if not paths:
        paths = ['../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2_features.txt',
                 '../data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2_features.txt', 
                 '../data/SEM-2012-SharedTask-CD-SCO-evaluation-simple.v2_features.txt']
        
    trainingfile = paths[0]
    evaluationfile =paths[1]
    outputfile = paths[2]
    
    train_and_run_crf_model(trainingfile, evaluationfile, outputfile)
    
    goldlabels = crf_annotation(evaluationfile, 'gold_label')
    predictions = crf_annotation(outputfile, 'gold_label')
      
    print_confusion_matrix(predictions, goldlabels)
    print_precision_recall_fscore(predictions, goldlabels)
    
if __name__ == '__main__':
    main()

