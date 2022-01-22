from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.metrics import classification_report
import sys
import os

selected_features = ['token', 'lemma', 'pos_tag', 'prev_token', 'next_token', 'pos_category', 
                     'prev_lemma', 'next_lemma', 'has_affix', 'stem_is_word']

# get inspired by machien learning course
def extract_features_and_labels(trainingfile, selected_features):

    data = []
    targets = []
    
    #mapping features to matching columns
    feature_to_index = {'token': 0, 'lemma': 1, 'pos_tag': 2, 'prev_token': 3, 'next_token': 4, 'pos_category': 5,'prev_lemma':6, 'next_lemma': 7, 'has_affix':8, 'stem_is_word':9}
        
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for i,line in enumerate(infile):
            if i == 0:
                pass
            else:
                components = line.rstrip('\n').split()
                #make sure the row is not empty 
                if len(components) > 0:   
                    feature_dict = {}
                    for feature_name in selected_features:
                        components_index = feature_to_index.get(feature_name)
                        feature_dict[feature_name] = components[components_index]
                    data.append(feature_dict)
                    
                    # the gold label is in the last column
                    targets.append(components[-1])
    return data, targets


def create_classifier(train_features, train_targets, modelname = 'SVM'):


    if modelname ==  'logreg':
        vec = DictVectorizer()
 
        tokens_vectorized = vec.fit_transform(train_features)
        classifier = LogisticRegression(solver='saga')
        classifier.fit(tokens_vectorized, train_targets)
        
    elif modelname ==  'NB':
 
        classifier = MultinomialNB()
        vec = DictVectorizer()
        tokens_vectorized = vec.fit_transform(train_features)
        classifier.fit(tokens_vectorized, train_targets)
    
    elif modelname ==  'SVM':
       
        classifier = svm.LinearSVC(max_iter=2000)
        vec = DictVectorizer()
        tokens_vectorized = vec.fit_transform(train_features)
        classifier.fit(tokens_vectorized, train_targets)
        
    return classifier, vec



def get_predicted_and_gold_labels(testfile, vectorizer, classifier, selected_features):
  

    # we use the same function as above (guarantees features have the same name and form)
    features, goldlabels = extract_features_and_labels(testfile, selected_features)
    
    # we need to use the same fitting as before, so now we only transform the current features according to this mapping (using only transform)
    test_features_vectorized = vectorizer.transform(features)
    predictions = classifier.predict(test_features_vectorized)

    return predictions, goldlabels




def print_confusion_matrix(predictions, goldlabels):
    

    # based on example from https://datatofish.com/confusion-matrix-python/
    data = {'Gold': goldlabels, 'Predicted': predictions}
    df = pd.DataFrame(data, columns=['Gold', 'Predicted'])

    confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
    print(confusion_matrix)
    return confusion_matrix


def print_precision_recall_fscore(predictions, goldlabels):
    

    report = classification_report(goldlabels,predictions,digits = 3)

    print('Evaluation Metrics: ')
    print()
    print(report)


def run_classifier(trainfile, testfile, selected_features):


    modelnames = ['SVM', 'NB', 'logreg']
    
    feature_values, labels = extract_features_and_labels(trainfile, selected_features)
    
    for modelname in modelnames:
        classifier, vectorizer = create_classifier(feature_values, labels,modelname=modelname)
        predictions, goldlabels = get_predicted_and_gold_labels(testfile, vectorizer, classifier, selected_features)
        print()
        print('---->'+ modelname + ' with ' + ' , '.join(selected_features) + ' as features <----')
        print_precision_recall_fscore(predictions, goldlabels)
        print('------')
    
    return predictions

def main() -> None:
    
    paths = sys.argv[1:]

    if not paths:
        paths = ['../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2_features.txt',
                 '../data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2_features.txt']

    
    print(f'evaluating {os.path.basename(paths[0])} and {os.path.basename(paths[1])}')
    run_classifier(paths[0], paths[1], selected_features)


if __name__ == '__main__':
    main()