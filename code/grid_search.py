from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

selected_features = ['token', 'lemma', 'pos_tag', 'prev_token', 'next_token', 'pos_category', 
                     'prev_lemma', 'next_lemma', 'has_affix', 'stem_is_word']


def extract_features_and_labels(trainingfile, selected_features):
  
    data = []
    targets = []
    
    #mapping features to matching columns
    feature_to_index = {'token': 0, 'lemma': 1, 'pos_tag': 2, 'prev_token': 3, 'next_token': 4, 'pos_category': 5, 'prev_lemma': 6, 'next_lemma': 7, 'has_affix':8, 'stem_is_word':9}
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for i,line in enumerate(infile):
            if i == 0:
                pass
            else:
                components = line.rstrip('\n').split()
                #checks if the row exists
                if len(components) > 0:   
                    feature_dict = {}
                    for feature_name in selected_features:
                        components_index = feature_to_index.get(feature_name)
                        feature_dict[feature_name] = components[components_index]
                    data.append(feature_dict)
                    
                    # the gold label is in the last column
                    targets.append(components[-1])
    return data, targets


def create_classifier(train_features, 
                      train_targets, 
                      test__features, 
                      test__targets):
                      
    # Create a pipeline with Count Vectorizer and SVC
    
    pipe_cvec_svc = Pipeline([
        ('dictvec', DictVectorizer),
        ('svc', SVC(random_state=42))
    ])

    # Search over the following values of hyperparameters:
    pipe_cvec_svc_params = {
        'dictvec__max_features': [500], 
        'dictvec__min_df': [2,3], 
        'dictvec__max_df': [.9,.95],    
        'svc__kernel': ['linear'],
        'svc__C': [.1]
    }


    gs_cvec_svc = GridSearchCV(pipe_cvec_svc, 
                              param_grid = pipe_cvec_svc_params, 
                              cv=10) # 10-fold cross validation

    # Fit model on to training data
    gs_cvec_svc.fit(train_features,train_targets)

    # Generate predictions on validation set
    cvec_svc_pred = gs_cvec_svc.predict(test_features)

    # Print best parameters
    print('Best parameters: ', gs_cvec_svc.best_params_)
    print('Best CV score: ', gs_cvec_svc.best_score_)
    print('Training score:', gs_cvec_svc.score(train_features, train_targets))
    print('Validation score:', gs_cvec_svc.score(test__features, test__targets))
    print('')
    
    return cvec_svc_pred


def print_precision_recall_fscore(predictions, goldlabels):
    '''
    Function that prints out precision, recall and f-score in a complete report

    :param predictions: predicted output by classifier
    :param goldlabels: original gold labels
    :type predictions, goldlabels: list of strings
    '''

    report = classification_report(goldlabels,predictions,digits = 3)

    print('METRICS: ')
    print()
    print(report)
    
    
trainfile = '../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2_features.txt'
testfile = '../data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2_features.txt'

train_features, train_targets = extract_features_and_labels(trainfile,selected_features )
test__features, test__targets = extract_features_and_labels(testfile,selected_features )

prediction = create_classifier(train_features, 
                                  train_targets, 
                                  test__features, 
                                  test__targets)

print_precision_recall_fscore(prediction, test__targets)


