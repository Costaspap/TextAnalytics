########################################################################################################
##################################################### Imports ##########################################
########################################################################################################

import gensim 
import re
import os
import string
from nltk import sent_tokenize, word_tokenize
from IPython.display import clear_output
from collections import Counter
import gc
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


########################################################################################################
##################################### Declarations #####################################################
########################################################################################################


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



def Tune_Threshold(estimator, validation_data, y_true):
    plt.figure()
    plt.xlabel('Thresholds')
    plt.ylabel('Score')
    prec, rec, thres = [],[],[]
    
    probs = estimator.predict_proba(validation_data).T[1]
        
    for thr in range(1,10,1):
    
        y_pred = probs.copy()
        
        y_pred[y_pred > thr/10] = 1
        
        y_pred[y_pred <= thr/10] = 0
        
        prec.append(precision_score(y_true = y_true, y_pred = y_pred))
        rec.append(recall_score(y_true = y_true, y_pred = y_pred)) 
        thres.append(thr/10)
        
    plt.xlim(0,1)
    plt.scatter(thres, prec)
    plt.plot(thres, prec, label = 'Precision')

    plt.scatter(thres, rec)
    plt.plot(thres, rec, label = 'Recall')  
    
    plt.legend()
    plt.title('Precision and Recall by Threshold')
    plt.show() 
    
    
def Custom_Predict(estimator, data, threshold = 0.5):
    
    probs = estimator.predict_proba(data).T[1]
    
    y_pred = probs.copy()
        
    y_pred[y_pred > threshold] = 1
        
    y_pred[y_pred <= threshold] = 0
    
    return y_pred


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def pretty_percentage(amount):
    
    print(str(np.round(amount*100,2)) + '%')

def clean_text(text):
    """ 
    1. Remove html like text from europarl e.g. <Chapter 1>
    2. Remove line breaks
    3. Reduce all whitespaces to 1
    4. turn everything to lower case
    """


    regex = re.compile('[%s]' % re.escape(string.punctuation))
    text = text.lower()  # Turn everything to lower case
    text = regex.sub(' ', text).strip()
    out = re.sub(' +', ' ', text)  # Reduce whitespace down to one
    
    return out

def Score_Classifier(y_train_pred, y_train_true, y_pred, y_true, labels = ['Negative','Positive'], Classifier_Title = None
                     , evaluation_type = 'Validation'):
    
    cm = confusion_matrix(y_true = y_true, y_pred = y_pred)
    
    print()
    if Classifier_Title is not None:
        print('************************',Classifier_Title,'************************')
    print('--------------------------------------------------------------')
    print('Training Accuracy:')
    pretty_percentage(accuracy_score(y_pred = y_train_pred, y_true = y_train_true))
    print('--------------------------------------------------------------')
    print(evaluation_type +' Accuracy:')
    pretty_percentage(accuracy_score(y_pred = y_pred, y_true = y_true))
    print('--------------------------------------------------------------')
    print(evaluation_type +' Precision:')
    pretty_percentage(precision_score(y_pred = y_pred, y_true = y_true))
    print('--------------------------------------------------------------')
    print(evaluation_type +' Recall:')
    pretty_percentage(recall_score(y_pred = y_pred, y_true = y_true))
    print('--------------------------------------------------------------')
    print(evaluation_type +' F1 Score:')
    pretty_percentage(f1_score(y_pred = y_pred, y_true = y_true))
    print('--------------------------------------------------------------')
    print(evaluation_type +' Confusion Matrix:')
    print_cm(cm, labels)
    print('--------------------------------------------------------------')
    

########################################################################################################
##################################### Preprocessing ####################################################
########################################################################################################


path = 'E:\\Desktop\\Python_Projects\\Project_2\\'

data = []
labels = []

for folder in ['pos','neg']:
    
    count = 0
    full_path = path + folder + '\\'
    total = len(os.listdir(full_path))
    
    for file in os.listdir(full_path):
        f = open(full_path + file, 'r', encoding="utf-8")
        file_text = f.read()
        f.close()
        
        clean = clean_text(file_text)

        data.append(clean)            
        
        if folder == 'pos':
            labels.append(1)
        else:
            labels.append(0)

        count += 1
        
        print('File ' + file + ' finished. Completed ' + str(round(count * 100 / total, 2)) + '%')
       
        
with open('data', 'wb') as f:
    pickle.dump(data, f)
    
with open('labels', 'wb') as f:
    pickle.dump(labels, f)
    
    
"""
with open('data', 'rb') as f:
        data = pickle.load(f)
        
with open('labels', 'rb') as f:
        labels = pickle.load(f)
""" 

########################################################################################################
######################################## Split into Sets  ##############################################
########################################################################################################   

        
X_Train, X_Tune, Y_Train, Y_Tune = train_test_split(data, 
                 labels,        
                 test_size=0.10,
                 random_state = 42)
        
        
X_Validation, X_Test, Y_Validation, Y_Test = train_test_split(X_Tune, 
                 Y_Tune,        
                 test_size=0.5,
                 random_state = 42)
        
             
del X_Tune, Y_Tune
gc.collect()


########################################################################################################
########################################## Create Grams ################################################
########################################################################################################   

Unigram_vectorizer = TfidfVectorizer(ngram_range = (1, 1), 
                                     min_df = 10,
                                     analyzer = 'word',
                                     stop_words  = stopwords.words('english'))


Bigram_vectorizer = TfidfVectorizer(ngram_range = (2,2),
                                    min_df = 10,
                                    analyzer = 'word')

Uni_X_Train = Unigram_vectorizer.fit_transform(X_Train)
Bi_X_Train = Bigram_vectorizer.fit_transform(X_Train)

Uni_X_Validation = Unigram_vectorizer.transform(X_Validation)
Bi_X_Validation = Bigram_vectorizer.transform(X_Validation)

Uni_X_Test = Unigram_vectorizer.transform(X_Test)
Bi_X_Test = Bigram_vectorizer.transform(X_Test)


print('Unigram Training Dataset:',Uni_X_Train.shape) 
print('Unigram Validation Dataset:',Uni_X_Validation.shape) 
print('Unigram Test Dataset:',Uni_X_Test.shape) 

print('Bigram Training Dataset:',Bi_X_Train.shape) 
print('Bigram Validation Dataset:',Bi_X_Validation.shape) 
print('Bigram Test Dataset:',Bi_X_Test.shape) 


########################################################################################################
######################## Perform Truncated SVD to reduce dimensions ####################################
######################################################################################################## 
from sklearn.decomposition import TruncatedSVD

print('Initial Shape of Unigram Data:',Uni_X_Train.shape)
print('Initial Shape of Bigram Data:',Bi_X_Train.shape)

tsvd1 = TruncatedSVD(n_components=8000, random_state=42)
tsvd2 = TruncatedSVD(n_components=10000, random_state=42)

svd_model_uni = tsvd1.fit(Uni_X_Train)

svd_model_bi = tsvd2.fit(Bi_X_Train)

uni_ex_var = np.sum(svd_model_uni.explained_variance_ratio_)
bi_ex_var = np.sum(svd_model_bi.explained_variance_ratio_)

print('Explained Variance of Unigram model: ')
pretty_percentage(uni_ex_var)

print('Explained Variance of Bigram model: ')
pretty_percentage(bi_ex_var)

reduced_uni_train = svd_model_uni.transform(Uni_X_Train)
reduced_uni_validation = svd_model_uni.transform(Uni_X_Validation)
reduced_uni_test = svd_model_uni.transform(Uni_X_Test)

reduced_bi_train = svd_model_bi.transform(Bi_X_Train)
reduced_bi_validation = svd_model_bi.transform(Bi_X_Validation)
reduced_bi_test = svd_model_bi.transform(Bi_X_Test)


print('Reduced Shape of Unigram Data:',reduced_uni_train.shape)
print('Reduced Shape of Bigram Data:',reduced_bi_train.shape)


with open('reduced_uni_train', 'wb') as f:
    pickle.dump(reduced_uni_train, f)

with open('reduced_uni_validation', 'wb') as f:
    pickle.dump(reduced_uni_validation, f)
    
with open('reduced_uni_test', 'wb') as f:
    pickle.dump(reduced_uni_test, f)

with open('reduced_bi_train', 'wb') as f:
    pickle.dump(reduced_bi_train, f)

with open('reduced_bi_validation', 'wb') as f:
    pickle.dump(reduced_bi_validation, f)
    
with open('reduced_bi_test', 'wb') as f:
    pickle.dump(reduced_bi_test, f)

with open('Y_Train', 'wb') as f:
    pickle.dump(Y_Train, f)

with open('Y_Validation', 'wb') as f:
    pickle.dump(Y_Validation, f)
    
with open('Y_Test', 'wb') as f:
    pickle.dump(Y_Test, f)

    
"""
with open('reduced_uni_train', 'rb') as f:
        reduced_uni_train = pickle.load(f)
        
with open('reduced_uni_validation', 'rb') as f:
        reduced_uni_validation = pickle.load(f)
        
with open('reduced_uni_test', 'rb') as f:
        reduced_uni_test = pickle.load(f)
        
with open('reduced_bi_train', 'rb') as f:
        reduced_bi_train = pickle.load(f)
        
with open('reduced_bi_validation', 'rb') as f:
        reduced_bi_validation = pickle.load(f)
        
with open('reduced_bi_test', 'rb') as f:
        reduced_bi_test = pickle.load(f)
        
with open('Y_Train', 'rb') as f:
    Y_Train = pickle.load(f)
    
with open('Y_Validation', 'rb') as f:
        Y_Validation = pickle.load(f)
        
with open('Y_Test', 'rb') as f:
        Y_Test = pickle.load(f)
""" 


########################################################################################################
###################################### Modelling #######################################################
######################################################################################################## 
  
    
#######################################################
##                Baseline Classifier                ##
#######################################################

Class_Freq_Train = pd.Series(Y_Train).value_counts()
print(Class_Freq_Train)

baseline_train = np.zeros((len(Y_Train)))

baseline_valid = np.zeros((len(Y_Validation)))

Score_Classifier(baseline_train, Y_Train, baseline_valid, Y_Validation, Classifier_Title = 'Baseline')


#######################################################
##                Naive Bayes Classifier             ##
#######################################################

#############################   For Unigrams    ###############################################################

from sklearn.naive_bayes import MultinomialNB

## Tune alpha ##
for alpha in range(1,10):
    
    NB_Uni_clf = MultinomialNB(alpha = alpha/10)
    
    NB_Uni_clf.fit(reduced_uni_train + np.abs(np.min(reduced_uni_train)), Y_Train)    
    
    NB_Pred_Valid = NB_Uni_clf.predict(reduced_uni_validation + np.abs(np.min(reduced_uni_validation)))

    NB_Pred_Train = NB_Uni_clf.predict(reduced_uni_train + np.abs(np.min(reduced_uni_train)))

    
    Score_Classifier(NB_Pred_Train, Y_Train, NB_Pred_Valid, Y_Validation, Classifier_Title = 'Naive Bayes on Unigrams')
  
    
## Tune threshold ##
NB_Uni_clf = MultinomialNB(alpha = 1)
    
NB_Uni_clf.fit(reduced_uni_train + np.abs(np.min(reduced_uni_train)), Y_Train)       


Tune_Threshold(NB_Uni_clf, reduced_uni_validation + np.abs(np.min(reduced_uni_validation)), Y_Validation)

    
for thr in range(490,500):
    
    preds = Custom_Predict(NB_Uni_clf, reduced_uni_validation + np.abs(np.min(reduced_uni_validation)), thr/1000)
    
    NB_Pred_Train = Custom_Predict(NB_Uni_clf, reduced_uni_train + np.abs(np.min(reduced_uni_train)), thr/1000)
    print('Threshold: ', thr/1000)
    Score_Classifier(NB_Pred_Train, Y_Train, preds, Y_Validation, Classifier_Title = 'Naive Bayes on Unigrams')
    
    

#############################   For Bigrams    ###############################################################



## Tune alpha ##
for alpha in range(1,10):
    
    NB_Βi_clf = MultinomialNB(alpha = alpha/10)
    
    NB_Βi_clf.fit(reduced_bi_train + np.abs(np.min(reduced_bi_train)), Y_Train)    
    
    NB_Pred_Valid = NB_Βi_clf.predict(reduced_bi_validation + np.abs(np.min(reduced_bi_validation)))

    NB_Pred_Train = NB_Βi_clf.predict(reduced_bi_train + np.abs(np.min(reduced_bi_train)))

    
    Score_Classifier(NB_Pred_Train, Y_Train, NB_Pred_Valid, Y_Validation, Classifier_Title = 'Naive Bayes on Bigrams')


## Tune threshold ##
NB_Βi_clf = MultinomialNB(alpha = 1)

NB_Βi_clf.fit(reduced_bi_train + np.abs(np.min(reduced_bi_train)), Y_Train) 


Tune_Threshold(NB_Βi_clf, reduced_bi_validation + np.abs(np.min(reduced_bi_validation)), Y_Validation)

    
for thr in range(4970,4980):
    
    preds = Custom_Predict(NB_Βi_clf, reduced_bi_validation + np.abs(np.min(reduced_bi_validation)), thr/10000)
    
    NB_Pred_Train = Custom_Predict(NB_Βi_clf, reduced_bi_train + np.abs(np.min(reduced_bi_train)), thr/10000)
    print('Threshold: ', thr/10000)
    Score_Classifier(NB_Pred_Train, Y_Train, preds, Y_Validation, Classifier_Title = 'Naive Bayes on Bigrams')
    
    


#######################################################
##                 Logistic Regression               ##
#######################################################

from sklearn.linear_model import LogisticRegression

#############################   For Unigrams    ###############################################################


## Tune C ##
for c in range(1,10):
    
    Logistic_Uni_clf = LogisticRegression(C = c/10)
    
    Logistic_Uni_clf.fit(reduced_uni_train, Y_Train)    
    
    LR_Pred_Valid = Logistic_Uni_clf.predict(reduced_uni_validation)

    LR_Pred_Train = Logistic_Uni_clf.predict(reduced_uni_train)

    
    Score_Classifier(LR_Pred_Train, Y_Train, LR_Pred_Valid, Y_Validation, Classifier_Title = 'Logistic Regression on Unigrams')

    
## Tune threshold ##
Logistic_Uni_clf = LogisticRegression(C = 0.9, penalty = 'l1')
    
Logistic_Uni_clf.fit(reduced_uni_train, Y_Train)       


Tune_Threshold(Logistic_Uni_clf, reduced_uni_validation, Y_Validation)

    
#############################   For Bigrams    ###############################################################

## Tune C ##
for c in range(1,10):
    
    Logistic_Bi_clf = LogisticRegression(C = c/10)
    
    Logistic_Bi_clf.fit(reduced_bi_train, Y_Train)    
    
    LR_Pred_Valid = Logistic_Bi_clf.predict(reduced_bi_validation)

    LR_Pred_Train = Logistic_Bi_clf.predict(reduced_bi_train)

    
    Score_Classifier(LR_Pred_Train, Y_Train, LR_Pred_Valid, Y_Validation, Classifier_Title = 'Logistic Regression on Bigrams')

    
## Tune threshold ##
Logistic_Bi_clf = LogisticRegression(C = 0.9, penalty = 'l1')
    
Logistic_Bi_clf.fit(reduced_bi_train, Y_Train)       


Tune_Threshold(Logistic_Bi_clf, reduced_bi_validation, Y_Validation)



## Testing ##

LR_Pred_test = Logistic_Uni_clf.predict(reduced_uni_test)

LR_Pred_Train = Logistic_Uni_clf.predict(reduced_uni_train)

Score_Classifier(LR_Pred_Train, Y_Train, LR_Pred_test, Y_Test, 
                 Classifier_Title = 'Logistic Regression test on Unigrams', evaluation_type = 'Test')

LR_Pred_test = Logistic_Bi_clf.predict(reduced_bi_test)

LR_Pred_Train = Logistic_Bi_clf.predict(reduced_bi_train)

Score_Classifier(LR_Pred_Train, Y_Train, LR_Pred_test, Y_Test, 
                 Classifier_Title = 'Logistic Regression test on Bigrams', evaluation_type = 'Test')



















