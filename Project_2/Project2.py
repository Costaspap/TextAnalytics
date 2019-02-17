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

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords


########################################################################################################
##################################### Preprocessing ####################################################
########################################################################################################


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
########################################## Create Grams ################################################
########################################################################################################   

Unigram_vectorizer = CountVectorizer(ngram_range = (1, 1), 
                                     min_df = 15,
                                     analyzer = 'word',
                                     stop_words  = stopwords.words('english'))


Bigram_vectorizer = CountVectorizer(ngram_range = (2,2),
                                    min_df = 15,
                                    analyzer = 'word')

Unigram_vectorizer.fit(data)
Bigram_vectorizer.fit(data)

uni_data = Unigram_vectorizer.transform(data)
bi_data = Bigram_vectorizer.transform(data)




########################################################################################################
################################# Split into Sets for Unigrams #########################################
########################################################################################################   

        
X_Uni_Train, X_Uni_Tune, Y_Uni_Train, Y_Uni_Tune = train_test_split(uni_data, 
                 labels,        
                 test_size=0.20,
                 random_state = 42)
        
        
X_Uni_Validation, X_Uni_Test, Y_Uni_Validation, Y_Uni_Test = train_test_split(X_Uni_Tune, 
                 Y_Uni_Tune,        
                 test_size=0.5,
                 random_state = 42)
        
             
del X_Uni_Tune, Y_Uni_Tune
gc.collect()
      

########################################################################################################
################################# Split into Sets for BiGrams  #########################################
########################################################################################################   

        
X_Bi_Train, X_Bi_Tune, Y_Bi_Train, Y_Bi_Tune = train_test_split(bi_data, 
                 labels,        
                 test_size=0.20,
                 random_state = 42)
        
        
X_Bi_Validation, X_Bi_Test, Y_Bi_Validation, Y_Bi_Test = train_test_split(X_Bi_Tune, 
                 Y_Bi_Tune,        
                 test_size=0.5,
                 random_state = 42)
        
             
del X_Bi_Tune, Y_Bi_Tune
gc.collect()
        

print('Unigram Training Dataset:',X_Uni_Train.shape)
print('Unigram Validation Dataset:',X_Uni_Validation.shape)
print('Unigram Test Dataset:',X_Uni_Test.shape)

print('Bigram Training Dataset:',X_Bi_Train.shape) 
print('Bigram Validation Dataset:',X_Bi_Validation.shape) 
print('Bigram Test Dataset:',X_Bi_Test.shape) 


########################################################################################################
######################## Perform Truncated SVD to reduce dimensions ####################################
######################################################################################################## 
from sklearn.decomposition import TruncatedSVD

print('Initial Shape of Unigram Data:',X_Uni_Train.shape)
print('Initial Shape of Bigram Data:',X_Bi_Train.shape)

tsvd1 = TruncatedSVD(n_components=4000, random_state=42)
tsvd2 = TruncatedSVD(n_components=8000, random_state=42)

svd_model_uni = tsvd1.fit(X_Uni_Train)

svd_model_bi = tsvd2.fit(X_Bi_Train)

uni_ex_var = np.sum(svd_model_uni.explained_variance_ratio_)
bi_ex_var = np.sum(svd_model_bi.explained_variance_ratio_)

print('Explained Variance of Unigram model: ', pretty_percentage(uni_ex_var))
print('Explained Variance of Bigram model: ', pretty_percentage(bi_ex_var))

reduced_uni_train = svd_model_uni.transform(X_Uni_Train)
reduced_uni_validation = svd_model_uni.transform(X_Uni_Validation)
reduced_uni_test = svd_model_uni.transform(X_Uni_Test)

reduced_bi_train = svd_model_bi.transform(X_Bi_Train)
reduced_bi_validation = svd_model_bi.transform(X_Bi_Validation)
reduced_bi_test = svd_model_bi.transform(X_Bi_Test)


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

with open('Y_Uni_Train', 'wb') as f:
    pickle.dump(Y_Uni_Train, f)

with open('Y_Uni_Validation', 'wb') as f:
    pickle.dump(Y_Uni_Validation, f)
    
with open('Y_Uni_Test', 'wb') as f:
    pickle.dump(Y_Uni_Test, f)

with open('Y_Bi_Train', 'wb') as f:
    pickle.dump(Y_Bi_Train, f)

with open('Y_Bi_Validation', 'wb') as f:
    pickle.dump(Y_Bi_Validation, f)
    
with open('Y_Bi_Test', 'wb') as f:
    pickle.dump(Y_Bi_Test, f)
    
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
        
with open('Y_Uni_Train', 'rb') as f:
    Y_Uni_Train = pickle.load(f)

with open('Y_Uni_Validation', 'rb') as f:
        Y_Uni_Validation = pickle.load(f)
        
with open('Y_Uni_Test', 'rb') as f:
        Y_Uni_Test = pickle.load(f)
        
with open('Y_Bi_Train', 'rb') as f:
    Y_Bi_Train = pickle.load(f)

with open('Y_Bi_Validation', 'rb') as f:
        Y_Bi_Validation = pickle.load(f)
        
with open('Y_Bi_Test', 'rb') as f:
        Y_Bi_Test = pickle.load(f)
""" 

########################################################################################################
###################################### Plot Learnng Curve ##############################################
######################################################################################################## 


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

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




########################################################################################################
###################################### Modelling #######################################################
######################################################################################################## 
  

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


####### KNN ###########

#############################   For Unigrams    ###############################################################

from sklearn.neighbors import KNeighborsClassifier

KNN_Uni_clf = KNeighborsClassifier(n_jobs = 6)

KNN_Uni_clf.fit(reduced_uni_train, Y_Uni_Train)

KNN_Pred = KNN_Uni_clf.predict(reduced_uni_validation)

pretty_percentage(accuracy_score(y_true = Y_Uni_Validation, y_pred = KNN_Pred))

#############################   For Bigrams    ###############################################################

KNN_Βi_clf = KNeighborsClassifier(n_jobs = 6)

KNN_Βi_clf.fit(reduced_bi_train, Y_Bi_Train)

KNN_Bi_Pred = KNN_Βi_clf.predict(reduced_bi_validation)

accuracy_score(y_true = Y_Bi_Validation, y_pred = KNN_Bi_Pred)




####### Logistic Regression Baseline ###########

from sklearn.linear_model import LogisticRegression


## Initialize and train models ##
Logistic_Uni_clf = LogisticRegression(penalty = 'l1', C = 0.7)

Logistic_Bi_clf = LogisticRegression(penalty = 'l1', C = 0.7)


Logistic_Uni_clf.fit(reduced_uni_train, Y_Uni_Train)

Logistic_Bi_clf.fit(reduced_bi_train, Y_Bi_Train)


Logistic_Uni_Pred = Logistic_Uni_clf.predict(reduced_uni_validation)
Logistic_Bi_Pred = Logistic_Bi_clf.predict(reduced_bi_validation)

print('Unigram baseline model accuracy:', 
      pretty_percentage(accuracy_score(y_true = Y_Uni_Validation, y_pred = Logistic_Uni_Pred)))

print('Bigram baseline model accuracy:', 
      pretty_percentage(accuracy_score(y_true = Y_Bi_Validation, y_pred = Logistic_Bi_Pred)))



#############################   For Unigrams    ###############################################################

##### Baseline Regularization Results ######

Logistic_Uni_clf = LogisticRegression(penalty = 'l1', C = 0.7)

plot_learning_curve(Logistic_Uni_clf,'Logistic Regression with inverse regularization C = ' + str(round(0.7,2)),
    reduced_uni_train, Y_Uni_Train, ylim=[.6,1], cv=2, n_jobs = 7, train_sizes=np.linspace(.1, 1.0, 5))

##### Tuning Regularization ######

plt.figure()

for c in np.arange(1.0,0.0,-0.1):
    Logistic_clf = LogisticRegression(penalty = 'l1', C = round(c,2))

    plot_learning_curve(Logistic_clf,'Logistic Regression with inverse regularization C = ' + str(round(c,2)),
    reduced_uni_train, Y_Uni_Train, ylim=[.6,1], cv=2, n_jobs = 7, train_sizes=np.linspace(.1, 1.0, 5))
plt.show()


##### Tuning Threshold ######

Logistic_Uni_clf = LogisticRegression(penalty = 'l1', C = 0.1)

Logistic_Uni_clf.fit(reduced_uni_train, Y_Uni_Train)

Logistic_Pred = Logistic_Uni_clf.predict(reduced_uni_validation)

accuracy_score(y_true = Y_Uni_Validation, y_pred = Logistic_Pred)


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
    plt.plot(thres, prec)
    plt.title('Precision and Recall by Threshold')
    
    plt.scatter(thres, rec)
    
    
    plt.plot(thres, rec)    
    
    plt.legend(['Precision','Recall'])
    plt.show() 


Tune_Threshold(Logistic_Uni_clf,reduced_uni_validation,Y_Uni_Validation)         
            
# Final Model

Logistic_Pred = Logistic_Uni_clf.predict(reduced_uni_test)

pretty_percentage(accuracy_score(y_true = Y_Uni_Test, y_pred = Logistic_Pred))


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

# first generate with specified labels
labels = ['Negative','Positive']
cm = confusion_matrix(y_true = Y_Uni_Test, y_pred = Logistic_Pred)


print('Accuracy:')
pretty_percentage(accuracy_score(y_true = Y_Uni_Test, y_pred = Logistic_Pred))
print()
print('Precision:')

pretty_percentage(precision_score(y_true = Y_Uni_Test, y_pred = Logistic_Pred))
print()
print('Recall:')

pretty_percentage(recall_score(y_true = Y_Uni_Test, y_pred = Logistic_Pred))
print()
print('Confusion Matrix:')
print_cm(cm, labels)


















