########################################################################################################
##################################################### Imports ##########################################
########################################################################################################


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

from pprint import pprint
import time
import sys
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


########################################################################################################
##################################### Preprocessing ####################################################
########################################################################################################




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

wordCounts = dict()

data = pd.DataFrame(columns=['Review', 'Class'])

path = 'E:\\Desktop\\Python_Projects\\'

vocabulary = set()

for folder in ['pos','neg']:
    
    count = 0
    full_path = path + folder + '\\'
    total = len(os.listdir(full_path))
    
    for file in os.listdir(full_path):
        f = open(full_path + file, 'r', encoding="utf-8")
        file_text = f.read()
        f.close()
        
        words = word_tokenize(clean_text(file_text))

        
        review = []
        
               
        for word in words:
            if word not in stopwords.words('english'):
                review.append(word)
                
                try:
                    wordCounts[word] += 1
                except:
                    wordCounts[word] = 1
        
        review = ' '.join(review)
            
        
        if folder == 'pos':
            data = data.append({'Review':review,'Class':1}, ignore_index = True) 
        else:
            data = data.append({'Review':review,'Class':0}, ignore_index = True) 

        count += 1
        
        print('File ' + file + ' finished. Completed ' + str(round(count * 100 / total, 2)) + '%')
       
        
with open('data', 'wb') as f:
    pickle.dump(data, f)
    
    
"""
with open('data', 'rb') as f:
        data = pickle.load(f)
""" 
########################################################################################################
##################### Reduce Dimensions by removing rare words #########################################
########################################################################################################   

clean_data = pd.DataFrame(columns=['Review', 'Class'])

valid = {k:v for k, v in wordCounts.items() if v > 15}

for i in range(0,data.shape[0]):
    
    words = word_tokenize(data.iloc[i,0])
    
    out = [word for word in words if word in valid]
    
    out = ' '.join(out)
    
    clean_data = clean_data.append({'Review':out,'Class':data.iloc[i,1]}, ignore_index = True) 


with open('clean_data', 'wb') as f:
    pickle.dump(clean_data, f)
    
    
"""
with open('clean_data', 'rb') as f:
        clean_data = pickle.load(f)
""" 


########################################################################################################
##################################### Split into Sets ##################################################
########################################################################################################   


        
        
X_Train, X_Tune, Y_Train, Y_Tune = train_test_split(data['Review'].values, 
                 data['Class'].values,        
                 test_size=0.70)
        
        
        
          
X_Validation, X_Test, Y_Validation, Y_Test = train_test_split(X_Tune, 
                 Y_Tune,        
                 test_size=0.5)
        
             
del X_Tune, Y_Tune
gc.collect()
        

########################################################################################################
############################ Create Matrix of Word Counts ##############################################
######################################################################################################## 


count_V = CountVectorizer(tokenizer=word_tokenize)
tok_train = count_V.fit_transform(X_Train)
tok_validation = count_V.transform(X_Validation)
tok_test = count_V.transform(X_Test)

Y_Train = Y_Train.astype('int')
Y_Validation = Y_Validation.astype('int')
Y_Test = Y_Test.astype('int')


########################################################################################################
###################################### Modelling #######################################################
######################################################################################################## 
from sklearn.metrics import confusion_matrix, accuracy_score


##### Logistic Regression #####

from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression(C=0.3, dual=True,max_iter = 1000)


log_reg.fit(tok_train, Y_Train)

y_pred_log_reg = log_reg.predict(tok_validation)

confusion_matrix(y_true = Y_Validation, y_pred = y_pred_log_reg)
accuracy_score(y_true = Y_Validation, y_pred = y_pred_log_reg)



####### SVM ###########


from sklearn.svm import SVC



SVM = SVC(kernel = 'linear',C=1)


SVM.fit(tok_train, Y_Train)

y_pred_SVM = SVM.predict(tok_validation)

confusion_matrix(y_true = Y_Validation, y_pred = y_pred_SVM)
accuracy_score(y_true = Y_Validation, y_pred = y_pred_SVM)



####### Multilayer Perceptron ###########


from sklearn.neural_network import MLPClassifier



NN = MLPClassifier((100,100,100,50), alpha = 0.01,verbose = True, max_iter = 60)


NN.fit(tok_train, Y_Train)

y_pred_NN = NN.predict(tok_validation)

confusion_matrix(y_true = Y_Validation, y_pred = y_pred_NN)
accuracy_score(y_true = Y_Validation, y_pred = y_pred_NN)



def ensemble_predict(data):
    
    pred1 = log_reg.predict(data)
    
    pred2 = SVM.predict(data)

    pred3 = NN.predict(data)
    
    full_pred = pred1 + pred2 + pred3
    
    full_pred[np.where(full_pred < 2)] = 0
    
    full_pred[np.where(full_pred >=2)] = 1
    
    return full_pred


y_pred_ensemble = ensemble_predict(tok_validation)


confusion_matrix(y_true = Y_Validation, y_pred = y_pred_ensemble)
accuracy_score(y_true = Y_Validation, y_pred = y_pred_ensemble)




