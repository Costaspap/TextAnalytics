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
from nltk.tokenize import RegexpTokenizer, TweetTokenizer
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
from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout, concatenate
from keras.models import Model
import pandas as pd
import keras.backend as K
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Activation
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



########################################################################################################
##################################### Declarations #####################################################
########################################################################################################

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
    

def plot_history(hs, epochs, title, upper = 0.9):
    #figure(num=None, figsize=(6, 6), dpi=100, facecolor='w', edgecolor='k')
    
    plt.close()
    plt.plot(hs.history['acc'])
    plt.plot(hs.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xticks(range(0,epochs,2))
    plt.ylim(top = upper)
    plt.xlabel('epoch')
    plt.legend(['Train Accuracy',
                 'Validation Accuracy'], loc = 4)
    plt.savefig(title + '_Acc.png')
    
    plt.close()   
    plt.plot(hs.history['loss'])
    plt.plot(hs.history['val_loss'])
    plt.xticks(range(0,epochs,2))
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train Loss','Validation Loss'], loc = 1)
    plt.savefig(title + '_Loss.png')


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


    regex = re.compile('[\.|\-|\,|\?|\_|\:|\"|\)|\(\)\/|\\|\>|\<]')
    text = text.lower()  # Turn everything to lower case
    text = regex.sub(' ', text).strip()
    out = re.sub(' +', ' ', text)  # Reduce whitespace down to one
    
    return out


########################################################################################################
##################################### Preprocessing ####################################################
########################################################################################################
tok = TweetTokenizer()

path = 'E:\\Desktop\\Python_Projects\\Project_3\\'
os.chdir(path)

raw = pd.read_csv('imdb_master.csv',encoding='iso-8859-1')

raw = raw[raw['label'] != 'unsup']

data = list(raw.review)
labels = raw.label

del raw

labels = list(labels.replace({'pos': 1, 'neg': 0}))
    

for i in range(0,len(data)):
    
    clean = tok.tokenize(clean_text(data[i]))
    
    clean = [word for word in clean if word != 'br']
    
    clean = ' '.join(clean)
    
    data[i] = clean        
    
    print('Review ' + str(i+1) + ' cleaned. Completed ' + str(round(i * 100 / len(data), 2)) + '%')
       
        
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
                 random_state = 40)
        
        
X_Validation, X_Test, Y_Validation, Y_Test = train_test_split(X_Tune, 
                 Y_Tune,        
                 test_size=0.5,
                 random_state = 40)
      
print('Training set size:', len(X_Train))

print('Validation set size:', len(X_Validation))

print('Test set size:', len(X_Test))  
             
del X_Tune, Y_Tune
gc.collect()

   
########################################################################################################
########################################## Create Grams ################################################
########################################################################################################   

Unigram_vectorizer = TfidfVectorizer(ngram_range = (1, 1), 
                                     min_df = 10,
                                     analyzer = 'word',
                                     stop_words  = stopwords.words('english'))


Uni_X_Train = Unigram_vectorizer.fit_transform(X_Train)

Uni_X_Validation = Unigram_vectorizer.transform(X_Validation)

Uni_X_Test = Unigram_vectorizer.transform(X_Test)


print('Unigram Training Dataset:',Uni_X_Train.shape) 
print('Unigram Validation Dataset:',Uni_X_Validation.shape) 
print('Unigram Test Dataset:',Uni_X_Test.shape) 

########################################################################################################
######################## Perform Truncated SVD to reduce dimensions ####################################
######################################################################################################## 

from sklearn.decomposition import TruncatedSVD



tsvd1 = TruncatedSVD(n_components=10000, random_state=42)

svd_model_uni = tsvd1.fit(Uni_X_Train)


uni_ex_var = np.sum(svd_model_uni.explained_variance_ratio_)



reduced_uni_train = svd_model_uni.transform(Uni_X_Train)
reduced_uni_validation = svd_model_uni.transform(Uni_X_Validation)
reduced_uni_test = svd_model_uni.transform(Uni_X_Test)

print('Initial Shape of Unigram Data:',Uni_X_Train.shape)
print('Explained Variance of Unigram model: ')
pretty_percentage(uni_ex_var)
print('Reduced Shape of Unigram Data:',reduced_uni_train.shape)


with open('reduced_uni_train', 'wb') as f:
    pickle.dump(reduced_uni_train, f)

with open('reduced_uni_validation', 'wb') as f:
    pickle.dump(reduced_uni_validation, f)
    
with open('reduced_uni_test', 'wb') as f:
    pickle.dump(reduced_uni_test, f)

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
##                MLP for Unigrams                   ##
#######################################################
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.regularizers import l1
from keras.layers import LeakyReLU



class Metrics(Callback):
    def on_train_begin(self, logs={}):
         self.val_f1s = []
         self.val_recalls = []
         self.val_precisions = []
     
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(Y_Validation, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(' — val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
        print()
        return
     
metrics = Metrics()



def Dense_Layer(input_tensor, n_neurons, l1_rate = 0.02, dropout_rate = 0):
    
    X = Dense(n_neurons, kernel_regularizer= l1(l1_rate))(input_tensor)
    X = BatchNormalization(axis = -1)(X)
    X = LeakyReLU(alpha = 0.1)(X)
    X = Dropout(dropout_rate)(X)
    
    return X


def Create_Model(structure, l1_rate = 0, dropout_rate = 0, opt = 'adam', inpt = 10000):
    
    X_input = Input((inpt,))
    X = BatchNormalization(axis = -1)(X_input)
    
    for i in range(0,len(structure)):
        
        X = Dense_Layer(X, structure[i], l1_rate = l1_rate, dropout_rate = dropout_rate)
    
    X = BatchNormalization(axis = -1)(X)
    X =  Dense(1, activation = 'sigmoid')(X)
    
    model = Model(inputs = X_input, outputs = X, name='Sentiment Recognizer')
    
    model.compile(optimizer = opt, loss = "binary_crossentropy", metrics = ['accuracy'])
    
    return model

###############################################################################
###################### Structure Tuning     ###################################
###############################################################################
    
scenarios = []

scenarios.append([10])
scenarios.append([50])
scenarios.append([10,10])
scenarios.append([50,50])
scenarios.append([10,10,10])
scenarios.append([50,50,50])
scenarios.append([10,10,10,10])
scenarios.append([50,50,50,50])
scenarios.append([10,10,10,10,10])
scenarios.append([50,50,50,50,50])

epochs = 15
cnt = 1

for scenario in scenarios:
    
    structure1 = scenario
    
    model1 = Create_Model(structure1, 0.005, 0.3, 'adam',10000)
    
    model1.summary()
    
    
    
    history1 = model1.fit(reduced_uni_train, Y_Train, 
     validation_data=(reduced_uni_validation, Y_Validation),
     epochs=15,
     batch_size=256,
     callbacks=[metrics])
    
    model1.evaluate(reduced_uni_test, Y_Test)
    
    print('Number of Hidden Layers:',str(len(structure1)))
    print('Number of Neurons per Layer:',str(structure1[0]))
    plot_history(history1, epochs, 'Scenario' + str(cnt))
    
    cnt+=1

final_structure1 = scenarios[9]

###############################################################################
######################      Optimizer Tuning     ##############################
###############################################################################
    
model1 = Create_Model(final_structure1, 0.005, 0.3, 'adagrad',10000)

model1.summary()

history1 = model1.fit(reduced_uni_train, Y_Train, 
 validation_data=(reduced_uni_validation, Y_Validation),
 epochs=epochs,
 batch_size=256,
 callbacks=[metrics])

plot_history(history1, epochs, 'adagrad', 1.0)

###############################################################################
######################      Regularization Tuning     #########################
###############################################################################

epochs = 15

model1 = Create_Model(final_structure1, 0.011, 0, 'adagrad',10000)

model1.summary()

history1 = model1.fit(reduced_uni_train, Y_Train, 
 validation_data=(reduced_uni_validation, Y_Validation),
 epochs=epochs,
 batch_size=256,
 callbacks=[metrics])


###############################################################################
#############################      Final Model     ############################
###############################################################################

epochs = 15

model1 = Create_Model(final_structure1, 0.011, 0, 'adagrad',10000)

model1.summary()

history1 = model1.fit(reduced_uni_train, Y_Train, 
 validation_data=(reduced_uni_validation, Y_Validation),
 epochs=epochs,
 batch_size=256,
 callbacks=[metrics])

tr_pred  = model1.predict(reduced_uni_train)
tr_pred[tr_pred>=0.5] = 1
tr_pred[tr_pred<0.5] = 0

ts_pred  = model1.predict(reduced_uni_test)
ts_pred[ts_pred>=0.5] = 1
ts_pred[ts_pred<0.5] = 0

Score_Classifier(tr_pred, Y_Train, ts_pred, Y_Test, 
                    labels = ['Negative','Positive'], Classifier_Title = 'Final Model', 
                    evaluation_type = 'Test')



###############################################################################
#################              Error Overview              ####################
###############################################################################

true = np.array(Y_Test).reshape(2500,)/1.

predicted = ts_pred.reshape(2500,)/1.


# False Positive

fps = np.where((predicted== 1.)& (true ==0))[0]
index = np.random.choice(fps.shape[0], 1)[0]
X_Test[index]

# False Negative

fps = np.where((predicted== 0.)& (true ==1))[0]
index = np.random.choice(fps.shape[0], 1)[0]
X_Test[index]




 

















