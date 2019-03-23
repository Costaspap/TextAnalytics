########################################################################################################
##################################################### Imports ##########################################
########################################################################################################


import re
import os
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

path = '.'
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

#######################################################
##        BIDIRECTIONAL RNN WITH MLP ON TOP          ##
#######################################################

       
#Custom keras layer for linear attention over RNNs output states

from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras import backend as K
from keras.layers import InputSpec

class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average attention mechanism 
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.w = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_w'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.w]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, h, mask=None):
        h_shape = K.shape(h)
        d_w, T = h_shape[0], h_shape[1]
        
        logits = K.dot(h, self.w)  # w^T h
        logits = K.reshape(logits, (d_w, T))
        alpha = K.exp(logits - K.max(logits, axis=-1, keepdims=True))  # exp
        
        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            alpha = alpha * mask
        alpha = alpha / K.sum(alpha, axis=1, keepdims=True) # softmax
        r = K.sum(h * K.expand_dims(alpha), axis=1)  # r = h*alpha^T
        h_star = K.tanh(r)  # h^* = tanh(r)
        if self.return_attention:
            return [h_star, alpha]
        return h_star

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

# Set Maximum number of words to be embedded
NUM_WORDS = 20000

# load tokenizer from keras

from keras.preprocessing.text import Tokenizer

# Define/Load Tokenize text function
tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',lower=True)

# Fit the function on the text
tokenizer.fit_on_texts(X_Train)

# Count number of unique tokens
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

word_vectors = dict()

# load the whole embedding into memory
f = open('glove.6B.300d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_vectors[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(word_vectors))

EMBEDDING_DIM=300
vocabulary_size=min(len(word_index)+1,(NUM_WORDS))

embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

for word, i in word_index.items():
    if i>=NUM_WORDS:
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
         pass

del(word_vectors)

# load embedding library from keras

from keras.layers import Embedding

# map words into their unique id for train and test set
X_TrainSeq = tokenizer.texts_to_sequences(X_Train)
X_TestSeq = tokenizer.texts_to_sequences(X_Test)

# padding comments into sentences in order to get fixed length comments into batch
SentLen = [len(sent) for sent in X_TrainSeq]
MAXLENGTH = int(pd.DataFrame(SentLen ).quantile(0.95)[0]) ### gets the value 148

# load pad_sequences from keras

from keras.preprocessing.sequence import pad_sequences

# create batches of length MAXLENGTH
X_TrainModified = pad_sequences(X_TrainSeq, maxlen=MAXLENGTH)
X_TestModified = pad_sequences(X_TestSeq, maxlen=MAXLENGTH)

# automatically set the number of input
inp = Input(shape=(MAXLENGTH, ))

# define the vector space of the embendings
embeddings = Embedding(vocabulary_size,EMBEDDING_DIM,weights=[embedding_matrix],trainable=True)(inp)

del(embedding_matrix)

from keras.layers import Bidirectional, Input
from keras.layers.recurrent import LSTM

drop_emb = Dropout(0.2)(embeddings)

# feed Tensor into the LSTM layer
# LSTM takes in a tensor of [Batch Size, Time Steps, Number of Inputs]

BATCHSIZE = 60 # number of samples in a batch
bilstm = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer'))(drop_emb)

DENSE = 200
x, attn = AttentionWeightedAverage(return_attention=True)(bilstm)
out = Dense(units=DENSE, activation="relu")(x)
out = Dense(units=1, activation="sigmoid")(out)
model = Model(inp, out)

print(model.summary())


from keras.optimizers import Adam
from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import ModelCheckpoint

model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])

checkpoint = ModelCheckpoint('keras_BiLSTM+attn_model', monitor='val_f1', verbose=1, save_best_only=True, mode='max')

# define the model

Model = model.fit(X_TrainModified, Y_Train,
              batch_size=32,
              epochs=2,
              verbose = 0,
              callbacks=[checkpoint,TQDMNotebookCallback()],
              validation_data=(X_TestModified, Y_Validation),
              shuffle=True)


# run the model
model = Model(inputs=inp, outputs=x)
