#######################################################################################################################
# Imports
#######################################################################################################################


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


from pprint import pprint
import time
import sys
import math
import numpy as np
from sklearn.model_selection import train_test_split



########## Select Run Mode. If load, it loads the preprocessed files. If create, it creates everything from scratch #######


mode = 'Create'
#mode = 'Load'

def clean_text(text):
    """ 
    1. Remove html like text from europarl e.g. <Chapter 1>
    2. Remove line breaks
    3. Reduce all whitespaces to 1
    4. turn everything to lower case
    """
    clean = re.compile('<.*?>')

    out = text.replace('\n', ' ')  # Remove line breaks
    out = re.sub(clean, ' ', out)  # Remove tagged text e.g. <Chapter 1>
    out = re.sub(' +', ' ', out)  # Reduce whitespace down to one

    out = out.lower()  # Turn everything to lower case

    return out

if mode == 'Create':
    
    corpus = []


    path = input('Give me the path to en folder: ')

    sentences = []

    total = len(os.listdir(path))  # Total files
    count = 0

    for file in os.listdir(path):
        f = open(path + file, 'r', encoding="utf-8")
        file_text = f.read()
        f.close()

        regex = re.compile('[%s]' % re.escape(string.punctuation))
        sentences = [word_tokenize(regex.sub(' ', sent).strip()) for sent in sent_tokenize(clean_text(file_text))]


        corpus += sentences
        # What just happened in cleansing:
        #
        # -1- The whole single text of the corpus had removed: <tags> & \n
        # -2- Then the whole single text of the corpus had all big spaces to become single spaces
        # -3- Then the whole single text of the corpus was tokenized using its punctuation.
        # -4- Each token (string) corresponds to a sentence
        # -5- From each sentence string we removed all punctuation.
        #       -- SOS: We do not need spaces in their place, the sentences have already spaces where needed.
        # -6- corpus_clean is where the new sentences are stored
        # -7- corpus_clean is a list of sentences, where the sentences are strings.

        count += 1

        print('File ' + file + ' finished. Completed ' + str(round(count * 100 / total, 2)) + '%')

    # Save the basic objects:
    with open('corpus', 'wb') as f:
        pickle.dump(corpus, f)


if mode == 'Load':

    with open('corpus', 'rb') as f:
        corpus = pickle.load(f)

#########################################################################################
####################            Create Sets              ################################
#########################################################################################

if mode == 'Create':
    
    total = len(corpus)
    sets = list(range(0, total))

    # 60% for train
    training_idx, tuning_idx = train_test_split(sets, train_size=.6, random_state=2019)

    # 20%,10%,10% for validation(development), test1 and test2 datasets.
    validation_idx, test_idx = train_test_split(tuning_idx, train_size=.5, random_state=2019)
    test1_idx, test2_idx = train_test_split(test_idx, train_size=.5, random_state=2019)

    training_set_init = [corpus[i] for i in training_idx]
    validation_set_init = [corpus[i] for i in validation_idx]
    test1_set_init = [corpus[i] for i in test1_idx]
    test2_set_init = [corpus[i] for i in test2_idx]

    del training_idx, validation_idx, tuning_idx, test1_idx, test2_idx
    gc.collect()

    print('Training Size: ', len(training_set_init))
    print('Validation Size: ', len(validation_set_init))
    print('Test1 Size: ', len(test1_set_init))
    print('Test2 Size: ', len(test2_set_init))

    with open('training_set_init', 'wb') as f:
        pickle.dump(training_set_init, f)
    with open('validation_set_init', 'wb') as f:
        pickle.dump(validation_set_init, f)
    with open('test1_set_init', 'wb') as f:
        pickle.dump(test1_set_init, f)
    with open('test2_set_init', 'wb') as f:
        pickle.dump(test2_set_init, f)


if mode == 'Load':
  
    with open('training_set_init', 'rb') as f:
        training_set_init = pickle.load(f)

    with open('validation_set_init', 'rb') as f:
        validation_set_init = pickle.load(f)

    with open('test1_set_init', 'rb') as f:
        test1_set_init = pickle.load(f)

    with open('test2_set_init', 'rb') as f:
        test2_set_init = pickle.load(f)




#########################################################################################
####################           Form Vocabulary           ################################
#########################################################################################

if mode == 'Create':
    
    AllWords = []

    for sentence in training_set_init:
        AllWords += sentence


    WordCounts = Counter(AllWords)

    vocabulary = [k for k, v in WordCounts.items() if v > 10]

    with open('vocabulary', 'wb') as f:
        pickle.dump(vocabulary, f)
    with open('AllWords', 'wb') as f:
        pickle.dump(AllWords, f)



if mode == 'Load':

    with open('vocabulary', 'rb') as f:
        vocabulary = pickle.load(f)

    with open('AllWords', 'rb') as f:
        AllWords = pickle.load(f)


#########################################################################################
####################           Replace OOV Words           ##############################
#########################################################################################


if mode == 'Create':
    
    valid_WordCounts = {k:v for k, v in WordCounts.items() if v>10}
    
    train_size = len(training_set_init)
    validation_size = len(validation_set_init)
    test1_size = len(test1_set_init)
    test2_size = len(test2_set_init)
    
    training_set  = training_set_init.copy()
    validation_set  = validation_set_init.copy()
    test1_set  = test1_set_init.copy()
    test2_set  = test2_set_init.copy()
    
    
    for i in range(0,train_size):
        for j in range(0,len(training_set_init[i])):
            if training_set_init[i][j] not in valid_WordCounts:
                training_set[i][j] = 'UNK'
    print('Training Set Cleaned')

    for i in range(0,validation_size):
        for j in range(0,len(validation_set_init[i])):
            if validation_set_init[i][j] not in valid_WordCounts:
                validation_set[i][j] = 'UNK'
    print('Validation Set Cleaned')
                
    for i in range(0,test1_size):
        for j in range(0,len(test1_set_init[i])):
            if test1_set_init[i][j] not in valid_WordCounts:
                test1_set[i][j] = 'UNK'
    print('Test1 Set Cleaned')
               
    for i in range(0,test2_size):
        for j in range(0,len(test2_set_init[i])):
            if test2_set_init[i][j] not in valid_WordCounts:
                test2_set[i][j] = 'UNK'
    print('Test2 Set Cleaned')
    

    with open('training_set', 'wb') as f:
        pickle.dump(training_set, f)
        
    with open('validation_set', 'wb') as f:
        pickle.dump(validation_set, f)
        
    with open('test1_set', 'wb') as f:
        pickle.dump(test1_set, f)
        
    with open('test2_set', 'wb') as f:
        pickle.dump(test2_set, f)


if mode == 'Load':
    
    with open('training_set', 'rb') as f:
        training_set = pickle.load(f)

    with open('validation_set', 'rb') as f:
        validation_set = pickle.load(f)

    with open('test1_set', 'rb') as f:
        test1_set = pickle.load(f)

    with open('test2_set', 'rb') as f:
        test2_set = pickle.load(f)
    print('Sets Loaded')


#########################################################################################
####################           N-Grams Modelling           ##############################
#########################################################################################

C = len(AllWords)
V = len(vocabulary)

def split_into_unigrams(sentence):
    if sentence.__class__ == str:
        print("Error in corpus sentence!!")
    ngs = [gram for gram in ngrams(sentence, 1)]
    return ngs


def split_into_bigrams(sentence, pad=True, s="start", e="end"):
    if sentence.__class__ == str:
        print("Error in corpus sentence!!")
    list = [s]+sentence+[e] if pad else sentence
    ngs = [gram for gram in ngrams(list, 2)]
    return ngs


def split_into_trigrams(sentence, pad=True, s1= "start1", s2= 'start2', e='end'):
    if sentence.__class__ == str:
        print("Error in corpus sentence!!")
    list = [s1, s2]+sentence+[e] if pad else sentence
    ngs = [gram for gram in ngrams(list, 3)]
    return ngs


# For Ram optimization I dont create the lists containing the ngrams, I count them directly
if mode == 'Create':

    unigrams_training_counter = {}
    bigrams_training_counter = {}
    trigrams_training_counter = {}
    
    ###### Counters ########
    
    for sentence in training_set:
        
        for unigram in split_into_unigrams(sentence):
            try:
                unigrams_training_counter[unigram] += 1
            except:
                unigrams_training_counter[unigram] = 1
                
        for bigram in split_into_bigrams(sentence):
            try:
                bigrams_training_counter[bigram] += 1
            except:
                bigrams_training_counter[bigram] = 1            
     
        for trigram in split_into_trigrams(sentence):
            try:
                trigrams_training_counter[trigram] += 1
            except:
                trigrams_training_counter[trigram] = 1
               
    
        
    with open('unigrams_training_counter', 'wb') as f:
        pickle.dump(unigrams_training_counter, f)
    with open('bigrams_training_counter', 'wb') as f:
        pickle.dump(bigrams_training_counter, f)
    with open('trigrams_training_counter', 'wb') as f:
        pickle.dump(trigrams_training_counter, f)



if mode == 'Load':
    
    with open('unigrams_training_counter', 'rb') as f:
        unigrams_training_counter = pickle.load(f)

    with open('bigrams_training_counter', 'rb') as f:
        bigrams_training_counter = pickle.load(f)

    with open('trigrams_training_counter', 'rb') as f:
        trigrams_training_counter = pickle.load(f)
    print('Counters Loaded')



## ii)  Check the log-probabilities that your trained models return when given (correct) sentences
##      from the test subset vs. (incorrect) sentences of the same length (in words) consisting of
##      randomly selected vocabulary words. 



def unigram_prob(ngram, vocab_size, C, a=0.01):
    x = ngram[0]
    try:
        out = (unigrams_training_counter[(x,)] + a) / (C + a*vocab_size)
    except:
        out = a / (C + a*vocab_size)
    return out


def bigram_prob(ngram, vocab_size, a=0.01):
    x = ngram[0]
    y = ngram[1]
    
    try:
        out = (bigrams_training_counter[(x,y)] + a) / (unigrams_training_counter[(x,)] + a*vocab_size)
    except:
        out = (a) / (a*vocab_size)
    return out

def trigram_prob(ngram, vocab_size, a=0.01):
    x = ngram[0]
    y = ngram[1]
    z = ngram[2]
    
    try:
        out = (trigrams_training_counter[(x,y,z)] + a) / (bigrams_training_counter[(x,y,)] + a*vocab_size)
    except:
        out = a / (a*vocab_size)
    return out




def print_sentence_unigram_probs(sentence, vocab_size, C, a=0.01):
    for unigram in split_into_unigrams(sentence):
        print(unigram, np.round(100*unigram_prob(unigram, vocab_size, C, a),2) , " %")

def print_sentence_bigram_probs(sentence, vocab_size, a=0.01):
    for bigram in split_into_bigrams(sentence):
        print(bigram, np.round(100*bigram_prob(bigram, vocab_size, a),2) , " %")

def print_sentence_trigram_probs(sentence, vocab_size, a=0.01):
    for trigram in split_into_trigrams(sentence):
        print(trigram, np.round(100*trigram_prob(trigram, vocab_size, a),2), " %")



## Test a normal Sentence


print_sentence_bigram_probs(validation_set[6],V)

## VS a non-sense sentence

print_sentence_bigram_probs(['not','dog','space','understand','laplace'],V)






## (iii) Estimate the language cross-entropy and perplexity of your models on the test subset of
## the corpus, treating the entire test subset as a single sequence, with *start* (or *start1*,
## *start2*) at the beginning of each sentence, and *end* at the end of each sentence. Do not
## include probabilities of the form P(*start*|…) (or P(*start1*|…) or P(*start2*|…)) in the
## computation of perplexity, but include probabilities of the form P(*end*|…).


'''
Compute corpus cross_entropy 
& perplexity for interpoladed bi-gram
& tri-gram LMs 
'''
def calculate_metrics(dataset,lamda = 0.9):
#We should fine-tune lamda on a held-out dataset

    sum_prob = 0
    ngram_cnt = 0
    for sent in training_set:
        sent = ['<s>'] + ['<s>'] + sent + ['<e>'] + ['<e>']
        for idx in range(2,len(sent)):
            tr_prob = trigram_prob([sent[idx-2],sent[idx-1], sent[idx]],V)
            b_prob = bigram_prob([sent[idx-1], sent[idx]],V)
    
            sum_prob += (lamda * math.log2(tr_prob)) +((1-lamda) * math.log2(b_prob))
            ngram_cnt+=1 
    
    HC = -sum_prob / ngram_cnt
    perpl = math.pow(2,HC)
    print("Cross Entropy: {0:.3f}".format(HC))
    print("perplexity: {0:.3f}".format(perpl))

## (iv) Optionally combine your two models using linear interpolation (slide 10) and check if the
## combined model performs better. 


calculate_metrics(validation_set)



