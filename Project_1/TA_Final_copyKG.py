#######################################################################################################################
# Imports
#######################################################################################################################

from nltk import sent_tokenize, word_tokenize
from IPython.display import clear_output
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from pprint import pprint
import time, sys, math, re, os, string, gc, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from Project_1.library import clean_text,bigram_prob,split_into_bigrams,bigram_language_model,\
    bigram_linear_interpolation_language_model, bigram_linear_interpolation_probs,crossentropy_perplexity,\
    print_sentence_bigram_probs,print_sentence_trigram_probs,print_sentence_unigram_probs,split_into_trigrams,\
    split_into_unigrams,trigram_language_model,trigram_linear_interpolation_language_model,\
    trigram_linear_interpolation_probs,trigram_prob,unigram_language_model,unigram_prob, cleanse

#######################################################################################################################
# Select Run Mode. If load, it loads the preprocessed files. If create, it creates everything from scratch
#######################################################################################################################

# mode = 'Create'
mode = 'Load'
if mode == 'Create':
    corpus = []
    path = input('Give me the path to en folder: ')
    path = path if path != 'default' else os.getcwd() + '/../en/'
    sentences = []
    total = len(os.listdir(path))  # Total files
    count = 0
    for file in os.listdir(path):
        f = open(path + file, 'r', encoding="utf-8")
        file_text = f.read()
        f.close()

        regex = re.compile('[%s]' % re.escape(string.punctuation))
        sentences = [word_tokenize(regex.sub('', sent).strip()) for sent in sent_tokenize(clean_text(file_text))]
        # Q : Kosta, to keno edw xreiazetai?
        # HANDLED: I replaced the space with empty string

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
    with open('vfinal_corpus', 'wb') as f:
        pickle.dump(corpus, f)

elif mode == 'Load':
    with open('vfinal_corpus', 'rb') as f:
        corpus = pickle.load(f)

#######################################################################################################################
# Create Sets
#######################################################################################################################

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

    with open('vfinal_training_set_init', 'wb') as f:
        pickle.dump(training_set_init, f)
    with open('vfinal_validation_set_init', 'wb') as f:
        pickle.dump(validation_set_init, f)
    with open('vfinal_test1_set_init', 'wb') as f:
        pickle.dump(test1_set_init, f)
    with open('vfinal_test2_set_init', 'wb') as f:
        pickle.dump(test2_set_init, f)

elif mode == 'Load':
    with open('vfinal_training_set_init', 'rb') as f:
        training_set_init = pickle.load(f)
    with open('vfinal_validation_set_init', 'rb') as f:
        validation_set_init = pickle.load(f)
    with open('vfinal_test1_set_init', 'rb') as f:
        test1_set_init = pickle.load(f)
    with open('vfinal_test2_set_init', 'rb') as f:
        test2_set_init = pickle.load(f)

#######################################################################################################################
# Form Vocabulary
#######################################################################################################################

if mode == 'Create':
    AllWords = []
    for sentence in training_set_init:
        AllWords += sentence
    WordCounts = Counter(AllWords)
    vocabulary = [k for k, v in WordCounts.items() if v > 10]
    with open('vfinal_vocabulary', 'wb') as f:
        pickle.dump(vocabulary, f)
    with open('vfinal_AllWords', 'wb') as f:
        pickle.dump(AllWords, f)
    with open('vfinal_WordCounts', 'wb') as f:
            pickle.dump(WordCounts, f)
elif mode == 'Load':
    with open('vfinal_vocabulary', 'rb') as f:
        vocabulary = pickle.load(f)
    with open('vfinal_AllWords', 'rb') as f:
        AllWords = pickle.load(f)
    with open('vfinal_WordCounts', 'rb') as f:
        WordCounts = pickle.load(f)

#######################################################################################################################
# Replace OOV Words
#######################################################################################################################

if mode == 'Create':

    valid_WordCounts = {k: v for k, v in WordCounts.items() if v > 10}

    train_size = len(training_set_init)
    validation_size = len(validation_set_init)
    test1_size = len(test1_set_init)
    test2_size = len(test2_set_init)

    training_set = training_set_init.copy()
    validation_set = validation_set_init.copy()
    test1_set = test1_set_init.copy()
    test2_set = test2_set_init.copy()

    training_set = cleanse(training_set, train_size, valid_WordCounts)
    print('Training Set Cleaned')
    validation_set = cleanse(validation_set, validation_size, valid_WordCounts)
    print('Validation Set Cleaned')
    test1_set = cleanse(test1_set, test1_size, valid_WordCounts)
    print('Test1 Set Cleaned')
    test2_set = cleanse(test2_set, test2_size, valid_WordCounts)
    print('Test2 Set Cleaned')

    with open('vfinal_training_set', 'wb') as f:
        pickle.dump(training_set, f)
    with open('vfinal_validation_set', 'wb') as f:
        pickle.dump(validation_set, f)
    with open('vfinal_test1_set', 'wb') as f:
        pickle.dump(test1_set, f)
    with open('vfinal_test2_set', 'wb') as f:
        pickle.dump(test2_set, f)

elif mode == 'Load':
    with open('vfinal_training_set', 'rb') as f:
        training_set = pickle.load(f)
    with open('vfinal_validation_set', 'rb') as f:
        validation_set = pickle.load(f)
    with open('vfinal_test1_set', 'rb') as f:
        test1_set = pickle.load(f)
    with open('vfinal_test2_set', 'rb') as f:
        test2_set = pickle.load(f)
    print('Sets Loaded')

#######################################################################################################################
# N-Grams Modelling
#######################################################################################################################

C = len(AllWords)
V = len(vocabulary)

# For Ram optimization I dont create the lists containing the ngrams, I count them directly
if mode == 'Create':
    unigrams_training_counter = {}
    bigrams_training_counter = {}
    trigrams_training_counter = {}

    with open('vfinal_unigrams_training_counter', 'wb') as f:
        pickle.dump(unigrams_training_counter, f)
    with open('vfinal_bigrams_training_counter', 'wb') as f:
        pickle.dump(bigrams_training_counter, f)
    with open('vfinal_trigrams_training_counter', 'wb') as f:
        pickle.dump(trigrams_training_counter, f)

elif mode == 'Load':
    with open('vfinal_unigrams_training_counter', 'rb') as f:
        unigrams_training_counter = pickle.load(f)
    with open('vfinal_bigrams_training_counter', 'rb') as f:
        bigrams_training_counter = pickle.load(f)
    with open('vfinal_trigrams_training_counter', 'rb') as f:
        trigrams_training_counter = pickle.load(f)
    print('Counters Loaded')


#######################################################################################################################
# ii)  Check the log-probabilities that your trained models return when given (correct) sentence
#      from the test subset vs. (incorrect) sentences of the same length (in words) consisting of
#      randomly selected vocabulary words.
#######################################################################################################################

# Test 10 random sentences in the test1 set against 10 sentences randomly created from the vocabulary
np.random.seed(666)
for i in range(0, 10):
    # Get random normal sentence
    print('-----------------------------------------------------------------------------')
    print('Test ', str(i))
    print('Normal sentence results:')
    sentence_idx = np.random.randint(low=0, high=len(test1_set))
    valid_sentence = test1_set[sentence_idx]
    print_sentence_unigram_probs(valid_sentence, V, C)
    print_sentence_bigram_probs(valid_sentence, V)
    print_sentence_trigram_probs(valid_sentence, V)
    ## VS a randomly created non-sense sentence
    print()
    print('Invalid sentence results:')
    random_sent_idx = np.random.randint(low=0, high=len(vocabulary), size=len(valid_sentence))
    invalid_sentence = [vocabulary[idx] for idx in random_sent_idx]
    print_sentence_unigram_probs(invalid_sentence, V, C)
    print_sentence_bigram_probs(invalid_sentence, V)
    print_sentence_trigram_probs(invalid_sentence, V)
    print('-----------------------------------------------------------------------------')

#######################################################################################################################
# (iii) Estimate the language cross-entropy and perplexity of your models on the test subset of
# the corpus, treating the entire test subset as a single sequence, with *start* (or *start1*,
# *start2*) at the beginning of each sentence, and *end* at the end of each sentence. Do not
# include probabilities of the form P(*start*|…) (or P(*start1*|…) or P(*start2*|…)) in the
# computation of perplexity, but include probabilities of the form P(*end*|…).
#######################################################################################################################

'''
Compute corpus cross_entropy 
& perplexity for interpoladed bi-gram
& tri-gram LMs 
'''

print("\n------------------------------")
print("Crossentropies & perplexities of models")
vocab_size = len(vocabulary)
for ngram_type in ['unigram', 'bigram', 'trigram']:
    crossentropy_perplexity(ngram_type, vocab_size, C, a=1, l=0.7, l1=0.7, l2=0.2)
    print("\n------------------------------")

#######################################################################################################################
# (iv) Optionally combine your two models using linear interpolation (slide 10) and check if the
# combined model performs better.
#######################################################################################################################

print("\n------------------------------")
print("Crossentropies & perplexities of models")
vocab_size = len(vocabulary)
for ngram_type in ['lin_pol_bi', 'lin_pol_tri']:
    crossentropy_perplexity(ngram_type, vocab_size, C, a=1, l=0.7, l1=0.7, l2=0.2)
    print("\n------------------------------")

#######################################################################################################################
# TODO : Tuning the parameters
#######################################################################################################################
for l in range(1,9):
    print('-----------------------------------------------------------------------------')
    print('For lambda = ',str(l/10))
    calculate_metrics(test1_set,l/10)
    print('-----------------------------------------------------------------------------')


for l in range(90,100,1):
    print('-----------------------------------------------------------------------------')
    print('For lambda = ',str(l/100))
    calculate_metrics(test1_set,l/100)
    print('-----------------------------------------------------------------------------')
    
#######################################################################################################################
#######################################################################################################################