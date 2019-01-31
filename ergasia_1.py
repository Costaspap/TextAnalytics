#######################################################################################################################
# Ergasia 1
#######################################################################################################################

import re
import os
import string
from nltk import sent_tokenize, word_tokenize, download
from IPython.display import clear_output
from collections import Counter
import gc
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
"""
# Save objects
with open('object_name', 'wb') as f:
    pickle.dump(object_variable, f)

# Load objects
with open('object_name', 'rb') as f:
    object_variable = pickle.load(f)
"""
from pprint import pprint
import time
import sys
import math
import numpy as np
print("Imports Completed")
# download('punkt')

#######################################################################################################################
# Cockpit:
#######################################################################################################################

read_corpus = False
clean_corpus = False
do_tokenize = False
shortcut_1 = False
do_vocabularies = False
do_oov = False
shortcut_2 = True
demo_ngrams = False
sample_ngrams = False

#######################################################################################################################


def clean_text(text):
    """ 
    1. Remove html like text from europarl e.g. <Chapter 1>
    2. Remove line breaks
    3. Reduce all whitespaces to 1
    4. turn everything to lower case
    """
    clean = re.compile('<.*?>')
    
    out = text.replace('\n', ' ') # Remove line breaks
    out = re.sub(clean, ' ', out) # Remove tagged text e.g. <Chapter 1>
    out = re.sub(' +', ' ', out) # Reduce whitespace down to one
    
    out = out.lower() # Turn everything to lower case
    
    return out


def telos():
    """
    Ends execution with pre-defined message.
    """
    time.sleep(1)
    sys.exit("Requested Exit.")

#######################################################################################################################

corpus_clean = []
corpus_original = ''
if read_corpus:
    abs_path = os.getcwd()
    path = abs_path + '/en/'
    sentences = []
    text = ''
    total = len(os.listdir(path)) # Total files
    count = 0

    for file in os.listdir(path):
        f = open(path + file, 'r', encoding="utf-8")
        file_text = f.read()
        corpus_original = corpus_original + file_text
        f.close()

        regex = re.compile('[%s]' % re.escape(string.punctuation))
        file_sentences = [regex.sub('', sent).strip() for sent in sent_tokenize(clean_text(file_text))]

        corpus_clean = corpus_clean + file_sentences
        count += 1

        clear_output(wait = True)
        print('File ' + file + ' finished. Completed ' + str(round(count*100/total,2)) + '%')

#######################################################################################################################

"""
demo corpus file:
------------------------------------------------------------------
original text: <class 'str'>
<CHAPTER ID="005">
Voting time
<SPEAKER ID="104" NAME="President">
The next item is the vote.
<P>
(For the results and other details on the vote: see Minutes)
------------------------------------------------------------------
Its clean_text(text): <class 'str'>
 voting time the next item is the vote. (for the results and other details on the vote: see minutes) 
------------------------------------------------------------------
Its sent_tokenize(clean_text(text)): <class 'list'>
[' voting time the next item is the vote.', '(for the results and other details on the vote: see minutes)']
------------------------------------------------------------------
Its sentences = [sent.strip() for sent in sent_tokenize(clean_text(text))]: <class 'list'>
['voting time the next item is the vote.', '(for the results and other details on the vote: see minutes)']
------------------------------------------------------------------
"""

#######################################################################################################################
# Save the basic objects:

if read_corpus:
    with open('corpus_original', 'wb') as f:
        pickle.dump(corpus_original, f)
    with open('corpus_clean', 'wb') as f:
        pickle.dump(corpus_clean, f)

#######################################################################################################################

corpus_clean_string = None
if clean_corpus:
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    corpus_clean_string = regex.sub('', clean_text(corpus_original))

    print('-------------------------')
    print('corpus_clean_string created.')
    print('-------------------------')

    with open('corpus_clean_string', 'wb') as f:
        pickle.dump(corpus_clean_string, f)

    del corpus_original
    gc.collect()

#######################################################################################################################

if do_tokenize:
    AllWords = word_tokenize(corpus_clean_string)
    print('-------------------------')
    print('Words Tokenized.')
    print('-------------------------')

    vocabulary = set(AllWords)
    print('Vocabulary Created.')
    print('-------------------------')

    WordCounts = Counter(AllWords)
    print('WordCounts Calculated.')
    print('-------------------------')

    with open('AllWords', 'wb') as f:
        pickle.dump(AllWords, f)

    with open('vocabulary', 'wb') as f:
        pickle.dump(vocabulary, f)

    with open('WordCounts', 'wb') as f:
        pickle.dump(WordCounts, f)

    del corpus_clean_string, AllWords
    gc.collect()

#######################################################################################################################

corpus_clean = None
WordCounts = None
vocabulary = None
if shortcut_1:

    # Load objects

    with open('corpus_clean', 'rb') as f:
        corpus_clean = pickle.load(f)

    with open('vocabulary', 'rb') as f:
        vocabulary = pickle.load(f)

    with open('WordCounts', 'rb') as f:
        WordCounts = pickle.load(f)

    with open('AllWords', 'rb') as f:
        AllWords = pickle.load(f)

#######################################################################################################################
# Ignore low frequency words

valid_vocabulary = None
invalid_vocabulary = None
if do_vocabularies:
    valid_vocabulary = [k for k,v in WordCounts.items() if v > 10]
    invalid_vocabulary = [k for k,v in WordCounts.items() if v <= 10]
    print("valid voc", len(valid_vocabulary))
    print("invalid voc", len(invalid_vocabulary))

    with open('valid_vocabulary', 'wb') as f:
        pickle.dump(valid_vocabulary, f)

    with open('invalid_vocabulary', 'wb') as f:
        pickle.dump(invalid_vocabulary, f)

#######################################################################################################################
# Replace OOV words in sentences


def split_sentence(sentence):
    PATTERN = '\w+|\(|\)|\.|\,'
    tokenizer = RegexpTokenizer(pattern=PATTERN) 
    return tokenizer.tokenize(sentence)


if do_oov:
    dummy_count = 0
    total = len(corpus_clean)
    for i in range(0,len(corpus_clean)):
        sentence = ''.join(corpus_clean[i]) # make it string
        splitted_sent = split_sentence(sentence)
        new_sent = []
        for word in splitted_sent:
            new_word = 'UNK' if word not in valid_vocabulary else word
            new_sent.append(new_word)
            corpus_clean[i] = new_sent
        clear_output(wait = True)
        print('Sentences processed ' + str(i+1) + ' out of ' + str(total))
        dummy_count = dummy_count + 1
        if 1000 < dummy_count:
            break
    # Have it here, in order to not to forget to save after the big computation burden.
    with open('corpus_clean_no_OOV', 'wb') as f:
        pickle.dump(corpus_clean, f)

#######################################################################################################################
# Deprecated (slower)
#
# Replace OOV words in sentences
# total = len(corpus_clean)
# for i in range(0,len(corpus_clean)):
#    for word in valid_vocabulary:
#        corpus_clean[i].replace(word, 'UNK')
#   clear_output(wait = True)
#   print('Sentences processed ' + str(i+1) + ' out of ' + str(total) )
#######################################################################################################################
# Creating the n-grams is actually not needed!! Only counting them is!!
#
# tokens = AllWords
# bigrams = [ gram for gram in ngrams(tokens, 2) ]
# trigrams = [ gram for gram in ngrams(tokens, 3) ]
# #pprint(bigrams)
# with open('bigrams', 'wb') as f:
#     pickle.dump(bigrams, f)
# with open('trigrams', 'wb') as f:
#    pickle.dump(bigrams, f)
#######################################################################################################################

valid_vocabulary = None
invalid_vocabulary = None
corpus_clean_no_OOV = None
C = None

if shortcut_2:
    # Load objects

    with open('invalid_vocabulary', 'rb') as f:
        invalid_vocabulary = pickle.load(f)

    with open('valid_vocabulary', 'rb') as f:
        valid_vocabulary = pickle.load(f)

    with open('AllWords', 'rb') as f:
        AllWords = pickle.load(f)
    C = len(AllWords)

    with open('corpus_clean_no_OOV', 'rb') as f:
        corpus_clean_no_OOV = pickle.load(f)

#######################################################################################################################

demo_ngrams = False
if demo_ngrams:
    print(corpus_clean_no_OOV.__class__)
    print(corpus_clean_no_OOV[0])
    print(corpus_clean_no_OOV[0].__class__)
    print(corpus_clean_no_OOV[0][0].__class__)

    print(AllWords.__class__)
    print(AllWords[0])
    print(AllWords[0].__class__)
    print(AllWords[0:10])

    gc.collect()

    # Single sentence, for testing corpus_clean_no_OOV:
    unigram_counter = Counter()
    unigram_counter.update([gram for gram in ngrams(corpus_clean_no_OOV[0], 1, pad_left=True, pad_right=True,
                                                       left_pad_symbol='<s>',right_pad_symbol='<e>') ])
    pprint(corpus_clean_no_OOV[0])
    pprint(unigram_counter)

    # Single sentence for testing AllWords:
    unigram_counter = Counter()
    unigram_counter.update([gram for gram in ngrams(AllWords[0:10], 1, pad_left=True, pad_right=True,
                                                       left_pad_symbol='<s>',right_pad_symbol='<e>') ])
    pprint(AllWords[0:10])
    pprint(unigram_counter)

#######################################################################################################################

unigram_counter = Counter()
bigram_counter = Counter()
trigram_counter = Counter()
sample_ngrams = False
if sample_ngrams:
    print("Just started the sample ngram-ing")
    sample = corpus_clean_no_OOV[0:1000]

    for sent in sample:
        unigram_counter.update([gram for gram in ngrams(sent, 1, pad_left=True, pad_right=True,
                                                       left_pad_symbol='<s>',right_pad_symbol='<e>') ])
    #pprint(unigram_counter)

    for sent in sample:
        bigram_counter.update([gram for gram in ngrams(sent, 2, pad_left=True, pad_right=True,
                                                       left_pad_symbol='<s>',right_pad_symbol='<e>') ])
    # pprint(bigram_counter)

    sample = corpus_clean_no_OOV[0:1000]
    for sent in sample:
        trigram_counter.update([gram for gram in ngrams(sent, 3, pad_left=True, pad_right=True,
                                                       left_pad_symbol='<s>',right_pad_symbol='<e>') ])
    # pprint(trigram_counter)
    print("Just ended the sample ngram-ing")

#######################################################################################################################
# Theory explained:
#######################################################################################################################
#
#  Language model =
#   P(w_i_k):=
#   for bigrams:= P(w_1|start)*P(w_2|w_1)*..*P(w_k|w_k-1) =
#       where P(w_k|w_k-1) = [c(w_k-1,w_k) + a] / [c(w_k-1 + a*|V|] =
#       e.g. P(('the', 'Department')) = [C(('the', 'Department')) + a ] / [ C(('the',)) + a*|V| ] =
#           = bigram_prob =
#           = (bigram_counter[('the', 'Department')] + a) / (unigram_counter[('the',)] + a*vocab_size)
#   for trigrams:= P(w_1|start1,start2)*P(w_2|start2,w_1)*P(w_3|w_1,w_2)*..*P(w_k|w_k-2,w_k-1)
#       where P(w_k|w_k-2,w_k-1) = [c(w_k-2,w_k-1,w_k) + a] / [c(w_k-2,w_k-1) + a * |V|]
#       e.g. P(('all', 'the', 'Departments')) =
#           = [C(('all', 'the', 'Departments')) + a ] / [ C(('all', 'the')) + a*|V| ] =
#           = trigram_prob =
#           = (trigram_counter[('all', 'the', 'Departments')] + a) /
#                   (bigram_counter[('all','the')] + alpha*vocab_size)
#
# Most probable sentence:=
#       t_1_k_opt = argmax{P(t_1_k | w_1_k)} =
#       argmax{P(t_1_k) * P(w_1_k | t_1_k)} =
#       argmax{language_model_for_1_k * P(w_1_k | t_1_k)}}
#       where
#       P(w_1_k|t_1_k) =
#       Π_i=1_k{P(w_i|t_i)} =
#       Π_i=1_k{ 1 / edit_distance}
#
#######################################################################################################################

'''
Calculate the probability
of bigram ('the', 'Department')
P(('the', 'Department')) = C(('the', 'Department')) + 1 / C(('the',)) + |V|

'''

#We should fine-tune alpha on a held-out dataset
alpha = 0.01
#Calculate vocab size
vocab_size = len(valid_vocabulary)
#Bigram prob + laplace smoothing
bigram_prob = (bigram_counter[('the', 'Department')] +alpha) / (unigram_counter[('the',)] + alpha*vocab_size)
print("bigram_prob: {0:.3f} ".format(bigram_prob))
bigram_log_prob = math.log2(bigram_prob)
print("bigram_log_prob: {0:.3f}".format(bigram_log_prob) )


#######################################################################################################################

# TODO
# -1- functions for spliting a sentence into all serial unigrams, bigrams and trigrams needed for the prob formulas (done)
# -2- functions for bigram prob (done)
# -3- functions for trigram prob (done)
# -4- functions for Linear interpolation (done)
# -5- functions for combining the above probs into making the P(t_i_k), aka the Language models (done)
# -6- function for edit distance
# -7- function for most probable sentence

#######################################################################################################################
# -1- function for spliting a sentence into all serial bigrams and trigrams needed for the prob formulas

# THESE 3 ARE A MUST:
unigram_counter = Counter()
bigram_counter = Counter()
trigram_counter = Counter()
# Reminder:
# C = len(AllWords)


def split_into_unigrams(sentence):
    ngs = [gram for gram in ngrams(sentence, 1)]
    unigram_counter.update(ngs)
    return ngs


def split_into_bigrams(sentence, pad=True, s="start", e="end"):
    list = [s]+sentence+[e] if pad else sentence
    ngs = [gram for gram in ngrams(list, 2)]
    bigram_counter.update(ngs)
    return ngs


def split_into_trigrams(sentence, pad=True, s1= "start1", s2= 'start2', e='end'):
    list = [s1, s2]+sentence+[e] if pad else sentence
    ngs = [gram for gram in ngrams(list, 3)]
    trigram_counter.update(ngs)
    return ngs


sentence = corpus_clean_no_OOV[0]
print(sentence)
print(split_into_bigrams(sentence))
print(split_into_trigrams(sentence))

#######################################################################################################################
# -2- functions for unigram & bigram prob


def unigram_prob(ngram, vocab_size, C, a=0.01):
    x = ngram[0]
    return (unigram_counter[(x,)] + a) / (C + a*vocab_size)


def bigram_prob(ngram, vocab_size, a=0.01):
    x = ngram[0]
    y = ngram[1]
    return (bigram_counter[(x,y)] + a) / (unigram_counter[(x,)] + a*vocab_size)


def print_sentence_unigram_probs(sentence, vocab_size, C, a=0.01):
    for unigram in split_into_unigrams(sentence):
        print(unigram, np.round(100*unigram_prob(unigram, vocab_size, C, a),2) , " %")


def print_sentence_bigram_probs(sentence, vocab_size, a=0.01):
    for bigram in split_into_bigrams(sentence):
        print(bigram, np.round(100*bigram_prob(bigram, vocab_size, a),2) , " %")


print("\n------------------------------")
sentence = corpus_clean_no_OOV[0]
vocab_size = len(valid_vocabulary)
print_sentence_unigram_probs(sentence, vocab_size, C, a=0.01)
print("\n------------------------------")
sentence = corpus_clean_no_OOV[0]
vocab_size = len(valid_vocabulary)
print_sentence_bigram_probs(sentence, vocab_size, a=0.01)
print("\n------------------------------")
sentence = corpus_clean_no_OOV[0] + ['next','item']
vocab_size = len(valid_vocabulary)
print_sentence_bigram_probs(sentence, vocab_size, a=0.01)
print("\n------------------------------")

#######################################################################################################################
# -3- functions for trigram prob (almost ready)


def trigram_prob(ngram, vocab_size, a=0.01):
    x = ngram[0]
    y = ngram[1]
    z = ngram[2]
    return (trigram_counter[(x,y,z)] + a) / (bigram_counter[(x,y,)] + a*vocab_size)


def print_sentence_trigram_probs(sentence, vocab_size, a=0.01):
    for trigram in split_into_trigrams(sentence):
        print(trigram, np.round(100*trigram_prob(trigram, vocab_size, a),2), " %")


print("\n------------------------------")
sentence = corpus_clean_no_OOV[0]
vocab_size = len(valid_vocabulary)
print_sentence_trigram_probs(sentence, vocab_size, a=0.01)
print("\n------------------------------")
sentence = corpus_clean_no_OOV[0] + ['next','item','is']
vocab_size = len(valid_vocabulary)
print_sentence_trigram_probs(sentence, vocab_size, a=0.01)
print("\n------------------------------")


#######################################################################################################################
# -4- functions for Linear interpolation

def bigram_linear_interpolation_probs(sentence, vocab_size, C, a=0.01, l = 0.7):
    """
    Due to the nature of the calculations, I return all bigram linear interpolations at once for the sentence
    :param sentence:
    :param vocab_size:
    :param a:
    :return:
    """
    bigrams = split_into_bigrams(sentence)
    bigram_linear_interpolations = []
    for bigram in bigrams:
        unigram = bigram[0]
        linear_interpolation = l * bigram_prob(bigram, vocab_size, a) + (l-1) * unigram_prob(unigram, vocab_size, C, a)
        bigram_linear_interpolations.append([bigram, np.round(linear_interpolation,4)])
    return bigram_linear_interpolations


def trigram_linear_interpolation_probs(sentence, vocab_size, C, a=0.01, l1 = 0.7, l2 = 0.2):
    """
    Due to the nature of the calculations, I return all trigram linear interpolations at once for the sentence
    :param sentence:
    :param vocab_size:
    :param a:
    :return:
    """
    trigrams = split_into_trigrams(sentence)
    trigram_linear_interpolations = []
    for trigram in trigrams:
        bigram = (trigram[0], trigram[1],)
        unigram = bigram[0]
        linear_interpolation = l1 * trigram_prob(trigram, vocab_size, a) + l2 * bigram_prob(bigram, vocab_size, a) +\
                               (1 - l1 -l2) * unigram_prob(unigram, vocab_size, C, a)
        trigram_linear_interpolations.append([trigram, np.round(linear_interpolation,4)])
    return trigram_linear_interpolations


print("\n------------------------------")
sentence = corpus_clean_no_OOV[0]
vocab_size = len(valid_vocabulary)
pprint(bigram_linear_interpolation_probs(sentence, vocab_size, C))
print("\n------------------------------")
sentence = corpus_clean_no_OOV[0]
vocab_size = len(valid_vocabulary)
pprint(trigram_linear_interpolation_probs(sentence, vocab_size, C))
print("\n------------------------------")

#######################################################################################################################
# -5- functions for combining the above probs into making the P(t_i_k), aka the Language models


def unigram_language_model(sentence, vocab_size, C, a = 0.01):
    language_model = 1 # neutral value
    for unigram in split_into_unigrams(sentence):
        language_model *= unigram_prob(unigram, vocab_size, C, a)
    return language_model


def bigram_language_model(sentence, vocab_size, a = 0.01):
    language_model = 1 # neutral value
    for bigram in split_into_bigrams(sentence):
        language_model *= bigram_prob(bigram, vocab_size, a)
    return language_model


def trigram_language_model(sentence, vocab_size, a = 0.01):
    language_model = 1 # neutral value
    for trigram in split_into_trigrams(sentence):
        language_model *= trigram_prob(trigram, vocab_size, a)
    return language_model


def bigram_linear_interpolation_language_model(sentence, vocab_size, C, a=0.01, l = 0.7):
    language_model = 1 # neutral value
    for pair in bigram_linear_interpolation_probs(sentence, vocab_size, C):
        prob = pair[1]
        language_model *= prob
    return language_model


def trigram_linear_interpolation_language_model(sentence, vocab_size, C, a=0.01, l1 = 0.7, l2 = 0.2):
    language_model = 1 # neutral value
    for pair in trigram_linear_interpolation_probs(sentence, vocab_size, C):
        prob = pair[1]
        language_model *= prob
    return language_model


print("\n------------------------------")
print("Language models for single sentence")
sentence = corpus_clean_no_OOV[0]
vocab_size = len(valid_vocabulary)
print(np.round(unigram_language_model(sentence, vocab_size, C),2)," %")
print(np.round(bigram_language_model(sentence, vocab_size, C),2)," %")
print(np.round(trigram_language_model(sentence, vocab_size, C),2)," %")
print(np.round(bigram_linear_interpolation_language_model(sentence, vocab_size, C),2)," %")
print(np.round(trigram_linear_interpolation_language_model(sentence, vocab_size, C),2)," %")
print("\n------------------------------")

#######################################################################################################################

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# FIN
#######################################################################################################################

# Can comment out locally, to move on.
telos()

########## L-TS additions ###########
    
    # import random 
import random

# get random sentence from test set
randomLength = randint(1,len(Test_Clean))
randomSentence = Test_Clean[randomLength]

# tokenize this sentece into words (to find its length)

# import word tokenizers

from nltk import word_tokenize
from nltk import WhitespaceTokenizer
from nltk.tokenize import TweetTokenizer

tweet_wt = TweetTokenizer()
randomSentenceTokenized = tweet_wt.tokenize(randomSentence )
randomSentenceLength = len(randomSentenceTokenized)

randomVocabularySentence = random.sample(valid_vocabulary, randomSentenceLength)

# grams of vocabulary

unigram_counter = Counter()
bigram_counter = Counter()
trigram_counter = Counter()

unigram_counter.update([gram for gram in ngrams(valid_vocabulary, 1, pad_left=True, pad_right=True,
                                                   left_pad_symbol='<s>',right_pad_symbol='<e>') ])

bigram_counter.update([gram for gram in ngrams(valid_vocabulary, 2, pad_left=True, pad_right=True,
                                                   left_pad_symbol='<s>',right_pad_symbol='<e>') ])

trigram_counter.update([gram for gram in ngrams(valid_vocabulary, 3, pad_left=True, pad_right=True,
                                               left_pad_symbol='<s>',right_pad_symbol='<e>') ])
# initialise grams of Random Sentence from test set
unigram_counterRandomSentence = Counter()
bigram_counterRandomSentence = Counter()
trigram_counterRandomSentence = Counter()

# initialise grams of Random Sentence from Vocabulary
unigram_counterRandomVocabularySentence = Counter()
bigram_counterRandomVocabularySentence = Counter()
trigram_counterRandomVocabularySentence = Counter()     

# compute grams

# grams of test set random sentence

unigram_counterRandomSentence.update([gram for gram in ngrams(randomSentenceTokenized, 1, pad_left=True, pad_right=True,
                                                   left_pad_symbol='<s>',right_pad_symbol='<e>') ])

bigram_counterRandomSentence.update([gram for gram in ngrams(randomSentenceTokenized, 2, pad_left=True, pad_right=True,
                                                   left_pad_symbol='<s>',right_pad_symbol='<e>') ])

trigram_counterRandomSentence.update([gram for gram in ngrams(randomSentenceTokenized, 3, pad_left=True, pad_right=True,
                                               left_pad_symbol='<s>',right_pad_symbol='<e>') ])
# grams of vocabulary random sentence  
unigram_counterRandomVocabularySentence.update([gram for gram in ngrams(randomVocabularySentence, 1, pad_left=True, pad_right=True,
                                                   left_pad_symbol='<s>',right_pad_symbol='<e>') ])

bigram_counterRandomVocabularySentence.update([gram for gram in ngrams(randomVocabularySentence, 2, pad_left=True, pad_right=True,
                                                   left_pad_symbol='<s>',right_pad_symbol='<e>') ])

trigram_counterRandomVocabularySentence.update([gram for gram in ngrams(randomVocabularySentence, 3, pad_left=True, pad_right=True,
                                               left_pad_symbol='<s>',right_pad_symbol='<e>') ])
# probability estimations
    
# Bigrams

#We should fine-tune alpha on a held-out dataset
alpha = 0.01
#Calculate vocab size 
vocab_size = len(valid_vocabulary)

# initialise probabilities to 1
bigram_prob_total_random_sentence = 1
bigram_prob_total_random_vocabulary_sentence = 1

#Bigram prob + laplace smoothing

for bigram in bigram_counterRandomSentence:
    bigram_prob = (bigram_counter[(bigram[1], bigram[0])] +alpha) / (unigram_counter[(bigram[1],)] + alpha*vocab_size)
    bigram_prob_total_random_sentence*=bigram_prob
print("bigram_prob: {0:.3f} ".format(bigram_prob_total_random_sentence))
bigram_log_prob_random_sentence = math.log2(bigram_prob_total_random_sentence)
print("bigram_log_prob: {0:.3f}".format(bigram_log_prob_random_sentence)) 

for bigram in bigram_counterRandomVocabularySentence:   
    bigram_prob = (bigram_counter[(bigram[1], bigram[0])] +alpha) / (unigram_counter[(bigram[1],)] + alpha*vocab_size)
    bigram_prob_total_random_vocabulary_sentence*=bigram_prob
print("bigram_prob: {0:.3f} ".format(bigram_prob_total_random_vocabulary_sentence))
bigram_log_prob_random_vocabulary_sentence = math.log2(bigram_prob_total_random_sentence)
print("bigram_log_prob: {0:.3f}".format(bigram_log_prob_random_vocabulary_sentence))

# Trigrams

#We should fine-tune alpha on a held-out dataset
alpha = 0.01
#Calculate vocab size 
vocab_size = len(valid_vocabulary)

# initialise probabilities to 1
trigram_prob_total_random_sentence = 1
trigram_prob_total_random_vocabulary_sentence = 1

#Trigram prob + laplace smoothing

for trigram in trigram_counterRandomSentence:
    trigram_prob = (trigram_counter[(trigram[1], trigram[0])] +alpha) / (bigram_counter[(trigram[1],)] + alpha*vocab_size)
    trigram_prob_total_random_sentence*=trigram_prob
print("trigram_prob: {0:.3f} ".format(trigram_prob_total_random_sentence))
trigram_log_prob_random_sentence = math.log2(trigram_prob_total_random_sentence)
print("trigram_log_prob: {0:.3f}".format(trigram_log_prob_random_sentence)) 

for trigram in trigram_counterRandomVocabularySentence:   
    trigram_prob = (trigram_counter[(trigram[2],trigram[1], trigram[0])] +alpha) / (bigram_counter[(trigram[1], trigram[0])] + alpha*vocab_size)
    trigram_prob_total_random_vocabulary_sentence*=trigram_prob
print("trigram_prob: {0:.3f} ".format(trigram_prob_total_random_vocabulary_sentence))
trigram_log_prob_random_vocabulary_sentence = math.log2(trigram_prob_total_random_sentence)
print("trigram_log_prob: {0:.3f}".format(trigram_log_prob_random_vocabulary_sentence))

# Cross_entropy & perplexity

# Compute corpus cross_entropy & perplexity for bi-gram LM 

sentences_tokenized = []
for sent in Test_Clean:
    #sent_tok = whitespace_wt.tokenize(sent)
    sent_tok = tweet_wt.tokenize(sent)
    #sent_tok = word_tokenize(sent)
    sentences_tokenized.append(sent_tok)
    
sum_prob = 0
bigram_cnt = 0
for sent in sentences_tokenized:
    sent = ['<start>'] + sent + ['<end>']
    for idx in range(len(sent)):
        bigram_prob = (bigram_counter[(sent[idx-1], sent[idx])] +alpha) / (unigram_counter[(sent[idx-1],)] + alpha*vocab_size)
        sum_prob += math.log2(bigram_prob)
        bigram_cnt+=1

HC = -sum_prob / bigram_cnt
perpl = math.pow(2,HC)

print("Cross Entropy: {0:.3f}".format(HC))
print("perplexity: {0:.3f}".format(perpl))

# Compute corpus cross_entropy & perplexity for tri-gram LM 


sum_prob = 0
trigram_cnt = 0
for sent in sentences_tokenized:
    sent = ['<start1>'] + ['<start2>'] + sent + ['<end>'] + ['<end>']
    for idx in range(2,len(sent)):
        trigram_prob = (trigram_counter[(sent[idx-2],sent[idx-1], sent[idx])] +alpha) / (bigram_counter[(sent[idx-1],sent[idx])] + alpha*vocab_size)
        sum_prob += math.log2(trigram_prob)
        trigram_cnt+=1

HC = -sum_prob / trigram_cnt
perpl = math.pow(2,HC)
print("Cross Entropy: {0:.3f}".format(HC))
print("perplexity: {0:.3f}".format(perpl))
# Compute corpus cross_entropy & perplexity for interpoladed bi-gram & tri-gram LMs 

#We should fine-tune lamda on a held-out dataset
lamda = 0.9
sum_prob = 0
ngram_cnt = 0
for sent in sentences_tokenized:
    sent = ['<start1>'] + ['<start2>'] + sent + ['<end>'] + ['<end>']
    for idx in range(2,len(sent)):
        trigram_prob = (trigram_counter[(sent[idx-2],sent[idx-1], sent[idx])] +alpha) / (bigram_counter[(sent[idx-1],sent[idx])] + alpha*vocab_size)
        bigram_prob = (bigram_counter[(sent[idx-1], sent[idx])] +alpha) / (unigram_counter[(sent[idx-1],)] + alpha*vocab_size)

        sum_prob += (lamda * math.log2(trigram_prob)) +((1-lamda) * math.log2(bigram_prob))
        ngram_cnt+=1 

HC = -sum_prob / ngram_cnt
perpl = math.pow(2,HC)
print("Cross Entropy: {0:.3f}".format(HC))
print("perplexity: {0:.3f}".format(perpl))
    
    