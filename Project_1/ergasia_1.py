#######################################################################################################################
# Ergasia 1
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
from sklearn.model_selection import train_test_split
#
# from nltk import download
# download('punkt')
#
print("Imports Completed")
#######################################################################################################################
# Cockpit:
#######################################################################################################################

read_corpus = False
clean_corpus = False
create_datasets = False
do_AllWords = False
do_vocabulary = False
do_WordCounts = False
# -deprecated- shortcut_1 = False
do_vocabularies = False
do_oov = False
loading_point = True
demo_ngrams = False
sample_ngrams = False
show_bugs = False

#######################################################################################################################


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


def telos():
    """
    Ends execution with pre-defined message.
    """
    time.sleep(1)
    sys.exit("Requested Exit.")

#######################################################################################################################


"""
corpus_original : a string with the whole corpus text uncleansed.
corpus_clean : a list of cleansed strings, where each string is a cleansed corpus sentence,
                where only single spaces and no punctuation exists.
"""
# Deprecated: corpus_clean_string : a string with the whole corpus text cleansed, with the cleansing of corpus_clean

#######################################################################################################################
corpus_clean = []
corpus_original = ''
if read_corpus:
    abs_path = os.getcwd()
    path = abs_path + '/en/'
    sentences = []
    text = ''
    total = len(os.listdir(path))  # Total files
    count = 0

    for file in os.listdir(path):
        f = open(path + file, 'r', encoding="utf-8")
        file_text = f.read()
        corpus_original = corpus_original + file_text
        f.close()

        regex = re.compile('[%s]' % re.escape(string.punctuation))
        file_sentences = [regex.sub('', sent).strip() for sent in sent_tokenize(clean_text(file_text))]

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

        corpus_clean = corpus_clean + file_sentences
        count += 1

        clear_output(wait=True)
        print('File ' + file + ' finished. Completed ' + str(round(count * 100 / total, 2)) + '%')

    # Save the basic objects:
    with open('corpus_original', 'wb') as f:
        pickle.dump(corpus_original, f)
    with open('corpus_clean', 'wb') as f:
        pickle.dump(corpus_clean, f)
# telos()
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

# DEPRECATED since we started using the validation set for the vocabulary
#
# Apply the exact same cleansing for the whole corpus text, and keep the final result a text.
# corpus_clean_string = None
# if clean_corpus:
#
#     with open('corpus_original', 'rb') as f:
#         corpus_original = pickle.load(f)
#
#     regex = re.compile('[%s]' % re.escape(string.punctuation))
#     corpus_clean_string = regex.sub('', clean_text(corpus_original))
#
#     print('-------------------------')
#     print('corpus_clean_string created.')
#     print('-------------------------')
#
#     with open('corpus_clean_string', 'wb') as f:
#         pickle.dump(corpus_clean_string, f)
#
#     del corpus_original
#     gc.collect()
# telos()

#######################################################################################################################
# Creating the datasets:

if create_datasets:
    with open('corpus_clean', 'rb') as f:
        corpus_clean = pickle.load(f)

    total = len(corpus_clean)
    sets = list(range(0, total))

    # 60% for train
    training_idx, tuning_idx = train_test_split(sets, train_size=.6, random_state=2019)

    # 20%,10%,10% for validation(development), test1 and test2 datasets.
    validation_idx, test_idx = train_test_split(tuning_idx, train_size=.5, random_state=2019)
    test1_idx, test2_idx = train_test_split(test_idx, train_size=.5, random_state=2019)

    training_set = [corpus_clean[i] for i in training_idx]
    validation_set = [corpus_clean[i] for i in validation_idx]
    test1_set = [corpus_clean[i] for i in test1_idx]
    test2_set = [corpus_clean[i] for i in test2_idx]

    del training_idx, validation_idx, tuning_idx, test1_idx, test2_idx
    gc.collect()

    print('Training Size: ', len(training_set))
    print('Validation Size: ', len(validation_set))
    print('Test1 Size: ', len(test1_set))
    print('Test2 Size: ', len(test2_set))

    with open('training_set', 'wb') as f:
        pickle.dump(training_set, f)
    with open('validation_set', 'wb') as f:
        pickle.dump(validation_set, f)
    with open('test1_set', 'wb') as f:
        pickle.dump(test1_set, f)
    with open('test2_set', 'wb') as f:
        pickle.dump(test2_set, f)
# telos()

#######################################################################################################################

AllWords = []
vocabulary = set()
WordCounts = Counter()
if do_AllWords:

    with open('training_set', 'rb') as f:
        training_set = pickle.load(f)

    total = len(training_set)
    # total = 100 # demo debuging size
    for i in range(0, total):
        sentence_words = word_tokenize(training_set[i])
        # SOS: This was an order of magnitude faster than doing AllWords = AllWords + sentence_words
        # readmore: https://stackoverflow.com/questions/2022031/python-append-vs-operator-on-lists-why-do-these-give-different-results
        AllWords.append(sentence_words)
        print("Sentence tokenized:", i)
        # print(sentence_words)
        # if(i>10):
        #     pprint(AllWords)
        #     telos()
        #     break

    with open('AllWords', 'wb') as f:
        pickle.dump(AllWords, f)
    # pprint(AllWords)
    print('-------------------------')
    print('Words Tokenized.')
    print('-------------------------')
# telos()

if do_vocabulary:

    with open('AllWords', 'rb') as f:
        AllWords = pickle.load(f)

    count = 0
    AllWords_oneList = []
    for lista in AllWords:
        count = count + 1
        AllWords_oneList += lista
        print("Sentence:", count)

    vocabulary = set(AllWords_oneList)
    with open('vocabulary', 'wb') as f:
        pickle.dump(vocabulary, f)
    pprint(vocabulary)

    print('Vocabulary Created.')
    print('-------------------------')
# telos()

if do_WordCounts:

    with open('AllWords', 'rb') as f:
        AllWords = pickle.load(f)

    count = 0
    AllWords_oneList = []
    for lista in AllWords:
        count = count + 1
        AllWords_oneList += lista
        print("Sentence:", count)

    WordCounts.update(AllWords_oneList)
    # pprint(WordCounts)

    print('WordCounts Calculated.')
    print('-------------------------')

    with open('WordCounts', 'wb') as f:
        pickle.dump(WordCounts, f)

    del AllWords
    gc.collect()
# telos()

#######################################################################################################################

# corpus_clean = None
# WordCounts = None
# vocabulary = None
# if shortcut_1:
#
#     # Load objects
#
#     with open('corpus_clean', 'rb') as f:
#         corpus_clean = pickle.load(f)
#
#     with open('vocabulary', 'rb') as f:
#         vocabulary = pickle.load(f)
#
#     with open('WordCounts', 'rb') as f:
#         WordCounts = pickle.load(f)
#
#     with open('AllWords', 'rb') as f:
#         AllWords = pickle.load(f)

#######################################################################################################################
# Ignore low frequency words

valid_vocabulary = None
invalid_vocabulary = None
if do_vocabularies:

    with open('vocabulary', 'rb') as f:
        vocabulary = pickle.load(f)

    with open('WordCounts', 'rb') as f:
        WordCounts = pickle.load(f)

    valid_vocabulary = [k for k, v in WordCounts.items() if v > 10]
    invalid_vocabulary = [k for k, v in WordCounts.items() if v <= 10]
    print("valid voc", len(valid_vocabulary))
    print("invalid voc", len(invalid_vocabulary))

    with open('valid_vocabulary', 'wb') as f:
        pickle.dump(valid_vocabulary, f)

    with open('invalid_vocabulary', 'wb') as f:
        pickle.dump(invalid_vocabulary, f)
# telos()

#######################################################################################################################
# Replace OOV words in sentences


def split_sentence(sentence):
    # This is the regex of where to tokenize.
    # If \w+ was \w, it would tokenize on every single letter.
    # However \w+ means "any repetition of letters", so it will tokenize on words, just as we want it to do.
    PATTERN = '\w+|\(|\)|\.|\,'
    tokenizer = RegexpTokenizer(pattern=PATTERN) 
    return tokenizer.tokenize(sentence)


# otan ekana antikatastash twn UNK to ekana mono se 10k records
# kai ekei kata lathos to ekana lista apo listes apo words..

if do_oov:  # TODO: This section worked, but needs refactoring in order to use valid_vocabulary as dict and not as list.

    with open('corpus_clean', 'rb') as f:
        corpus_clean = pickle.load(f)

    with open('valid_vocabulary', 'rb') as f:
        valid_vocabulary = pickle.load(f)

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
        # dummy_count = dummy_count + 1
        # if 1000 < dummy_count:
        #     break
    # Have it here, in order to not to forget to save after the big computation burden.
    with open('corpus_clean_no_OOV', 'wb') as f:
        pickle.dump(corpus_clean, f)
# telos()

#######################################################################################################################
"""
corpus_clean_no_OOV : A list of lists of string tokens = a list of lists of words = a list of sentences, 
                        where each sentence is a list of string words.
"""
#######################################################################################################################

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

valid_vocabulary = None
invalid_vocabulary = None
corpus_clean_no_OOV = None
C = None
if loading_point:
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
# Theory explained:
#######################################################################################################################
#
#  Language model =
#   P(w_1_k):=
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
#  Language model =
#   P(w_i_k):=
#       for bigrams:= P(w_i|w_i-1)*P(w_i+1|w_i)*..*P(w_k|w_k-1) = ...
#       for trigrams:= P(w_i|w_i-2,w_i-1)*P(w_i+1|w_i-1,w_i)*P(w_i+2|w_i,w_i+1)*..*P(w_k|w_k-2,w_k-1) = ...
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

# TODO-list:
#
# -1- functions for spliting a sentence into all serial unigrams, bigrams & trigrams needed for the prob formulas (done)
# -2- functions for bigram prob (done)
# -3- functions for trigram prob (done)
# -4- functions for Linear interpolation (done)
# -5- functions for combining the above probs into making the P(t_i_k), aka the Language models (done)
# -6- function for edit distance (cancelled, out of scope)
# -7- function for most probable sentence (cancelled, out of scope)
# -8- function for model cross-entropy & preplexity (in same function) (slides 29,30) (done)

#######################################################################################################################
# -1- function for spliting a sentence into all serial bigrams and trigrams needed for the prob formulas (done)

# THESE 3 ARE A MUST:
unigram_counter = Counter()
bigram_counter = Counter()
trigram_counter = Counter()
# Reminder:
# C = len(AllWords)


def split_into_unigrams(sentence):
    if sentence.__class__ == str:
        print(sentence)
        print("Error in corpus sentence (unigrams func)!!")
        telos()
    ngs = [gram for gram in ngrams(sentence, 1)]
    unigram_counter.update(ngs)
    return ngs


def split_into_bigrams(sentence, pad=True, s="start", e="end"):
    if sentence.__class__ == str:
        print("Error in corpus sentence (bigrams func)!!")
        print(sentence)
        telos()
    list = [s]+sentence+[e] if pad else sentence
    # print("debug_padding:",s,sentence,e)
    ngs = [gram for gram in ngrams(list, 2)]
    bigram_counter.update(ngs)
    return ngs


def split_into_trigrams(sentence, pad=True, s1= "start1", s2= 'start2', e='end'):
    if sentence.__class__ == str:
        print(sentence)
        print("Error in corpus sentence (trigrams func)!!")
        telos()
    list = [s1, s2]+sentence+[e] if pad else sentence
    ngs = [gram for gram in ngrams(list, 3)]
    trigram_counter.update(ngs)
    return ngs

# print("\n Printing section 1:")
# sentence = corpus_clean_no_OOV[94755]
# print(sentence)
# print(split_into_bigrams(sentence))
# print(split_into_trigrams(sentence))
# telos()

#######################################################################################################################
# -2- functions for unigram & bigram prob (done)


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


# print("\n Printing section 2:")
# print("\n------------------------------")
# sentence = corpus_clean_no_OOV[94755]
# vocab_size = len(valid_vocabulary)
# print_sentence_unigram_probs(sentence, vocab_size, C, a=0.01)
# print("\n------------------------------")
# sentence = corpus_clean_no_OOV[94755]
# vocab_size = len(valid_vocabulary)
# print_sentence_bigram_probs(sentence, vocab_size, a=0.01)
# print("\n------------------------------")
# sentence = corpus_clean_no_OOV[0] + ['next','item']
# vocab_size = len(valid_vocabulary)
# print_sentence_bigram_probs(sentence, vocab_size, a=0.01)
# print("\n------------------------------")
# telos()

#######################################################################################################################
# -3- functions for trigram prob (done)


def trigram_prob(ngram, vocab_size, a=0.01):
    x = ngram[0]
    y = ngram[1]
    z = ngram[2]
    return (trigram_counter[(x,y,z)] + a) / (bigram_counter[(x,y,)] + a*vocab_size)


def print_sentence_trigram_probs(sentence, vocab_size, a=0.01):
    for trigram in split_into_trigrams(sentence):
        print(trigram, np.round(100*trigram_prob(trigram, vocab_size, a),2), " %")


# print("\n Printing section 3:")
# print("\n------------------------------")
# sentence = corpus_clean_no_OOV[94755]
# vocab_size = len(valid_vocabulary)
# print_sentence_trigram_probs(sentence, vocab_size, a=0.01)
# print("\n------------------------------")
# sentence = corpus_clean_no_OOV[0] + ['next','item','is']
# vocab_size = len(valid_vocabulary)
# print_sentence_trigram_probs(sentence, vocab_size, a=0.01)
# print("\n------------------------------")
# telos()

#######################################################################################################################
# -4- functions for Linear interpolation (done)


def bigram_linear_interpolation_probs(sentence, vocab_size, C, a=0.01, l = 0.7):
    """
    Due to the nature of the calculations, I return all bigram linear interpolations at once for the sentence
    :param sentence:
    :param vocab_size:
    :param a:
    :return:
    """
    if (l > 1) or (l < 0):
        print("Error: Lambas should be 0 <= l <= 1.")
        return False

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
    if (l1+l2 > 1) or (l1 < 0) or (l2 < 0):
        print("Error: Lambas should be 0 <= l1,l2,(l1+l2) <= 1.")
        return False

    trigrams = split_into_trigrams(sentence)
    trigram_linear_interpolations = []
    for trigram in trigrams:
        bigram = (trigram[0], trigram[1],)
        unigram = bigram[0]
        linear_interpolation = l1 * trigram_prob(trigram, vocab_size, a) + l2 * bigram_prob(bigram, vocab_size, a) +\
                               (1 - l1 -l2) * unigram_prob(unigram, vocab_size, C, a)
        trigram_linear_interpolations.append([trigram, np.round(linear_interpolation,4)])
    return trigram_linear_interpolations


# print("\n Printing section 4:")
# print("\n------------------------------")
# sentence = corpus_clean_no_OOV[94755]
# vocab_size = len(valid_vocabulary)
# pprint(bigram_linear_interpolation_probs(sentence, vocab_size, C))
# print("\n------------------------------")
# sentence = corpus_clean_no_OOV[94755]
# vocab_size = len(valid_vocabulary)
# pprint(trigram_linear_interpolation_probs(sentence, vocab_size, C))
# print("\n------------------------------")
# telos()

#######################################################################################################################
# -5- functions for combining the above probs into making the P(t_i_k), aka the Language models

# SOS : sums of logs instead of mults !!! Then pow, so that low numbers wont get zero.

def unigram_language_model(sentence, vocab_size, C, a=0.01):
    language_model = 0  # neutral value
    for unigram in split_into_unigrams(sentence):
        language_model += math.log2(unigram_prob(unigram, vocab_size, C, a))
    return math.pow(2, language_model)


def bigram_language_model(sentence, vocab_size, a = 0.01):
    language_model = 0 # neutral value
    for bigram in split_into_bigrams(sentence):
        language_model += math.log2(bigram_prob(bigram, vocab_size, a))
    return math.pow(2, language_model)


def trigram_language_model(sentence, vocab_size, a = 0.01):
    language_model = 0 # neutral value
    for trigram in split_into_trigrams(sentence):
        language_model += math.log2(trigram_prob(trigram, vocab_size, a))
    return math.pow(2, language_model)


def bigram_linear_interpolation_language_model(sentence, vocab_size, C, a=0.01, l = 0.7):
    language_model = 0 # neutral value
    for pair in bigram_linear_interpolation_probs(sentence, vocab_size, C, a, l):
        prob = pair[1]
        language_model += math.log2(prob)
    return math.pow(2, language_model)


def trigram_linear_interpolation_language_model(sentence, vocab_size, C, a=0.01, l1 = 0.7, l2 = 0.2):
    language_model = 0 # neutral value
    for pair in trigram_linear_interpolation_probs(sentence, vocab_size, C, a, l1, l2):
        prob = pair[1]
        language_model += math.log2(prob)
    return math.pow(2, language_model)


print("\n Printing section 5:")
print("\n------------------------------")
print("Language models for single sentence")
vocab_size = len(valid_vocabulary)
# sentence = corpus_clean_no_OOV[0]
for sentence in [corpus_clean_no_OOV[94755]]:
    print(np.round(100 * unigram_language_model(sentence, vocab_size, C, a=1), 2), " %")
    print(np.round(100 * bigram_language_model(sentence, vocab_size, a=1), 2), " %")
    print(np.round(100 * trigram_language_model(sentence, vocab_size, a=1), 2), " %")
    print(np.round(100 * bigram_linear_interpolation_language_model(sentence, vocab_size, C, a=1), 2), " %")
    print(np.round(100 * trigram_linear_interpolation_language_model(sentence, vocab_size, C, a=1), 2), " %")
    print("\n------------------------------")
# telos()

#######################################################################################################################
# -8- function for model cross-entropy & preplexity (in same function) (slides 29,30) (almost done)


def unigram_crossentropy_perplexity(vocab_size, C, a=0.01):
    sum_prob = 0
    unigram_cnt = 0
    sentcount = -1
    for sentence in corpus_clean_no_OOV:
    # for sentence in [corpus_clean_no_OOV[94754],corpus_clean_no_OOV[94755],corpus_clean_no_OOV[94756]]:
        sentcount += 1
        if (sentence == None) or (sentence == []) or (sentence == '') or (sentence in ['‘', '•', '–', '–']):
            # print("Erroneous Sentence", sentcount,"start:", sentence, ":end")
            continue
        # These lines below are very similar to the internals of language models
        for unigram in split_into_unigrams(sentence):
            sum_prob += math.log2(unigram_prob(unigram, vocab_size, C, a))
            unigram_cnt += 1
    HC = -sum_prob / unigram_cnt
    perpl = math.pow(2, HC)
    print("Cross Entropy: {0:.3f}".format(HC))
    print("perplexity: {0:.3f}".format(perpl))


def bigram_crossentropy_perplexity(vocab_size, a=0.01):
    sum_prob = 0
    bigram_cnt = 0
    sentcount = -1
    for sentence in corpus_clean_no_OOV:
    # for sentence in [corpus_clean_no_OOV[94754],corpus_clean_no_OOV[94755],corpus_clean_no_OOV[94756]]:
        sentcount += 1
        if (sentence == None) or (sentence == []) or (sentence == '') or (sentence in ['‘', '•', '–', '–']):
            # print("Erroneous Sentence", sentcount,"start:", sentence, ":end")
            continue
        # These lines below are very similar to the internals of language models
        for bigram in split_into_bigrams(sentence):
            sum_prob += math.log2(bigram_prob(bigram, vocab_size, a))
            bigram_cnt += 1
    HC = -sum_prob / bigram_cnt
    perpl = math.pow(2, HC)
    print("Cross Entropy: {0:.3f}".format(HC))
    print("perplexity: {0:.3f}".format(perpl))


def trigram_crossentropy_perplexity(vocab_size, a=0.01):
    sum_prob = 0
    trigram_cnt = 0
    sentcount = -1
    for sentence in corpus_clean_no_OOV:
    # for sentence in [corpus_clean_no_OOV[94755]]:
        sentcount += 1
        if (sentence == None) or (sentence == []) or (sentence == '') or (sentence in ['‘','•','–','–']):
            # print("Erroneous Sentence",sentcount,"start:", sentence, ":end")
            continue
        # These lines below are very similar to the internals of language models
        for trigram in split_into_trigrams(sentence):
            prob = trigram_prob(trigram, vocab_size, a)
            sum_prob += math.log2(prob)
            trigram_cnt += 1
    HC = - sum_prob / trigram_cnt
    perpl = math.pow(2, HC)
    print("Cross Entropy: {0:.3f}".format(HC))
    print("perplexity: {0:.3f}".format(perpl))

#######################################################################################################################
# -------------
# Theory Again
# -------------
#
#  Language model =
#   P(w_i_k):=
#       for bigrams:= P(w_i|w_i-1)*P(w_i+1|w_i)*..*P(w_k|w_k-1) = ...
#       for trigrams:= P(w_i|w_i-2,w_i-1)*P(w_i+1|w_i-1,w_i)*P(w_i+2|w_i,w_i+1)*..*P(w_k|w_k-2,w_k-1) = ...
#
# Language entropy = - (1/N) * log2(P(w_1_N)) ,
#   where N is:
#       -- the number of unigrams or bigrams or trigrams in the whole corpus.
# So log2(P(w_1_N)) == Language model P(w_1_N) for the whole corpus as a single sentence.
# Language entropy = - (1/N) * log2(Language model at the corpus as test)
#
# So Language entropy can be calculated with unigram,bigram,trigram,linear_bigram & linear_trigram language models.
#
# So N changes depending on what language model you use to calculate the language entropy.
#######################################################################################################################

# print(corpus_clean_no_OOV[53959])
# print(corpus_clean_no_OOV[53966])
# pprint(corpus_clean_no_OOV[94755])

print("\n Printing section 8:")
print("\n------------------------------")
print("Crossentropies & perplexities of models")
vocab_size = len(valid_vocabulary)
unigram_crossentropy_perplexity(vocab_size, C, a=1)
bigram_crossentropy_perplexity(vocab_size, a=1)
trigram_crossentropy_perplexity(vocab_size, a=1)
print("\n------------------------------")


# TODO :
# -1- Initial replacements contain mistakes, thus the need for edge conditions in the functions
# -2- Perplexity type seems wrong

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# FIN
#######################################################################################################################
#
#######################################################################################################################
# Debug session:
if show_bugs:
    sntcnt = -1
    for sentence in corpus_clean_no_OOV:
        sntcnt +=1
        if (sentence.__class__ == str) and sentence !='':
            print("start",sntcnt,sentence,"end")
    # Problematic sentences: (should have been removed at the beginning:
    # The numbers are the sentence index in the corpus_clean_no_OOV:
    #
    # start 94754 ‘ end
    # start 118178 • end
    # start 370802 – end
    # start 1007142 – end
# telos()
#######################################################################################################################
#
#######################################################################################################################
# Demo Runs:
#######################################################################################################################
# demo_ngrams = False
# if demo_ngrams:
#     print(corpus_clean_no_OOV.__class__)
#     print(corpus_clean_no_OOV[0])
#     print(corpus_clean_no_OOV[0].__class__)
#     print(corpus_clean_no_OOV[0][0].__class__)
#
#     print(AllWords.__class__)
#     print(AllWords[0])
#     print(AllWords[0].__class__)
#     print(AllWords[0:10])
#
#     gc.collect()
#
#     # Single sentence, for testing corpus_clean_no_OOV:
#     unigram_counter = Counter()
#     unigram_counter.update([gram for gram in ngrams(corpus_clean_no_OOV[0], 1, pad_left=True, pad_right=True,
#                                                        left_pad_symbol='<s>',right_pad_symbol='<e>') ])
#     pprint(corpus_clean_no_OOV[0])
#     pprint(unigram_counter)
#
#     # Single sentence for testing AllWords:
#     unigram_counter = Counter()
#     unigram_counter.update([gram for gram in ngrams(AllWords[0:10], 1, pad_left=True, pad_right=True,
#                                                        left_pad_symbol='<s>',right_pad_symbol='<e>') ])
#     pprint(AllWords[0:10])
#     pprint(unigram_counter)
#######################################################################################################################
# unigram_counter = Counter()
# bigram_counter = Counter()
# trigram_counter = Counter()
# sample_ngrams = False
# if sample_ngrams:
#     print("Just started the sample ngram-ing")
#     sample = corpus_clean_no_OOV[0:1000]
#
#     for sent in sample:
#         unigram_counter.update([gram for gram in ngrams(sent, 1, pad_left=True, pad_right=True,
#                                                        left_pad_symbol='<s>',right_pad_symbol='<e>') ])
#     #pprint(unigram_counter)
#
#     for sent in sample:
#         bigram_counter.update([gram for gram in ngrams(sent, 2, pad_left=True, pad_right=True,
#                                                        left_pad_symbol='<s>',right_pad_symbol='<e>') ])
#     # pprint(bigram_counter)
#
#     sample = corpus_clean_no_OOV[0:1000]
#     for sent in sample:
#         trigram_counter.update([gram for gram in ngrams(sent, 3, pad_left=True, pad_right=True,
#                                                        left_pad_symbol='<s>',right_pad_symbol='<e>') ])
#     # pprint(trigram_counter)
#     print("Just ended the sample ngram-ing")
#######################################################################################################################
