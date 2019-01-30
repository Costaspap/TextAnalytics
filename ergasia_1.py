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
print("Imports Completed")
import math
download('punkt')

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

if shortcut_2:
    # Load objects

    with open('invalid_vocabulary', 'rb') as f:
        invalid_vocabulary = pickle.load(f)

    with open('valid_vocabulary', 'rb') as f:
        valid_vocabulary = pickle.load(f)

    with open('AllWords', 'rb') as f:
        AllWords = pickle.load(f)

    with open('corpus_clean_no_OOV', 'rb') as f:
        corpus_clean_no_OOV = pickle.load(f)

#######################################################################################################################

demo_ngrams = True
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

#  Language model =
#   P(w_i_k):=
#   for bigrams:= P(w_1|start)*P(w_2|w_1)*..*P(w_k|w_k-1) =
#       where P(w_k|w_k-1) = [c(w_k-1,w_k) + a] / [c(w_k-1 + a*|V|]
#   for trigrams:= P(w_1|start1,start2)*P(w_2|start2,w_1)*P(w_3|w_1,w_2)*..*P(w_k|w_k-2,w_k-1)
#       where P(w_k|w_k-2,w_k-1) = [c(w_k-2,w_k-1,w_k) + a] / [c(w_k-2,c_k-1) + a * |V|]
#
# P(w_1_k|t_1_k) = ?? =
#       Π_i=1_k{P(w_i|t_i)} =
#       Π_i=1_k{[c(t_i,w_i) + a] / [c(t_i + a*|V|]} ??? which a is here ???

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
#######################################################################################################################
#######################################################################################################################
# FIN
#######################################################################################################################

