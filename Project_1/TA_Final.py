#######################################################################################################################
# Imports
#######################################################################################################################


from nltk import sent_tokenize, word_tokenize
from IPython.display import clear_output
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams


from pprint import pprint
import time, sys, math, re, os, string, gc, pickle
import numpy as np
from sklearn.model_selection import train_test_split


########## Select Run Mode. If load, it loads the preprocessed files. If create, it creates everything from scratch #######


#mode = 'Create'
mode = 'Load'

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
        # TODO : Kosta, to keno edw xreiazetai?

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
    with open('WordCounts', 'wb') as f:
            pickle.dump(WordCounts, f)



if mode == 'Load':

    with open('vocabulary', 'rb') as f:
        vocabulary = pickle.load(f)

    with open('AllWords', 'rb') as f:
        AllWords = pickle.load(f)

    with open('WordCounts', 'rb') as f:
        WordCounts = pickle.load(f)


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
    
    # TODO : Costa, den ta kaneis auta me function gia na meiwthei to space?
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

# TODO : Costa , giati na metrame xwria ta sentences? An se enoxlei to global scope twn counters,
#   de nomizw oti einai logos anhsyxias. Epishs an ta bgaleis apo ta functions, metras kai tis skartes sentences.
def split_into_unigrams(sentence):
    if sentence.__class__ == str:
        print(sentence)
        print("Error in corpus sentence (unigrams func)!!")
        sys.exit("Requested Exit.")
    ngs = [gram for gram in ngrams(sentence, 1)]
    unigram_counter.update(ngs)
    return ngs


def split_into_bigrams(sentence, pad=True, s="start", e="end"):
    if sentence.__class__ == str:
        print("Error in corpus sentence (bigrams func)!!")
        print(sentence)
        sys.exit("Requested Exit.")
    list = [s]+sentence+[e] if pad else sentence
    # print("debug_padding:",s,sentence,e)
    ngs = [gram for gram in ngrams(list, 2)]
    bigram_counter.update(ngs)
    return ngs


def split_into_trigrams(sentence, pad=True, s1= "start1", s2= 'start2', e='end'):
    if sentence.__class__ == str:
        print(sentence)
        print("Error in corpus sentence (trigrams func)!!")
        sys.exit("Requested Exit.")
    list = [s1, s2]+sentence+[e] if pad else sentence
    ngs = [gram for gram in ngrams(list, 3)]
    trigram_counter.update(ngs)
    return ngs


# For Ram optimization I dont create the lists containing the ngrams, I count them directly
if mode == 'Create':

    unigrams_training_counter = {}
    bigrams_training_counter = {}
    trigrams_training_counter = {}
    
    ###### Counters ########
    # TODO : Costa tha proteina auto na fygei
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


######################################################################################################################
## ii)  Check the log-probabilities that your trained models return when given (correct) sentence                   ## 
##      from the test subset vs. (incorrect) sentences of the same length (in words) consisting of                  ##
##      randomly selected vocabulary words.                                                                         ##
######################################################################################################################

# TODO : Costa, pws ginetai na mhn yparxei kapoio n-gram? Fovamai mhn einai to tackling edw
# san na kryvoume kapoio lathos katw apo to xali

def unigram_prob(ngram, vocab_size, C, a=0.01):
    x = ngram[0]
    
    # Tackle cases where the gram doesn't exist in the training counter
    try:
        out = (unigrams_training_counter[(x,)] + a) / (C + a*vocab_size)
    except:
        out = a / (C + a*vocab_size)
    return out


def bigram_prob(ngram, vocab_size, a=0.01):
    x = ngram[0]
    y = ngram[1]
    
    if (x,y) not in bigrams_training_counter:
        out = a
    else:
        out = (bigrams_training_counter[(x,y)] + a)

    # Tackle cases where the gram doesn't exist in the training counter
    if (x,) not in unigrams_training_counter:
        out *= 1 / (a*vocab_size)
    else:
        out *=  1 / (unigrams_training_counter[(x,)] + a*vocab_size)
        
    return out


def trigram_prob(ngram, vocab_size, a=0.01):
    x = ngram[0]
    y = ngram[1]
    z = ngram[2]
    
    if (x,y,z) not in trigrams_training_counter:
        out = a
    else:
        out = (trigrams_training_counter[(x,y,z)] + a)

    # Tackle cases where the gram doesn't exist in the training counter
    if (x,y,) not in bigrams_training_counter:
        out *= 1 / (a*vocab_size)
    else:
        out *=  1 / (bigrams_training_counter[(x,y,)] + a*vocab_size)
        
    return out






def print_sentence_unigram_probs(sentence, vocab_size, C, a=0.01):
    sum_prob = 0
    for unigram in split_into_unigrams(sentence):
        sum_prob += math.log2(unigram_prob(unigram, vocab_size, C, a))
    print('Unigram log Prob of sentence: ',str(sum_prob))

def print_sentence_bigram_probs(sentence, vocab_size, a=0.01):
    sum_prob = 0
    for bigram in split_into_bigrams(sentence):
        sum_prob += math.log2(bigram_prob(bigram, vocab_size, a))
    print('Bigram log Prob of sentence: ',str(sum_prob))

def print_sentence_trigram_probs(sentence, vocab_size, a=0.01):
    sum_prob = 0
    for trigram in split_into_trigrams(sentence):
        sum_prob += math.log2(trigram_prob(trigram, vocab_size, a))
    print('Trigram log Prob of sentence: ',str(sum_prob))


## Test 10 random sentences in the test1 set against 10 sentences randomly created from the vocabulary

np.random.seed(666)

for i in range(0,10):
    
    # Get random normal sentence
    print('-----------------------------------------------------------------------------')
    print('Test ',str(i))
    print('Normal sentence results:')
    sentence_idx = np.random.randint(low = 0, high = len(test1_set))
    
    valid_sentence = test1_set[sentence_idx]
    
    print_sentence_unigram_probs(valid_sentence,V,C)
    print_sentence_bigram_probs(valid_sentence,V)
    print_sentence_trigram_probs(valid_sentence,V)

## VS a randomly created non-sense sentence
    print()
    print('Invalid sentence results:')
    
    random_sent_idx = np.random.randint(low = 0, high = len(vocabulary), size = len(valid_sentence)) 
    
    invalid_sentence = [vocabulary[idx] for idx in random_sent_idx]
    print_sentence_unigram_probs(invalid_sentence,V,C)
    print_sentence_bigram_probs(invalid_sentence,V)
    print_sentence_trigram_probs(invalid_sentence,V)
    print('-----------------------------------------------------------------------------')





## (iii) Estimate the language cross-entropy and perplexity of your models on the test subset of
## the corpus, treating the entire test subset as a single sequence, with *start* (or *start1*,
## *start2*) at the beginning of each sentence, and *end* at the end of each sentence. Do not
## include probabilities of the form P(*start*|…) (or P(*start1*|…) or P(*start2*|…)) in the
## computation of perplexity, but include probabilities of the form P(*end*|…).


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


# functions for combining the above probs into making the P(t_i_k), aka the Language models
# SOS : sums of logs instead of mults !!! Then pow, so that low numbers wont get zero.


def unigram_language_model(sentence, vocab_size, C, a=0.01):
    language_model = 0  # neutral value
    for unigram in split_into_unigrams(sentence):
        try: # Skip zero probabilities (error: "ValueError: math domain error")
            language_model += math.log2(unigram_prob(unigram, vocab_size, C, a))
        except Exception:
            continue
    return math.pow(2, language_model)


def bigram_language_model(sentence, vocab_size, a = 0.01):
    language_model = 0 # neutral value
    for bigram in split_into_bigrams(sentence):
        try: # Skip zero probabilities (error: "ValueError: math domain error")
            language_model += math.log2(bigram_prob(bigram, vocab_size, a))
        except Exception:
            continue
    return math.pow(2, language_model)


def trigram_language_model(sentence, vocab_size, a = 0.01):
    language_model = 0 # neutral value
    for trigram in split_into_trigrams(sentence):
        try: # Skip zero probabilities (error: "ValueError: math domain error")
            language_model += math.log2(trigram_prob(trigram, vocab_size, a))
        except Exception:
            continue
    return math.pow(2, language_model)


def bigram_linear_interpolation_language_model(sentence, vocab_size, C, a=0.01, l = 0.7):
    language_model = 0 # neutral value
    for pair in bigram_linear_interpolation_probs(sentence, vocab_size, C, a, l):
        prob = pair[1]
        try: # Skip zero probabilities (error: "ValueError: math domain error")
            language_model += math.log2(prob)
        except Exception:
            continue
    return math.pow(2, language_model)


def trigram_linear_interpolation_language_model(sentence, vocab_size, C, a=0.01, l1 = 0.7, l2 = 0.2):
    language_model = 0 # neutral value
    for pair in trigram_linear_interpolation_probs(sentence, vocab_size, C, a, l1, l2):
        prob = pair[1]
        try: # Skip zero probabilities (error: "ValueError: math domain error")
            language_model += math.log2(prob)
        except Exception:
            continue
    return math.pow(2, language_model)

# function for model cross-entropy & preplexity (in same function) (slides 29,30) (almost done)


def crossentropy_perplexity(dataset, ngram_type, vocab_size, C, a=0.01, l=0.7, l1=0.7, l2=0.2):
    sum_prob = 0
    ngram_cnt = 0
    sentcount = -1
    for sentence in dataset:
    # for sentence in [corpus_clean_no_OOV[94754],corpus_clean_no_OOV[94755],corpus_clean_no_OOV[94756]]:
        sentcount += 1
        if (sentence == None) or (sentence == []) or (sentence == '') or (sentence in ['‘', '•', '–', '–']):
            # print("Erroneous Sentence", sentcount,"start:", sentence, ":end")
            continue
        # These lines below are very similar to the internals of language models
        if ngram_type == 'unigram':
            for ngram in split_into_unigrams(sentence):
                try:
                    sum_prob += math.log2(unigram_prob(ngram, vocab_size, C, a))
                except Exception:
                    ""
                ngram_cnt += 1
        elif ngram_type == 'bigram':
            for ngram in split_into_bigrams(sentence):
                try:
                    sum_prob += math.log2(bigram_prob(ngram, vocab_size, a))
                except Exception:
                    ""
                ngram_cnt += 1
        elif ngram_type == 'trigram':
            for ngram in split_into_trigrams(sentence):
                try:
                    sum_prob += math.log2(trigram_prob(ngram, vocab_size, a))
                except Exception:
                    ""
                ngram_cnt += 1
        elif ngram_type == 'lin_pol_bi':
            try:
                sum_prob += math.log2(bigram_linear_interpolation_language_model(sentence, vocab_size, C, a, l))
            except Exception:
                ""
            for ngram in split_into_bigrams(sentence):
                ngram_cnt += 1
        elif ngram_type == 'lin_pol_tri':
            try:
                sum_prob += math.log2(trigram_linear_interpolation_language_model(sentence, vocab_size, C, a, l1, l2))
            except Exception:
                ""
            for ngram in split_into_trigrams(sentence):
                ngram_cnt += 1
    HC = -sum_prob / ngram_cnt
    perpl = math.pow(2, HC)
    print("Cross Entropy: {0:.3f}".format(HC))
    print("perplexity: {0:.3f}".format(perpl))

'''
Compute corpus cross_entropy 
& perplexity for interpoladed bi-gram
& tri-gram LMs 
'''


def calculate_metrics(dataset,lamda = 0.9):
#We should fine-tune lamda on a held-out dataset

    sum_prob = 0
    ngram_cnt = 0
    for sent in dataset:
        sent = ['<s>'] + ['<s>'] + sent + ['<e>'] + ['<e>']
        for idx in range(2,len(sent)):
            tr_prob = trigram_prob([sent[idx-2],sent[idx-1], sent[idx]],V)
            b_prob = bigram_prob([sent[idx-1], sent[idx]],V)
            
            if sent[idx-2] != ['<s>'] and sent[idx-1] != ['<s>']:
                sum_prob += (lamda * math.log2(tr_prob)) +((1-lamda) * math.log2(b_prob))
                ngram_cnt+=1 

    HC = -sum_prob / ngram_cnt
    perpl = math.pow(2,HC)
    print("Cross Entropy: {0:.3f}".format(HC))
    print("perplexity: {0:.3f}".format(perpl))

## (iv) Optionally combine your two models using linear interpolation (slide 10) and check if the
## combined model performs better. 


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
    
    
    
    
calculate_metrics(test1_set,1)
