import re, math, sys
import numpy as np
from nltk.util import ngrams


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


def cleanse(dataset, dataset_size, valid_WordCounts):
    for i in range(0, dataset_size):
        for j in range(0, len(dataset[i])):
            if dataset[i][j] not in valid_WordCounts:
                dataset[i][j] = 'UNK'
    return dataset


# TODO : Costa , giati na metrame xwria ta sentences? An se enoxlei to global scope twn counters,
#   de nomizw oti einai logos anhsyxias. Epishs an ta bgaleis apo ta functions, metras kai tis skartes sentences.
def split_into_unigrams(sentence,unigram_counter):
    if sentence.__class__ == str:
        print(sentence)
        print("Error in corpus sentence (unigrams func)!!")
        sys.exit("Requested Exit.")
    ngs = [gram for gram in ngrams(sentence, 1)]
    unigram_counter.update(ngs)
    return ngs


def split_into_bigrams(sentence, bigram_counter, pad=True, s="start", e="end"):
    if sentence.__class__ == str:
        print("Error in corpus sentence (bigrams func)!!")
        print(sentence)
        sys.exit("Requested Exit.")
    list = [s]+sentence+[e] if pad else sentence
    # print("debug_padding:",s,sentence,e)
    ngs = [gram for gram in ngrams(list, 2)]
    bigram_counter.update(ngs)
    return ngs


def split_into_trigrams(sentence, trigram_counter, pad=True, s1= "start1", s2= 'start2', e='end'):
    if sentence.__class__ == str:
        print(sentence)
        print("Error in corpus sentence (trigrams func)!!")
        sys.exit("Requested Exit.")
    list = [s1, s2]+sentence+[e] if pad else sentence
    ngs = [gram for gram in ngrams(list, 3)]
    trigram_counter.update(ngs)
    return ngs


# TODO : Costa, pws ginetai na mhn yparxei kapoio n-gram? Fovamai mhn einai to tackling edw
# san na kryvoume kapoio lathos katw apo to xali
def unigram_prob(ngram, unigrams_training_counter, vocab_size, C, a=0.01):
    x = ngram[0]
    # Tackle cases where the gram doesn't exist in the training counter
    try:
        out = (unigrams_training_counter[(x,)] + a) / (C + a * vocab_size)
    except:
        out = a / (C + a * vocab_size)
    return out


def bigram_prob(ngram, unigrams_training_counter, bigrams_training_counter, vocab_size, a=0.01):
    x = ngram[0]
    y = ngram[1]
    if (x, y) not in bigrams_training_counter:
        out = a
    else:
        out = (bigrams_training_counter[(x, y)] + a)
    # Tackle cases where the gram doesn't exist in the training counter
    if (x,) not in unigrams_training_counter:
        out *= 1 / (a * vocab_size)
    else:
        out *= 1 / (unigrams_training_counter[(x,)] + a * vocab_size)
    return out


def trigram_prob(ngram, bigrams_training_counter, trigrams_training_counter, vocab_size, a=0.01):
    x = ngram[0]
    y = ngram[1]
    z = ngram[2]
    if (x, y, z) not in trigrams_training_counter:
        out = a
    else:
        out = (trigrams_training_counter[(x, y, z)] + a)
    # Tackle cases where the gram doesn't exist in the training counter
    if (x, y,) not in bigrams_training_counter:
        out *= 1 / (a * vocab_size)
    else:
        out *= 1 / (bigrams_training_counter[(x, y,)] + a * vocab_size)
    return out


def print_sentence_unigram_probs(sentence, unigram_counter, vocab_size, C, a=0.01):
    sum_prob = 0
    for unigram in split_into_unigrams(sentence, unigram_counter):
        sum_prob += math.log2(unigram_prob(unigram, unigram_counter, vocab_size, C, a))
    print('Unigram log Prob of sentence: ', str(sum_prob))


def print_sentence_bigram_probs(sentence, unigram_counter, bigram_counter, vocab_size, a=0.01):
    sum_prob = 0
    for bigram in split_into_bigrams(sentence, bigram_counter):
        sum_prob += math.log2(bigram_prob(bigram,unigram_counter, bigram_counter, vocab_size, a))
    print('Bigram log Prob of sentence: ', str(sum_prob))


def print_sentence_trigram_probs(sentence, bigram_counter, trigram_counter, vocab_size, a=0.01):
    sum_prob = 0
    for trigram in split_into_trigrams(sentence, trigram_counter):
        sum_prob += math.log2(trigram_prob(trigram,bigram_counter,trigram_counter, vocab_size, a))
    print('Trigram log Prob of sentence: ', str(sum_prob))


def bigram_linear_interpolation_probs(sentence, unigram_counter, bigram_counter, vocab_size, C, a=0.01, l=0.7):
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
    bigrams = split_into_bigrams(sentence, bigram_counter)
    bigram_linear_interpolations = []
    for bigram in bigrams:
        unigram = bigram[0]
        linear_interpolation =\
            l * bigram_prob(bigram, unigram_counter, bigram_counter,vocab_size, a) +\
            (l - 1) * unigram_prob(unigram,unigram_counter, vocab_size, C, a)
        bigram_linear_interpolations.append([bigram, np.round(linear_interpolation, 4)])
    return bigram_linear_interpolations


def trigram_linear_interpolation_probs(sentence, unigram_counter, bigram_counter, trigram_counter, vocab_size, C, a=0.01, l1=0.7, l2=0.2):
    """
    Due to the nature of the calculations, I return all trigram linear interpolations at once for the sentence
    :param sentence:
    :param vocab_size:
    :param a:
    :param trigram_counter:
    :return:
    """
    if (l1 + l2 > 1) or (l1 < 0) or (l2 < 0):
        print("Error: Lambas should be 0 <= l1,l2,(l1+l2) <= 1.")
        return False

    trigrams = split_into_trigrams(sentence, trigram_counter)
    trigram_linear_interpolations = []
    for trigram in trigrams:
        bigram = (trigram[0], trigram[1],)
        unigram = bigram[0]
        linear_interpolation =\
            l1 * trigram_prob(trigram,bigram_counter, trigram_counter, vocab_size, a) +\
            l2 * bigram_prob(bigram, unigram_counter, bigram_counter,vocab_size, a) + \
            (1 - l1 - l2) * unigram_prob(unigram, unigram_counter, vocab_size, C, a)
        trigram_linear_interpolations.append([trigram, np.round(linear_interpolation, 4)])
    return trigram_linear_interpolations


# functions for combining the above probs into making the P(t_i_k), aka the Language models
# SOS : sums of logs instead of mults !!! Then pow, so that low numbers wont get zero.
def unigram_language_model(sentence, unigram_counter, vocab_size, C, a=0.01):
    language_model = 0  # neutral value
    for unigram in split_into_unigrams(sentence, unigram_counter):
        try:  # Skip zero probabilities (error: "ValueError: math domain error")
            language_model += math.log2(unigram_prob(unigram, unigram_counter, vocab_size, C, a))
        except Exception:
            continue
    return math.pow(2, language_model)


def bigram_language_model(sentence, unigram_counter, bigram_counter, vocab_size, a=0.01):
    language_model = 0  # neutral value
    for bigram in split_into_bigrams(sentence, bigram_counter):
        try:  # Skip zero probabilities (error: "ValueError: math domain error")
            language_model += math.log2(bigram_prob(bigram, unigram_counter, bigram_counter,vocab_size, a))
        except Exception:
            continue
    return math.pow(2, language_model)


def trigram_language_model(sentence, bigram_counter, trigram_counter, vocab_size, a=0.01):
    language_model = 0  # neutral value
    for trigram in split_into_trigrams(sentence, trigram_counter):
        try:  # Skip zero probabilities (error: "ValueError: math domain error")
            language_model += math.log2(trigram_prob(trigram,bigram_counter,trigram_counter, vocab_size, a))
        except Exception:
            continue
    return math.pow(2, language_model)


def bigram_linear_interpolation_language_model(sentence,unigram_counter, bigram_counter,
                                               vocab_size, C, a=0.01, l=0.7):
    language_model = 0  # neutral value
    for pair in bigram_linear_interpolation_probs(sentence, unigram_counter, bigram_counter,
                                                  vocab_size, C, a, l):
        prob = pair[1]
        try:  # Skip zero probabilities (error: "ValueError: math domain error")
            language_model += math.log2(prob)
        except Exception:
            return 0
            # continue
    return math.pow(2, language_model)


def trigram_linear_interpolation_language_model(sentence, unigram_counter, bigram_counter, trigram_counter,
                                                vocab_size, C, a=0.01, l1=0.7, l2=0.2):
    language_model = 0  # neutral value
    for pair in trigram_linear_interpolation_probs(sentence, unigram_counter, bigram_counter, trigram_counter,
                                                   vocab_size, C, a, l1, l2):
        prob = pair[1]
        try:  # Skip zero probabilities (error: "ValueError: math domain error")
            language_model += math.log2(prob)
        except Exception:
            return 0
            # continue
    return math.pow(2, language_model)


# function for model cross-entropy & preplexity (in same function) (slides 29,30)
def crossentropy_perplexity(dataset, ngram_type, unigram_counter, bigram_counter,trigram_counter,
                            vocab_size, C, a=0.01, l=0.7, l1=0.7, l2=0.2):
    sum_prob = 0
    ngram_cnt = 0
    sentcount = -1
    for sentence in dataset:
        sentcount += 1
        if (sentence == None) or (sentence == []) or (sentence == '') or (sentence in ['‘', '•', '–', '–']):
            # print("Erroneous Sentence", sentcount,"start:", sentence, ":end")
            continue
        # These lines below are very similar to the internals of language models
        if ngram_type == 'unigram':
            for ngram in split_into_unigrams(sentence, unigram_counter):
                try:
                    sum_prob += math.log2(unigram_prob(ngram,unigram_counter, vocab_size, C, a))
                except Exception:
                    ""
                ngram_cnt += 1
        elif ngram_type == 'bigram':
            for ngram in split_into_bigrams(sentence, bigram_counter):
                try:
                    sum_prob += math.log2(bigram_prob(ngram,unigram_counter, bigram_counter, vocab_size, a))
                except Exception:
                    ""
                ngram_cnt += 1
        elif ngram_type == 'trigram':
            for ngram in split_into_trigrams(sentence, trigram_counter):
                try:
                    sum_prob += math.log2(trigram_prob(ngram,bigram_counter,trigram_counter, vocab_size, a))
                except Exception:
                    ""
                ngram_cnt += 1
        elif ngram_type == 'lin_pol_bi':
            try:
                sum_prob += math.log2(bigram_linear_interpolation_language_model(
                    sentence, unigram_counter, bigram_counter,vocab_size, C, a, l))
            except Exception:
                ""
            for ngram in split_into_bigrams(sentence, bigram_counter):
                ngram_cnt += 1
        elif ngram_type == 'lin_pol_tri':
            try:
                sum_prob += math.log2(trigram_linear_interpolation_language_model(
                    sentence, unigram_counter, bigram_counter,trigram_counter,vocab_size, C, a, l1, l2))
            except Exception:
                ""
            for ngram in split_into_trigrams(sentence, trigram_counter):
                ngram_cnt += 1
    HC = -sum_prob / ngram_cnt
    perpl = math.pow(2, HC)
    print("Cross Entropy: {0:.3f}".format(HC))
    print("perplexity: {0:.3f}".format(perpl))
