########## L-TS additions ###########

# import random
import random

# get random sentence from test set
randomLength = randint(1, len(Test_Clean))
randomSentence = Test_Clean[randomLength]

# tokenize this sentece into words (to find its length)

# import word tokenizers

from nltk import word_tokenize
from nltk import WhitespaceTokenizer
from nltk.tokenize import TweetTokenizer

tweet_wt = TweetTokenizer()
randomSentenceTokenized = tweet_wt.tokenize(randomSentence)
randomSentenceLength = len(randomSentenceTokenized)

randomVocabularySentence = random.sample(valid_vocabulary, randomSentenceLength)

# grams of vocabulary

unigram_counter = Counter()
bigram_counter = Counter()
trigram_counter = Counter()

unigram_counter.update([gram for gram in ngrams(valid_vocabulary, 1, pad_left=True, pad_right=True,
                                                left_pad_symbol='<s>', right_pad_symbol='<e>')])

bigram_counter.update([gram for gram in ngrams(valid_vocabulary, 2, pad_left=True, pad_right=True,
                                               left_pad_symbol='<s>', right_pad_symbol='<e>')])

trigram_counter.update([gram for gram in ngrams(valid_vocabulary, 3, pad_left=True, pad_right=True,
                                                left_pad_symbol='<s>', right_pad_symbol='<e>')])
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
                                                              left_pad_symbol='<s>', right_pad_symbol='<e>')])

bigram_counterRandomSentence.update([gram for gram in ngrams(randomSentenceTokenized, 2, pad_left=True, pad_right=True,
                                                             left_pad_symbol='<s>', right_pad_symbol='<e>')])

trigram_counterRandomSentence.update([gram for gram in ngrams(randomSentenceTokenized, 3, pad_left=True, pad_right=True,
                                                              left_pad_symbol='<s>', right_pad_symbol='<e>')])
# grams of vocabulary random sentence
unigram_counterRandomVocabularySentence.update(
    [gram for gram in ngrams(randomVocabularySentence, 1, pad_left=True, pad_right=True,
                             left_pad_symbol='<s>', right_pad_symbol='<e>')])

bigram_counterRandomVocabularySentence.update(
    [gram for gram in ngrams(randomVocabularySentence, 2, pad_left=True, pad_right=True,
                             left_pad_symbol='<s>', right_pad_symbol='<e>')])

trigram_counterRandomVocabularySentence.update(
    [gram for gram in ngrams(randomVocabularySentence, 3, pad_left=True, pad_right=True,
                             left_pad_symbol='<s>', right_pad_symbol='<e>')])
# probability estimations

# Bigrams

# We should fine-tune alpha on a held-out dataset
alpha = 0.01
# Calculate vocab size
vocab_size = len(valid_vocabulary)

# initialise probabilities to 1
bigram_prob_total_random_sentence = 1
bigram_prob_total_random_vocabulary_sentence = 1

# Bigram prob + laplace smoothing

for bigram in bigram_counterRandomSentence:
    bigram_prob = (bigram_counter[(bigram[1], bigram[0])] + alpha) / (
    unigram_counter[(bigram[1],)] + alpha * vocab_size)
    bigram_prob_total_random_sentence *= bigram_prob
print("bigram_prob: {0:.3f} ".format(bigram_prob_total_random_sentence))
bigram_log_prob_random_sentence = math.log2(bigram_prob_total_random_sentence)
print("bigram_log_prob: {0:.3f}".format(bigram_log_prob_random_sentence))

for bigram in bigram_counterRandomVocabularySentence:
    bigram_prob = (bigram_counter[(bigram[1], bigram[0])] + alpha) / (
    unigram_counter[(bigram[1],)] + alpha * vocab_size)
    bigram_prob_total_random_vocabulary_sentence *= bigram_prob
print("bigram_prob: {0:.3f} ".format(bigram_prob_total_random_vocabulary_sentence))
bigram_log_prob_random_vocabulary_sentence = math.log2(bigram_prob_total_random_sentence)
print("bigram_log_prob: {0:.3f}".format(bigram_log_prob_random_vocabulary_sentence))

# Trigrams

# We should fine-tune alpha on a held-out dataset
alpha = 0.01
# Calculate vocab size
vocab_size = len(valid_vocabulary)

# initialise probabilities to 1
trigram_prob_total_random_sentence = 1
trigram_prob_total_random_vocabulary_sentence = 1

# Trigram prob + laplace smoothing

for trigram in trigram_counterRandomSentence:
    trigram_prob = (trigram_counter[(trigram[1], trigram[0])] + alpha) / (
    bigram_counter[(trigram[1],)] + alpha * vocab_size)
    trigram_prob_total_random_sentence *= trigram_prob
print("trigram_prob: {0:.3f} ".format(trigram_prob_total_random_sentence))
trigram_log_prob_random_sentence = math.log2(trigram_prob_total_random_sentence)
print("trigram_log_prob: {0:.3f}".format(trigram_log_prob_random_sentence))

for trigram in trigram_counterRandomVocabularySentence:
    trigram_prob = (trigram_counter[(trigram[2], trigram[1], trigram[0])] + alpha) / (
    bigram_counter[(trigram[1], trigram[0])] + alpha * vocab_size)
    trigram_prob_total_random_vocabulary_sentence *= trigram_prob
print("trigram_prob: {0:.3f} ".format(trigram_prob_total_random_vocabulary_sentence))
trigram_log_prob_random_vocabulary_sentence = math.log2(trigram_prob_total_random_sentence)
print("trigram_log_prob: {0:.3f}".format(trigram_log_prob_random_vocabulary_sentence))

# Cross_entropy & perplexity

# Compute corpus cross_entropy & perplexity for bi-gram LM

sentences_tokenized = []
for sent in Test_Clean:
    # sent_tok = whitespace_wt.tokenize(sent)
    sent_tok = tweet_wt.tokenize(sent)
    # sent_tok = word_tokenize(sent)
    sentences_tokenized.append(sent_tok)

sum_prob = 0
bigram_cnt = 0
for sent in sentences_tokenized:
    sent = ['<start>'] + sent + ['<end>']
    for idx in range(len(sent)):
        bigram_prob = (bigram_counter[(sent[idx - 1], sent[idx])] + alpha) / (
        unigram_counter[(sent[idx - 1],)] + alpha * vocab_size)
        sum_prob += math.log2(bigram_prob)
        bigram_cnt += 1

HC = -sum_prob / bigram_cnt
perpl = math.pow(2, HC)

print("Cross Entropy: {0:.3f}".format(HC))
print("perplexity: {0:.3f}".format(perpl))

# Compute corpus cross_entropy & perplexity for tri-gram LM


sum_prob = 0
trigram_cnt = 0
for sent in sentences_tokenized:
    sent = ['<start1>'] + ['<start2>'] + sent + ['<end>'] + ['<end>']
    for idx in range(2, len(sent)):
        trigram_prob = (trigram_counter[(sent[idx - 2], sent[idx - 1], sent[idx])] + alpha) / (
        bigram_counter[(sent[idx - 1], sent[idx])] + alpha * vocab_size)
        sum_prob += math.log2(trigram_prob)
        trigram_cnt += 1

HC = -sum_prob / trigram_cnt
perpl = math.pow(2, HC)
print("Cross Entropy: {0:.3f}".format(HC))
print("perplexity: {0:.3f}".format(perpl))
# Compute corpus cross_entropy & perplexity for interpoladed bi-gram & tri-gram LMs

# We should fine-tune lamda on a held-out dataset
lamda = 0.9
sum_prob = 0
ngram_cnt = 0
for sent in sentences_tokenized:
    sent = ['<start1>'] + ['<start2>'] + sent + ['<end>'] + ['<end>']
    for idx in range(2, len(sent)):
        trigram_prob = (trigram_counter[(sent[idx - 2], sent[idx - 1], sent[idx])] + alpha) / (
        bigram_counter[(sent[idx - 1], sent[idx])] + alpha * vocab_size)
        bigram_prob = (bigram_counter[(sent[idx - 1], sent[idx])] + alpha) / (
        unigram_counter[(sent[idx - 1],)] + alpha * vocab_size)

        sum_prob += (lamda * math.log2(trigram_prob)) + ((1 - lamda) * math.log2(bigram_prob))
        ngram_cnt += 1

HC = -sum_prob / ngram_cnt
perpl = math.pow(2, HC)
print("Cross Entropy: {0:.3f}".format(HC))
print("perplexity: {0:.3f}".format(perpl))

