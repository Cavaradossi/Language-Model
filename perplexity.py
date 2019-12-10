import numpy as np
import os
from collections import Counter
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
import sklearn
import string
from collections import deque
from itertools import islice
import collections
import math
import argparse
import time
'''
    only for n=2,3 or more, generate {words:count}
    usually take training ngram tokens such as {a,b,c} as input,
    generate ngram count with UNK
    when input test/dev data, it is used for error analysis

'''


def word_freq(tokens):
    start_time = time.time()
    ngram_freq = {}
    # initial work-count dict population
    for token in tokens:
        ngram_freq[token] = ngram_freq.get(token, 0) + 1
    return ngram_freq


''' calculate MLE Probablity of unigram
    input word-freq dict for training data, which is Vocaborary
    this function will run even n specified by the shell is not 1
'''


def unigrams_prob(uni_count_dict):
    # probability dict {word:prob}
    prob_dict = uni_count_dict
    # print vocabulary
    items = prob_dict.iteritems()
    for word, count in items:
        prob_dict[word] = float(count) / float(total_words_len)
    return prob_dict


'''
calculate MLE probability of ngram, n>=2
: param: n: count dict of ngram,start from bigram
: param: input untokened train texts with STOP sign
'''


def ngram_prob(n, tokens, unigram_count):
    # print('------start ngram_prob---------------')
    start_time = time.time()
    # generate {ngrams:count} from training data
    ngram_list = list(ngrams_gen(tokens, n))

    ngram_count_pairs = word_freq(ngram_list)
    prob_dict = ngram_count_pairs
    if (n == 2):
        items = prob_dict.iteritems()
        uni_count = unigram_count
        # current probablity and word, in case n = 2, input is bigram words:count dict
        # input {a,b}: count, continue to get {a}: count
        for words, count in items:
            # extract the first item in bigram.
            prior_word = words[0]
            # get the count from {unigram: count} generated before
            cnt_prior = uni_count[prior_word]
            # print(prior_word,words,cnt_prior,count)
            # q(w/v) = c(v,w)/c(v)
            prob_dict[words] = count / cnt_prior
            # print(count,cnt_prior)
        # this should save as global for later use as bigram_prob_dict
        return prob_dict
    if (n > 2):
        items = prob_dict.iteritems()
        # get {n-1gram:count} pairs
        priorgram_list = list(ngrams_gen(tokens, n - 1))
        priorgram_count_pairs = word_freq(priorgram_list)
        # -----------need to discard first few items--------
        for words, count in items:
            prior_word = words[:n - 1]
            cnt_prior = priorgram_count_pairs[prior_word]
            # print(prior_word,words,cnt_prior,count)
            prob_dict[words] = count / cnt_prior
        return prob_dict


def logprob(word, prob_dict):
    prob_dict = prob_dict
    return -math.log(prob_dict[word], 2)


''' calculate entropy given a test/dev text
# input text should be processed propriately
# N = 1,2,3
# smooth_type used to deal with unseen word in different smoothing method
'''


def entropy(test_test, n, prob_dict, smooth_type):
    entr = 0.0
    text = test_test
    tokens = tokenize_unigram(text)
    # number of words in text
    text_len = len(tokens)
    global vocabulary

    sentences = set([s for s in text.splitlines()])
    # number of sentences
    sent_num = len(sentences)
    voc_set = set(prob_dict.keys())

    if (n == 1):

        for sent in sentences:
            sent_temp = tokenize_unigram(sent)
            for word in sent_temp:
                if word not in voc_set:
                    entr += logprob(UNK_token, prob_dict)
                else:
                    entr += logprob(word, prob_dict)
    if (n > 1):
        # ngram_prob_dict = ngram_prob(n,train_cut)
        for sent in sentences:
            # generate ngram for single sentence test data
            ngram_tmp = tuple(ngrams_gen(tokenize_unigram(sent), n))
            # iterate ngram in one sentence, skip first n-1 items
            for i in xrange(n - 1, len(list(ngram_tmp))):
                # print i, ngram_tmp[i]
                if ngram_tmp[i] not in voc_set:
                    if (smooth_type == NO_SMOOTHING):
                        entr += -math.log(0, 2)
                    if (smooth_type == ADD_K_SMOOTHING):
                        entr += logprob(UNSEEN_NGRAM, prob_dict)

                else:
                    entr += logprob(ngram_tmp[i], prob_dict)
    return entr / float(text_len - (n - 1) * sent_num)


''' 
perplexity for  ngram
'''


def perplexity(test_text, n, prob_dict, smooth_type):
    return math.pow(2.0, entropy(test_text, n, prob_dict, smooth_type))