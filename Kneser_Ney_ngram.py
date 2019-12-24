"""
实现了ngram的Kneser-Ney平滑
"""

from collections import defaultdict

from collections import Counter

import numpy as np

import random

import math

import re

from nltk import WordNetLemmatizer, pos_tag, word_tokenize

from nltk.corpus import wordnet


class KneserNeyNGram():
    def __init__(self,
                 sents,
                 n=4,
                 discount=0.75):

        """
        sents: list of sentences
        full_seq: the input sequence
        input_type: whether the input is a list of sentences or a long sequence
        n: order of the model
        D: discount value
        """

        self.n = n
        self.discount = discount
        self._N_dot_tokens_dict = N_dot_tokens = defaultdict(set) # N+(·w_<i+1>)
        self._N_tokens_dot_dict = N_tokens_dot = defaultdict(set) # N+(w^<n-1> ·)
        self._N_dot_tokens_dot_dict = N_dot_tokens_dot = defaultdict(set) # N+(· w_<i-n+1>^<i-1> ·)
        self.counts = defaultdict(int)
        vocabulary = []

        # padding each sentence to the right model order
        sents = list(map(lambda x: ['<s>']*(n-1) + x , sents))

        for sent in sents:
            for j in range(n+1):
                # all k-grams for 0 <= k <= n
                for i in range(n-j, len(sent) - j + 1):
                    ngram = tuple(sent[i: i + j])
                    self.counts[ngram] += 1
                    if ngram:
                        if len(ngram) == 1:
                            vocabulary.append(ngram[0])
                        else:
                            right_token, left_token, right_kgram, left_kgram, middle_kgram =\
                                ngram[-1:], ngram[:1], ngram[1:], ngram[:-1], ngram[1:-1]
                            N_dot_tokens[right_kgram].add(left_token)                                
                            N_tokens_dot[left_kgram].add(right_token)
                            if middle_kgram:
                                N_dot_tokens_dot[middle_kgram].add((left_token,right_token))
            if n-1:
                self.counts[('<EOS>',)*(n-1)] = len(sents)
            self.vocab = set(vocabulary)

        temp = 0
        for w in self.vocab:
            temp += len(self._N_dot_tokens_dict[(w,)])
        self._N_dot_dot_attr = temp


    def count(self, tokens):

        """
        returns the count for an n-gram or (n-1)-gram.
        tokens be the n-gram or (n-1)-gram tuple.
        """
        return self.counts[tokens]

    def V(self):
        """
        returns vocabulary size.
        """
        return len(self.vocab)

    def N_dot_dot(self):
        """
        returns N+(..)
        """
        return self._N_dot_dot_attr

    def N_tokens_dot(self, tokens):
        """
        returns the set of words w for which C(tokens,w) > 0
        """
        if type(tokens) is not tuple:
            raise TypeError('`tokens` has to be a tuple of strings')
        return self._N_tokens_dot_dict[tokens]

    def N_dot_tokens(self, tokens):
        """
        returns the set of words w for which C(w,tokens) > 0
        """
        if type(tokens) is not tuple:
            raise TypeError('`tokens` has to be a tuple of strings')
        return self._N_dot_tokens_dict[tokens]

    def N_dot_tokens_dot(self, tokens):
        """
        returns the set of word pairs (w,w') for which C(w,tokens,w') > 0
        """
        if type(tokens) is not tuple:
            raise TypeError('`tokens` has to be a tuple of strings')
        return self._N_dot_tokens_dot_dict[tokens]


    def cond_prob(self, token, prev_tokens=tuple()):
        n = self.n

        # unigram case
        if not prev_tokens and n == 1:
            return (self.count((token,))) / (self.count(()))


        # lowest ngram (n >1 and unigram back-off)
        if not prev_tokens and n > 1:
            temp1 = len(self.N_dot_tokens((token,)))
            temp2 = self.N_dot_dot()
            return ((temp1 + 1) * 1.0) / (temp2 + self.V())

        # highest n-gram (no back-off)
        if len(prev_tokens) == n-1:
            c = self.count(prev_tokens)
            if c == 0:
                return self.cond_prob(token, prev_tokens[1:])
            term1 = max(self.count(prev_tokens + (token,)) - self.discount, 0) / c
            unassigned_mass = self.discount * len(self.N_tokens_dot(prev_tokens)) / c
            back_off = self.cond_prob(token, prev_tokens[1:])
            return term1 + unassigned_mass * back_off

        # lower ngram (back off to lower-order models except the unigram)
        else:
            temp = len(self.N_dot_tokens_dot(prev_tokens))
            if temp == 0:
                return self.cond_prob(token, prev_tokens[1:])
            term1 = max(len(self.N_dot_tokens(prev_tokens + (token,))) - self.discount, 0) / temp
            unassigned_mass = self.discount * len(self.N_tokens_dot(prev_tokens)) / temp
            back_off = self.cond_prob(token, prev_tokens[1:])
            return term1 + unassigned_mass * back_off

def check_sum(model, seq):
    """
    used to check the probabilities of a row sum to 1

    """
    prob = []
    for i in range(model.n-1,100+model.n-1):
        context = tuple(seq[i-model.n+1:i])
        sum_prob = 0
        for w in model.vocab:
            sum_prob += model.cond_prob(w, context)
        prob.append(sum_prob)
    return prob

def sent_log_prob(model,sent):

    """
    log-probability of a sentence.

    """
    prob = 0
    #print(sent)
    for i in range(model.n - 1, len(sent)):
        c_p = model.cond_prob(sent[i], tuple(sent[i - model.n + 1:i]))
        #print(c_p) ############################
        if not c_p:
            return float('-inf')
        prob = prob + np.log(c_p)

    return prob

def perplexity(model,sents):
    """
    Perplexity of a model when the inputs are sentences

    """

    words = 0
    for sent in sents:
        words += len(sent) - model.n
    l = 0
    print ('number of test words: ', words)
    for sent in sents:
        l += sent_log_prob(model,sent)
    l = l / words
    return np.exp(-l)



def ngram_generator(s, n, k=2):
    if (len(s.split()) < n):
        return "ERROR, NUMBER OF GRAM EXCEED TEXT!"

    # Convert to lowercases
    s = s.lower()

    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)

    s = s.strip() #去掉换行符
    s = "<BOS> " * (k-1) + s + " <EOS>"
        
    # Break sentence in the token, remove empty tokens
    token = [token for token in s.split(" ") if token != ""]

    # Stemming and Lemmatizing
    # 将句子中每个词变为原词形式输出由原形单词组成的句子数组存储在token中
    lemmatizer = WordNetLemmatizer()
    tagged = pos_tag(token)
    token = []
    for word, tag in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:  # not supply tag in case of None
            lemma = lemmatizer.lemmatize(word)
            token.append(lemma)
        else:
            lemma = lemmatizer.lemmatize(word, pos=wntag)
            token.append(lemma)

    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[token[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# ngram = ngram_generator('I am a student working in the library', 3)
# print(ngram)
# input('press enter')

def f_original_shape(file_name, k):
    f_list = []
    f = open(file_name, 'r', encoding='utf-8', errors='ignore')
    lines = f.readlines()
    f.close()
    for line in lines:
        if line.strip() == "": #跳过空行
                continue
        line = ngram_generator(line, 1, k)
        f_list.append(line)
    return f_list


train_file = 'train.txt'
test_file = 'test.txt'
ngram = 2


train_sents = f_original_shape(train_file, ngram)
test_sents = f_original_shape(test_file, ngram)

print(ngram, " gram时")
print ("Number of train sentences: ", len(train_sents))
print ("Number of test sentences: ", len(test_sents))

model = KneserNeyNGram(train_sents, n=ngram, discount=0.75)
print ("Vocabulary size: ", model.V())
print ("perplexity = ", perplexity(model,test_sents))
print(test_sents[0])



train_file = 'train.txt'
test_file = 'test.txt'
ngram = 3

train_sents = f_original_shape(train_file, ngram)
test_sents = f_original_shape(test_file, ngram)

print(ngram, " gram时")
print ("Number of train sentences: ", len(train_sents))
print ("Number of test sentences: ", len(test_sents))

model = KneserNeyNGram(train_sents, n=ngram, discount=0.75)
print ("Vocabulary size: ", model.V())
print ("perplexity = ", perplexity(model,test_sents))
print(test_sents[0])


train_file = 'train.txt'
test_file = 'test.txt'
ngram = 4

train_sents = f_original_shape(train_file, ngram)
test_sents = f_original_shape(test_file, ngram)

print(ngram, " gram时")
print ("Number of train sentences: ", len(train_sents))
print ("Number of test sentences: ", len(test_sents))

model = KneserNeyNGram(train_sents, n=ngram, discount=0.75)
print ("Vocabulary size: ", model.V())
print ("perplexity = ", perplexity(model,test_sents))
print(test_sents[0])

train_file = 'train.txt'
test_file = 'test.txt'
ngram = 5


train_sents = f_original_shape(train_file, ngram)
test_sents = f_original_shape(test_file, ngram)

print(ngram, " gram时")
print ("Number of train sentences: ", len(train_sents))
print ("Number of test sentences: ", len(test_sents))

model = KneserNeyNGram(train_sents, n=ngram, discount=0.75)
print ("Vocabulary size: ", model.V())
print ("perplexity = ", perplexity(model,test_sents))
print(test_sents[0])