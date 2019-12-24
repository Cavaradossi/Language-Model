from collections import Counter

import os

import numpy as np

import random

import math

import re

from nltk import WordNetLemmatizer, pos_tag, word_tokenize

from nltk.corpus import wordnet

from run import test


def ngram_generator(s):
    # Convert to lowercases
    s = s.lower()
    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    #s = s.strip() #去掉换行符
    #for i in range(k-1):  #加k-1个"<BOS> "和" <EOS>"
        #s = "<BOS> " + s + " <EOS>"
        
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
    ngrams = zip(*[token[i:] for i in range(1)])
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




def prob_perplexity(sentence, word2id, bigram, oov):    
    count = 0
    p = 0.0
    n = len(sentence) - 1
    for i in range(1,len(sentence)):
        if sentence[i - 1] not in word2id: #oov来替换未登录词
            sentence[i - 1] = oov
        if sentence[i] not in word2id:
            sentence[i] = oov
        if bigram[word2id[sentence[i - 1]], word2id[sentence[i]]] > 0:
            p += math.log(bigram[word2id[sentence[i - 1]], word2id[sentence[i]]], 2)
        else:
            print(sentence[i - 1],sentence[i])
            p = 0
            break
    perplexity = 2 ** (- p / n)
    return perplexity

#读取概率矩阵bigram
def read_bigram(file):
    bigram = np.loadtxt(file, delimiter=',')
    return bigram

#读取字典word2id
def read_word2id(file):
    with open(file,'r+') as f:
        word2id = eval(f.read())
    return word2id

def calP_bigram(sentence,  word2id, bigram, oov):
    sent = ngram_generator(sentence)
    if len(sent) == 0:
        return "未输入"
    sent = ["<BOS>"] + sent + ["<EOS>"]
    perplexity = prob_perplexity(sent, word2id, bigram, oov)
    return perplexity

def predict(sentence,  word2id, bigram, oov):
    sent = ngram_generator(sentence)
    unk_id = word2id["unk"]
    if len(sent) == 0:
        return "未输入"
    word = sent[-1]
    if word not in word2id: 
            word = oov
    id2word = id2word = {i: w for w, i in word2id.items()}
    x = word2id[word]
    k = 1
    pmax = bigram[x][1]
    for i in range(2, len(id2word)-20):
        if bigram[x][i] > pmax and i != unk_id:
            pmax = bigram[x][i]
            k = i
    pre_word = id2word[k]
    return pre_word
    


def read_data(smooth_id):
    bigram_dic = {}
    word2id_dic = {}
    for smooth in smooth_id:
        if smooth != "RNN":
            bigram_file = smooth + "_data.txt"
            bigram = read_bigram(bigram_file)
            bigram_dic[smooth] = bigram
            word2id_file = smooth + "_word2id.txt"
            word2id = read_word2id(word2id_file)
            word2id_dic[smooth] = word2id
    word2id = word2id_dic[smooth_id[0]]
    id2word = {i: w for w, i in word2id.items()}
    n = len(id2word)
    oov = id2word[n//3 * 2]
    return bigram_dic, word2id_dic, oov  

 

def load_data(): #加载数据
    smooth_id = ["addone", "good_turing", "katz", "absolute_discouting",
            "linear_discouting", "kneser_ney", "deleted_interpolation","RNN"]
    bigram_dic, word2id_dic, oov = read_data(smooth_id)
    return smooth_id, bigram_dic, word2id_dic, oov

def jisuan(s, smooth_id, bigram_dic, word2id_dic, oov): #计算句子
    words = {"addone":"加一法平滑", "good_turing":"古德-图灵平滑","katz":"Katz平滑",
    "absolute_discouting":"绝对减值平滑","linear_discouting":"线性减值平滑",
    "kneser_ney":"Kneser-Ney平滑", "deleted_interpolation":"删除插值平滑", "RNN":"RNN神经网络"}
    A = []
    for smooth in smooth_id:
        if smooth != "RNN":
            try:
                p = calP_bigram(s,  word2id_dic[smooth], bigram_dic[smooth], oov)
                p = int(p)
                if p > 10:
                    A.append(words[smooth] + " 困惑度："+ str(p))
            except:
                pass
        else:
            try:
                p = test(s)
                p = int(p)
                A.append(words[smooth] + " 困惑度："+ str(p))
                try:
                    with open("log/prob_result", "r") as f:
                        lines = f.readlines()
                        line = lines[-1]
                        line = line.split(",")
                    B = []
                    a,b = 0, 0
                    for x in line:
                        B.append(float(x))
                    for i in range(len(B)):
                        if B[i] > B[a]:
                            a, b = i, a
                        elif B[a]>= B[i] > B[b]:
                            a, b = a, i 
                    with open("vocab", "r") as f:
                        lines = f.read().splitlines()
                    pre_word = lines[a]
                    pre_word2 = lines[b]
                    A.append("下一个可能出现的单词是： " + pre_word + " 或者 " + pre_word2)
                except:
                    pass
            except:
                pass
    return A

def begin(s): #加载数据， 计算句子
    smooth_id, bigram_dic, word2id_dic, oov = load_data()
    A = jisuan(s, smooth_id, bigram_dic, word2id_dic, oov)
    return A

#s = "go to work "
#A = begin(s)
#print(A)