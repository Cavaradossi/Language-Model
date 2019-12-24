"""
实现了几个平滑方法的2gram模型，包括：古德-图灵平滑、Katz平滑、绝对减值平滑、
线性减值平滑、Kneser-Ney平滑
"""

from collections import Counter

import numpy as np

import random

import math

import re

from nltk import WordNetLemmatizer, pos_tag, word_tokenize

from nltk.corpus import wordnet
'''
由train_data得到test_sentence的unigram概率和bigram概率
计算步骤
1. 将train_data预处理,将train_data中的单词变成原形,以列表flist输出
2.统计flist中各单词的个数,以counter:(word,times)输出
3.给单词表建立id,得到word2id:(word,id)
4.得到id2word(id,word)
5.得到二维数组,俩俩单词出现的次数
6.得到二维数组,p(w2|w1)组成的矩阵frequence
7.根据频率矩阵frequence计算概率矩阵bigram
8.计算test sentence bigram
'''

def ngram_generator(s, n, k = 2):
    if (len(s.split()) < n):
        return "ERROR, NUMBER OF GRAM EXCEED TEXT!"

    # Convert to lowercases
    s = s.lower()

    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)

    s = s.strip() #去掉换行符
    for i in range(k-1):  #加k-1个"<BOS> "和" <EOS>"
        s = "<BOS> " + s + " <EOS>"
        
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




'''
语料库预处理,将语料库按行读取(一行为一句),将所有词还原为原型,以句子列表形式输出
输入:
file_name eg.'train_LM.txt' 同一目录下的txt文件
输出:
f_list:文件中句子以列表输出,每个句子中的词都是原型
e.g. [['<BOS>',i', 'be', 'fine', '<EOS>'], ['<BOS>','i', 'be', 'a', 'student', '<EOS>'],...]
输入输出举例
输入
i am fine\n
i am a student\n
输出
[['<BOS>',i', 'be', 'fine', '<EOS>'], ['<BOS>','i', 'be', 'a', 'student', '<EOS>']]
'''

def f_original_shape(file_name):
    f_list = []
    f = open(file_name, 'r', encoding='utf-8', errors='ignore')
    lines = f.readlines()
    f.close()
    for line in lines:
        if line.strip() == "": #跳过空行
                continue
        line = ngram_generator(line, 1)
        f_list.append(line)
    return f_list


'''
统计预处理过的语料库文件f_list中个单词出现的次数
输入:
预处理过的文件f_list
输出:
n*2的列表counter,n是词汇表总次数
counter[n][0]是word,没有重复的word
counter[n][1]是word出现次数
e.g.[('be', 4), ('a', 3), ('i', 2), ('fine', 1),...]
'''


def generate_counter_list(flist):
    counter = Counter()  # 词频统计
    for sentence in flist:
        for word in sentence:
            counter[word] += 1  # 计算每个字的出现次数{"word1":times,"word2":times2,...}
    counter = counter.most_common()  # 将上面的结果排序{"word1":top_times,"word2":top2_times,...}
    return counter


'''
给单词增加id索引
输入:
(word,出现次数)
输出:
word2id:(word,id)
id = 0 为"<BOS>"， id = n -1 为"<EOS>"
'''


def get_word2id(counter):
    lec = len(counter)  # counter中word的个数
    word2id = {}  # {'的': 0, '很': 1, '菜': 2, '她': 3, '好': 4, '他': 5, '香': 6}
    id = 1
    for i in range(lec):
        if(counter[i][0] != "<BOS>" and counter[i][0] != "<EOS>"):
            word2id[counter[i][0]] = id
            id += 1
    word2id["<BOS>"] = 0
    word2id["<EOS>"] = lec - 1
    return word2id


'''
将word2id反转
输入
word2id
输出
id2word:(id,word)
'''


def get_id2word(word2id):
    id2word = {i: w for w, i in word2id.items()}
    return id2word


'''
返回bigram组成的矩阵中俩俩词的出现个数，即频率矩阵frequence
'''

def get_frequence(word2id, flist):
    lec = len(word2id)
    frequence = np.zeros((lec, lec)) 
    for sentence in flist:
        pre_word = "<BOS>"
        for word in sentence:
            if word != "<BOS>":
                x = word2id[pre_word]
                y = word2id[word]
                frequence[x][y] += 1
                pre_word = word
    return frequence               


#事先随机选择一个频率为1-3的已经记录的词，用来替换未登录词
def select_oov(counter): 
    oov = '$$$$$$$'
    lec = len(counter)
    for i in range(500):
        t = random.randint(0, lec)
        if 1<= counter[t][1] <= 3:
            oov = counter[t][0]
            break
    if oov == '$$$$$$$':
        t = random.randint(0, lec)
        oov = counter[t][0]
    return oov   #oov为所选的词
    
    
    
"""
log结果
输入
test_sentence,word2id,由训练数据得到的所有单词概率bigram矩阵， oov来替换未登录词
输出
p_bigram(test_sentence)， n 为词总数
对n个单词的句子，加入<EOS>，<BOS>后长n+2,计算句子概率时计算n+1次2gram,算困惑度时取单词数n+1。
"""


def prob_bigram(sentence, word2id, bigram, oov):
    
    sentence = ngram_generator(sentence, 1)  #对句子进行处理，然后加<BOS>,<EOS>
    
    # s = [word2id[w] for w in sentence]  # 将句子编程id序列[wordid1,wordid2,wordid3,...]
    count = 0
    p = 0.00
    n = 0
    for i in range(1,len(sentence)):
        if sentence[i - 1] not in word2id: #oov来替换未登录词
            sentence[i - 1] = oov
            count += 1
        if sentence[i] not in word2id:
            sentence[i] = oov
            count += 1
        p += math.log(bigram[word2id[sentence[i - 1]], word2id[sentence[i]]], 2)
        n += 1
    n -= 1
    return p, n, count


'''得到整个测试集的概率'''
def prob_bigram_T(test_filename,word2id,bigram, oov):
    f = open(test_filename, 'r', encoding='utf-8', errors='ignore')
    lines = f.readlines()
    f.close()
    n = 0
    count = 0
    p=0
    for line in lines:
        res2, k, c = prob_bigram(line, word2id,bigram, oov)
        p+=  res2
        n += k
        count += c
    return p, n, count


'''
该函数从数据集中返回词汇数目
Inputs:
f: 数据集文件
Returns:
vocab_num: 词汇数目
'''

def get_vocab_num(f):
    f_list = f_original_shape(f)
    list = generate_counter_list(f_list)
    vocab_num = len(list)

    return vocab_num


'''
Inputs:
vocab_num: 词汇数量，对n个单词的句子，加入<EOS>，<BOS>后长n+2,计算句子概率时计算n+1次2gram,算困惑度时对每句话取单词数n+1。
corpus_p: 整个测试集的概率
Returns: 交叉熵
'''


def cross_entropy(vocab_num, log_corpus_p):
    cross_entropy = -(1 / vocab_num) * log_corpus_p
    return cross_entropy


'''
Inputs:
cross_entropy: 模型交叉熵
Returns
perplexity: 模型困惑度
'''


def perplexity(cross_entropy):
    perplexity = 2 ** cross_entropy
    return perplexity


"""
平滑方法
"""

def get_unigram(counter, word2id): #输出一维数组，计算概率时不包括<BOS>,<EOS>
    K = {x[0]:x[1] for x in counter}
    K.pop("<BOS>")
    K.pop("<EOS>")
    unigram = [0] * len(word2id)
    for word, x in K.items():
        unigram[word2id[word]] = x
    N = sum(unigram)
    for i in range(len(word2id)):
        unigram[i] = unigram[i] / N
    return unigram


def get_backoff_unigram(counter, word2id):#输出一维数组，计算概率时包括<BOS>,<EOS>
    K = {x[0]:x[1] for x in counter}
    unigram = [0] * len(word2id)
    for word, x in K.items():
        unigram[word2id[word]] = x
    N = sum(unigram)
    for i in range(len(word2id)):
        unigram[i] = unigram[i] / N
    return unigram


'''
未平滑
Inputs:
frequence: 频率矩阵
Returns
bigram: 概率矩阵
'''

def unsmooth(frequence): #输出二维矩阵
    lec = len(frequence)
    bigram = np.zeros((lec, lec))
    for i in range(lec-1):
        bigram[i] = frequence[i] / sum(frequence[i])
    return bigram
'''
加一法
Inputs:
frequence: 频率矩阵
Returns
bigram: 概率矩阵
'''

def add_one(frequence):
    lec = len(frequence)
    bigram = np.zeros((lec, lec))
    for i in range(lec-1):
        N = sum(frequence[i]) + lec - 1
        bigram[i][:] = (frequence[i][:] + 1) / N            
    return bigram

'''
good_turing
Inputs:
frequence: 频率矩阵
Returns
bigram: 概率矩阵
'''

def good_turing(frequence):
    lec = len(frequence)
    bigram = np.zeros((lec, lec))
    for i in range(lec-1):
        Rn_count = {}
        for x in frequence[i]:
            Rn_count[x] = Rn_count.get(x, 0) + 1
        Rn2Rxing = {}
        r_max = max(Rn_count.keys())
        r = 0
        while(r<r_max):
            if r in Rn_count:
                rj = r + 1
                while(rj not in Rn_count):
                    rj += 1
                Rn2Rxing[r] = Rn_count[rj] * rj / Rn_count[r]
                r = rj
            else:
                r += 1
        Rn2Rxing[r_max] = r_max
        for j in range(1, lec):
            rn = frequence[i][j]
            bigram[i][j] = Rn2Rxing[rn]
        bigram[i] = bigram[i] / sum(bigram[i])        
    return bigram

'''
Katz
Inputs:
frequence: 频率矩阵，backoff_unigram:一维数组unigram
Returns
bigram: 概率矩阵
'''

def Katz(frequence, backoff_unigram):
    p_good_turing = good_turing(frequence)
    lec = len(frequence)
    bigram = np.zeros((lec, lec))
    for i in range(lec-1):
        beta = 0
        Na = 0
        for j in range(1, lec):
            if frequence[i][j] != 0:
                beta += p_good_turing[i][j]
                Na += backoff_unigram[j]
        alpha = (1 - beta) / (1 - Na)                    
        for j in range(1, lec):
            if frequence[i][j] != 0:
                bigram[i][j] = p_good_turing[i][j]               
            else:
                bigram[i][j] = alpha * backoff_unigram[j]
        bigram[i] = bigram[i] / sum(bigram[i])
    return bigram

'''
绝对减值
Inputs:
frequence: 频率矩阵，backoff_unigram:一维数组unigram
Returns
bigram: 概率矩阵
'''

def absolute_discouting(frequence, backoff_unigram):
    b = 0.75
    lec = len(frequence)
    bigram = np.zeros((lec, lec))
    for i in range(lec-1):
        N = sum(frequence[i])
        num = 0
        for j in range(1, lec):
            if frequence[i][j] != 0:
                num += 1
        for j in range(1, lec):
            if frequence[i][j] != 0:
                bigram[i][j] = (frequence[i][j] - b) / N + b / N * num * backoff_unigram[j]
            else:
                bigram[i][j] = b / N * num * backoff_unigram[j]
        bigram[i] = bigram[i] / sum(bigram[i])
    return bigram
        
'''
线性减值
Inputs:
frequence: 频率矩阵，backoff_unigram:一维数组unigram
Returns
bigram: 概率矩阵
'''
    
def linear_discouting(frequence):
    lec = len(frequence)
    bigram = np.zeros((lec, lec))
    for i in range(lec-1):
        r1_count = 0
        num = 0
        for j in range(1, lec):
            if frequence[i][j] != 0:
                num += 1
            if frequence[i][j] == 1:
                r1_count += 1
        N = sum(frequence[i])
        alpha = r1_count / N
        if(alpha == 0 or alpha > 0.75):
            alpha = 0.25
        for j in range(1, lec):
            if frequence[i][j] != 0: 
                bigram[i][j] = frequence[i][j] * (1 - alpha) / N
            else:
                bigram[i][j] = alpha / (lec - 1 - num)
    return bigram

'''
Kneser_Ney
Inputs:
frequence: 频率矩阵，counter，word2id
Returns
bigram: 概率矩阵
'''
    
def Kneser_Ney(frequence, counter, word2id):
    backoff_unigram = get_backoff_unigram(counter, word2id)
    K = {x[0]:x[1] for x in counter}
    count = [0] * len(word2id)
    for word, x in K.items():
        count[word2id[word]] = x    
    
    b = 0.75
    lec = len(frequence)
    bigram = np.zeros((lec, lec))
    kn2 = 0
    for i in range(lec-1):
        for j in range(1, lec):
            if frequence[i][j] != 0:
                kn2 += 1
    
    for i in range(lec-1):
        N = sum(frequence[i])
        num = 0
        for j in range(1, lec):
            if frequence[i][j] != 0:
                num += 1
        for j in range(1, lec):
            kn = count[j]
            pkn = kn / kn2
            if frequence[i][j] != 0:
                bigram[i][j] = (frequence[i][j] - b) / N + b / N * num * pkn
            else:
                bigram[i][j] = b / N * num * pkn
        bigram[i] = bigram[i] / sum(bigram[i])
    return bigram
                

'''
处理训练数据
Inputs:
frequence: 训练文本.txt
Returns: frequence, word2id, counter, oov
'''

def get_train(train_file):
    flist = f_original_shape(train_file)
    counter = generate_counter_list(flist)
    word2id = get_word2id(counter)
    oov = select_oov(counter)
    frequence = get_frequence(word2id, flist)
    return frequence, word2id, counter, oov


'''
选择平滑方法计算概率矩阵
Inputs:smooth：平滑方法, frequence, word2id, counter
Returns: bigram:概率矩阵
'''
def get_bigram(smooth, frequence, word2id, counter):
    if smooth == add_one or smooth == good_turing or smooth ==linear_discouting:
        bigram = smooth(frequence)
    elif smooth == Katz or smooth == absolute_discouting:
        backoff_unigram = get_backoff_unigram(counter, word2id)
        bigram = smooth(frequence, backoff_unigram)
    elif smooth == Kneser_Ney:
        bigram = smooth(frequence, counter, word2id)
    else:
        return print("false")
    return bigram

'''
计算测试集
Inputs:smooth：test_file：测试集, bigram, word2id, oov
Returns: n 单词数, res2 log概率, pp 困惑熵, count 未登录词数
'''

def get_test(test_file, bigram, word2id, oov):
    res2, n, count = prob_bigram_T(test_file,word2id,bigram, oov)
    cross_entropy_res=cross_entropy(n,res2)
    pp=perplexity(cross_entropy_res)
    return n, res2, pp, count


'''
测试
'''

train_file = "train.txt"
test_txt = 'test.txt'
frequence, word2id, counter, oov = get_train(train_file)

bigram1 = get_bigram(add_one, frequence, word2id, counter)#加一法
n, res2, pp, count = get_test(test_txt, bigram1, word2id, oov)
print("单词数：", n)
print("加一法 log概率:",res2)
print("加一法 困惑熵:",pp)
print("未登录词：", count)

np.savetxt("addone_data.txt", bigram1,fmt='%f',delimiter=',')
with open("addone_word2id.txt",'w+') as fw:
    fw.write(str(word2id)) 


bigram2 = get_bigram(good_turing, frequence, word2id, counter) #good_turing
n, res2, pp, count = get_test(test_txt, bigram2, word2id, oov)
print("单词数：", n)
print("good_turing log概率:",res2)
print("good_turing 困惑熵:",pp)
print("未登录词：", count)

np.savetxt("good_turing_data.txt", bigram2,fmt='%f',delimiter=',')
with open("good_turing_word2id.txt",'w+') as fw:
    fw.write(str(word2id)) 

bigram3 = get_bigram(Katz, frequence, word2id, counter)#Katz
n, res2, pp, count = get_test(test_txt, bigram3, word2id, oov)
print("单词数：", n)
print("Katz log概率:",res2)
print("Katz 困惑熵:",pp)
print("未登录词：", count)

np.savetxt("katz_data.txt", bigram3,fmt='%f',delimiter=',')
with open("katz_word2id.txt",'w+') as fw:
    fw.write(str(word2id)) 

bigram4 = get_bigram(absolute_discouting, frequence, word2id, counter)#绝对减值
n, res2, pp, count = get_test(test_txt, bigram4, word2id, oov)
print("单词数：", n)
print("绝对减值 log概率:",res2)
print("绝对减值 困惑熵:",pp)
print("未登录词：", count)

np.savetxt("absolute_discouting_data.txt", bigram4,fmt='%f',delimiter=',')
with open("absolute_discouting_word2id.txt",'w+') as fw:
    fw.write(str(word2id)) 

bigram5 = get_bigram(linear_discouting, frequence, word2id, counter)#线性减值
n, res2, pp, count = get_test(test_txt, bigram5, word2id, oov)
print("单词数：", n)
print("线性减值 log概率:",res2)
print("线性减值 困惑熵:",pp)
print("未登录词：", count)

np.savetxt("linear_discouting_data.txt", bigram5,fmt='%f',delimiter=',')
with open("linear_discouting_word2id.txt",'w+') as fw:
    fw.write(str(word2id)) 

bigram6 = get_bigram(Kneser_Ney, frequence, word2id, counter)#Kneser_Ney
n, res2, pp, count = get_test(test_txt, bigram6, word2id, oov)
print("单词数：", n)
print("Kneser_Ney log概率:",res2)
print("Kneser_Ney 困惑熵:",pp)
print("未登录词：", count)

np.savetxt("kneser_ney_data.txt", bigram6,fmt='%f',delimiter=',')
with open("kneser_ney_word2id.txt",'w+') as fw:
    fw.write(str(word2id)) 