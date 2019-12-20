from collections import Counter
import numpy as np

from ngram_generator import ngram_generator
import math

'''
由train_data得到test_sentence的unigram概率和bigram概率
author:Zhu Jingwen
计算步骤
1. 将train_data预处理,将train_data中的单词变成原形,以列表flist输出
2.统计flist中各单词的个数,以counter:(word,times)输出
3.给单词表建立id,得到word2id:(word,id)
4.得到id2word(id,word)
5.得到二维数组,俩俩单词出现的次数
6.得到二维数组,p(w2|w1)组成的矩阵
7.得到1维数组,train_data中所有词的unigram
8.计算test sentence unigram
9.计算test sentence bigram
'''

'''
语料库预处理,将语料库按行读取(一行为一句),将所有词还原为原型,以句子列表形式输出
输入:
file_name eg.'train_LM.txt' 同一目录下的txt文件
输出:
f_list:文件中句子以列表输出,每个句子中的词都是原型
e.g. [['i', 'be', 'fine', '\n'], ['i', 'be', 'a', 'student', '\n'],...]
输入输出举例
输入
i am fine\n
i am a student\n
输出
[['i', 'be', 'fine', '\n'], ['i', 'be', 'a', 'student', '\n']]
'''


def f_original_shape(file_name):
    f_list = []
    f = open(file_name, 'r', encoding='utf-8', errors='ignore')
    lines = f.readlines()
    f.close()
    for line in lines:
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
e.g.[('be', 4), ('\n', 3), ('a', 3), ('i', 2), ('fine', 1),...]
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
'''


def get_word2id(counter):
    lec = len(counter)  # counter中word的个数
    word2id = {counter[i][0]: i for i in range(lec)}  # {'的': 0, '很': 1, '菜': 2, '她': 3, '好': 4, '他': 5, '香': 6}
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
返回bigram组成的矩阵中俩俩词的出现个数
'''


def get_bigram_times(word2id, flist):
    lec = len(word2id)
    bigram = np.zeros((lec, lec)) + 1e-8
    for sentence in flist:
        sentence = [word2id[w] for w in sentence]
        for i in range(1, len(sentence)):
            bigram[[sentence[i - 1]], [sentence[i]]] += 1

    return bigram


'''
输入:word2id,预处理的文件f_list
输出:bigram概率矩阵,举例
[[2.49999991e-09 2.49999991e-09 4.99999984e-01 2.49999991e-09
  2.49999993e-01 2.49999991e-09 2.49999991e-09 2.49999993e-01
  2.49999991e-09 2.49999991e-09 2.49999991e-09 2.49999991e-09
  2.49999991e-09 2.49999991e-09 2.49999991e-09]
 [6.66666667e-02 6.66666667e-02 6.66666667e-02 6.66666667e-02
  6.66666667e-02 6.66666667e-02 6.66666667e-02 6.66666667e-02
  6.66666667e-02 6.66666667e-02 6.66666667e-02 6.66666667e-02
  6.66666667e-02 6.66666667e-02 6.66666667e-02]
...]
'''


def get_bigram(word2id, flist):
    lec = len(word2id)
    bigram = np.zeros((lec, lec)) + 1e-8
    for sentence in flist:
        sentence = [word2id[w] for w in sentence]
        for i in range(1, len(sentence)):
            bigram[[sentence[i - 1]], [sentence[i]]] += 1

    for i in range(lec):
        bigram[i] /= bigram[i].sum()
    return bigram


'''
由counter得到训练数据的unigram概率1维数组
'''


def get_unigram(counter):
    unigram = np.array([i[1] for i in counter]) / sum(i[1] for i in counter)
    return unigram


'''
log
计算test sentence的unigram的概率
输入
test_sentence,word2id,由训练data得到的unigram单词概率1维度数组
输出
p_unigram(test_sentence)
'''


def prob_unigram(sentence, word2id, unigram):
    lec=len(word2id)
    if lec < 1:
        return 0
    p=0.00
    for w in sentence:
        if w in word2id:
            p += math.log(unigram[word2id[w]],2)


    return p

    # s = [word2id[w] for w in sentence]  # 将句子编程id序列[wordid1,wordid2,wordid3,...]
    # p = 0
    # les = len(s)  # 句子单词个数
    # if les < 1:
    #     return 0
    # for i in range(0, les):
    #     # p *= unigram[s[i]]
    #     p += math.log(unigram[s[i]])
    # return p


"""
log结果
输入
test_sentence,word2id,由训练数据得到的所有单词概率的unigram向量,bigram矩阵
输出
p_bigram(test_sentence)
"""


def prob_bigram(sentence, word2id, unigram, bigram):
    #
    # s = [word2id[w] for w in sentence]  # 将句子编程id序列[wordid1,wordid2,wordid3,...]
    p = 0.00
    les = len(sentence)  # 句子单词个数
    if les < 1:
        return 0
    if les < 2:  # 如果句子只有一个词,则值返回unigram计算结果
        # p = math.log(unigram[s[0]],2) # 对第一个词用unigram
        p=prob_unigram(sentence,word2id,unigram)
        return p
    for i in range(1,len(sentence)):
        if sentence[i] in word2id and sentence[i-1] in word2id:
            p += math.log(bigram[word2id[sentence[i - 1]], word2id[sentence[i]]], 2)
    # for i in range(1, les):  # 从第2个词开始和前一个两两组合的bigram值的乘积/如果是add_smoothing,则bigram值换成add_one矩阵代入即可
    #     p += math.log(bigram[s[i - 1], s[i]],2)
    return p


'''得到整个测试集的概率'''
def prob_bigram_T(test_filename,word2id,unigram,bigram):
    f = open(test_filename, 'r', encoding='utf-8', errors='ignore')
    lines = f.readlines()
    f.close()
    p=0
    for line in lines:
        line = ngram_generator(line, 1)
        res2 = prob_bigram(line, word2id, unigram, bigram)
        p+=res2

    return p


'''
该函数从数据集中返回词汇数目
Inputs:
f: 数据集文件

Returns:
vocab_num: 词汇数目
'''


def get_vocab_num(f):
    f_list = f_original_shape(f)
    vocab_num=0
    for sentence in f_list:
        vocab_num-=1  # 计算中每个句子多了一个结尾符号,所以每句话都-1
        for w in sentence:
            vocab_num+=1

    return vocab_num



'''
Inputs:
vocab_num: 词汇数量
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




'''
使用举例
通过训练数据得到test sentence的unigram句子概率和bigram句子概率
'''


def main():

    flist = f_original_shape('train_LM.txt')
    counter = generate_counter_list(flist)
    word2id = get_word2id(counter)
    unigram = get_unigram(counter)
    bigram = get_bigram(word2id, flist)
    test_txt='test_LM.txt'
    res_T=prob_bigram_T(test_txt,word2id,unigram,bigram)
    print(test_txt,"log概率:",res_T)


    # test_txt = 'i am working'
    # test_pre = ngram_generator(test_txt, 1)
    # print("方法2:",test_pre)
    # res1 = prob_unigram(test_pre, word2id, unigram)  # test unigram概率
    # res2 = prob_bigram(test_pre, word2id, unigram, bigram)  # test bigram 概率
    # cross_entropy_res=cross_entropy(3,res2)
    # pp=perplexity(cross_entropy_res)
    # print("unigram log概率:",res1)
    # print("bigram log概率:",res2)
    # print("bigram 困惑熵:",pp)


if __name__ == "__main__":
    main()
