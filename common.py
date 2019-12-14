from ngram_generator import ngram_generator
from collections import Counter
import numpy as np




'''
语料库预处理,将语料库按行读取(一行为一句),将所有词还原为原型,以句子列表形式输出
输出:
f_list:文件中句子以列表输出,每个句子中的词都是原型
e.g. [['i', 'be', 'fine', '\n'], ['i', 'be', 'a', 'student', '\n'],...]
'''
def f_original_shape():
    f_list=[]
    f = open('train_LM.txt', 'r', encoding='utf-8', errors='ignore')
    lines = f.readlines()
    f.close()
    for line in lines:
        line=ngram_generator(line ,1)
        f_list.append(line)
    return f_list



'''
该函数从文件生成制定格式的gram内容及出现次数列表
Returns: gram_list, shape = (2 * gram_tuple_num)
word2id[n][0]是gram内容
word2id[n][1]是出现次数
e.g.[('be', 4), ('\n', 3), ('a', 3), ('i', 2), ('fine', 1),...]
'''
def generate_gram_list(flist):

    counter = Counter()  # 词频统计
    for sentence in flist:
        for word in sentence:
            counter[word] += 1  # 计算每个字的出现次数{"word1":times,"word2":times2,...}
    counter = counter.most_common()  # 将上面的结果排序{"word1":top_times,"word2":top2_times,...}
    return counter

def get_word2id(counter):
    lec = len(counter)  # counter中word的个数
    word2id = {counter[i][0]: i for i in range(lec)}  # {'的': 0, '很': 1, '菜': 2, '她': 3, '好': 4, '他': 5, '香': 6}
    return word2id





'''
输入:语料库处理后,word2id形式,举例
{"word1":top_times,"word2":top2_times,...}
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
def get_bigram(word2id,flist):
    lec=len(word2id)
    bigram = np.zeros((lec, lec)) + 1e-8
    for sentence in flist:
        sentence = [word2id[w] for w in sentence]
        for i in range(1, len(sentence)):
            bigram[[sentence[i - 1]], [sentence[i]]] += 1

    for i in range(lec):
        bigram[i] /= bigram[i].sum()
    return bigram



def get_unigram(counter):
    unigram = np.array([i[1] for i in counter]) / sum(i[1] for i in counter)
    return unigram

'''
计算句子unigram的概率
'''
def prob_unigram(sentence,word2id,unigram):
    s = [word2id[w] for w in sentence]  # 将句子编程id序列[wordid1,wordid2,wordid3,...]
    p = 1.00
    les = len(s)  # 句子单词个数
    if les < 1:
        return 0
    p = unigram[s[0]]
    for i in range(1, les):
        p *= unigram[s[i]]
    return p

"""test中一个句子概率,用于纯2-gram或者addsmoothing的2-gram"""
def prob(sentence,word2id,unigram,bigram):
    s = [word2id[w] for w in sentence]  # 将句子编程id序列[wordid1,wordid2,wordid3,...]
    p = 1.00
    les = len(s)  # 句子单词个数
    if les < 1:
        return 0
    # p = unigram[s[0]]
    if les < 2:  # 如果句子只有一个词,则值返回unigram计算结果
        p = unigram[s[0]] # 对第一个词用unigram
        return p
    for i in range(1, les):  # 从第2个词开始和前一个两两组合的bigram值的乘积/如果是add_smoothing,则bigram值换成add_one矩阵代入即可
        p *= bigram[s[i - 1], s[i]]
    return p



def main():
    flist = f_original_shape()
    counter=generate_gram_list(flist)
    word2id=get_word2id(counter)
    unigram=get_unigram(counter)
    bigram=get_bigram(word2id,flist)
    test_txt='i am working'
    test_pre=ngram_generator(test_txt,1)
    res=prob_unigram(test_pre,word2id,unigram)  #test unigram概率
    res=prob(test_pre,word2id,unigram,bigram)  # test bigram 概率
    print(res)

if __name__ == "__main__":
    main()
