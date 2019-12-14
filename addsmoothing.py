import math
import sympy
from ngram_generator import ngram_generator
import ngram_generator
from collections import Counter

'''
Inputs: 
vocab_num: 词汇数量
word_num: 在给定历史条件下单词出现的次数
hist_num: 历史条件出现的次数

Returns:
new_p: 加值平滑后的词汇条件概率
'''
def add_smoothing(vocab_num, word_num, hist_num):

    new_p = (word_num + 1)/(vocab_num + hist_num)
    return new_p

'''
要点：具有相同历史的gram对应的参数λ相同，要先计算相同历史的gram出现的次数，然后计算概率乘积，对参数求偏导得到极值点
Inputs: 
gram_num: gram的数量
p_list: 不同gram的值列表 shape = (gram_num, probability_num)

Returns:
p_2gram_matrix: 插值平滑一轮后的gram概率矩阵
'''
def delete_interpolation(gram_num, p_list):
    p_1gram_matrix = p_list[0]
    if gram_num == 2:
        p_2gram_matrix = p_list[1]
        '''
    if gram_num == 3:
        p_2gram_matrix = p_list[1]
        p_3gram_matrix = p_list[2]
        '''

    with open('rest.txt','r') as f:
        gram_list = generate_gram_list(f)
        corpus_p = cal_corpus_p(f, p_2gram_matrix)
        vocab_num = get_vocab_num(f)
    old_cross_entropy = -1
    new_cross_entropy = cross_entropy(vocab_num, corpus_p)

    while(old_cross_entropy != new_cross_entropy):
        old_cross_entropy = new_cross_entropy
        p_2gram_matrix = round(gram_list, p_1gram_matrix, p_2gram_matrix)
        new_cross_entropy = cross_entropy(corpus_p, p_2gram_matrix)

    return p_2gram_matrix




'''
一轮参数更新
Inputs:
gram_list, shape = (2 * gram_tuple_num)
gram_list[n][0]是gram内容
gram_list[n][1]是出现次数
p_1gram_matrix: unigram概率矩阵
p_2gram_matrix: bigram概率矩阵

Returns:
p_2gram_matrix: 更新的bigram概率矩阵
'''
def round(gram_list, p_1gram_matrix, p_2gram_matrix):
    p_case = {}

    for i in range(len(gram_list)):
        gram = gram_list[i][0]
        gram_count = gram_list[i][1]
        gram_split = gram.split()
        '''
        word = gram_split[2]
        history2 = ' '.join(gram_split[0], gram_split[1])
        history1 = gram_split[1]
        '''
        word = gram_split[1]
        history = gram_split[0]
        if history not in p_case:
            p_case[history] = 1

        x = sympy.symbols("x")
        p_1gram = get_p_with_gram(word, p_1gram_matrix)
        p_2gram = get_p_with_gram(gram, p_2gram_matrix)
        p_new = x * p_1gram + (1-x) * p_2gram
        p_case[history] = p_case[history] * pow(p_new, gram_count)
        #p_3gram = get_p()

    for history in p_case:
        x_result = 0
        p_case_t = p_case[history].subs(x, x_result)
        difpx = sympy.diff(p_case[history], x)
        t = sympy.solve(difpx,x)
        for x_candidate in t:
            if  0 < x_candidate < 1 and p_case[history].subs(x, x_candidate) < p_case_t:
                p_case_t = p_case[history].subs(x, x_candidate)
                x_result = x_candidate
        p_2gram_matrix = update_gram_matrix_by_history(x_result, history,p_2gram_matrix)

    return p_2gram_matrix

'''
该函数计算数据集的bigram概率
Inputs:
f: 数据集文件
p_2gram_matrix: bigram概率矩阵

Returns:
cal_corpus_p: 数据集的概率
'''
def cal_corpus_p(f, p_2gram_matrix):
    '''
    TODO
    '''
    return cal_corpus_p


'''
该函数从数据集中返回词汇数目
Inputs:
f: 数据集文件

Returns:
vocab_num: 词汇数目
'''
def get_vocab_num(f):
    return vocab_num


'''
该函数以插值参数x更新gram概率矩阵
Inputs:
x: 插值参数
history: gram历史
p_matrix: gram概率矩阵

Returns:
updated_gram_matrix: 更新过的gram概率矩阵
'''
def update_gram_matrix_by_history(x, history, p_matrix):
    '''
    TODO
    '''
    return updated_gram_matrix


'''
该函数从gram矩阵中返回指定gram的概率
Inputs:
gram: gram的内容
gram_matrix: gram概率矩阵

Returns:
p: gram的概率
'''
def get_p_with_gram(gram, gram_matrix):
    '''
    TODO
    '''
    return p


'''
该函数从文件生成制定格式的gram内容及出现次数列表
Returns: gram_list, shape = (2 * gram_tuple_num)
gram_list[n][0]是gram内容
gram_list[n][1]是出现次数
'''
def generate_gram_list():
    flist=f_original_shape()
    counter = Counter()  # 词频统计
    for sentence in flist:
        for word in sentence:
            counter[word] += 1  # 计算每个字的出现次数{"word1":times,"word2":times2,...}
    counter = counter.most_common()  # 将上面的结果排序{"word1":top_times,"word2":top2_times,...}
    lec = len(counter)  # counter中word的个数
    gram_list = {counter[i][0]: i for i in range(lec)}  # {'的': 0, '很': 1, '菜': 2, '她': 3, '好': 4, '他': 5, '香': 6}

    return gram_list






'''
Inputs:
vocab_num: 词汇数量
corpus_p: 整个测试集的概率

Returns: 交叉熵
'''
def cross_entropy(vocab_num, corpus_p):
    cross_entropy = -(1/vocab_num) * math.log2(corpus_p)
    return cross_entropy


'''
Inputs:
cross_entropy: 模型交叉熵

Returns
perplexity: 模型困惑度
'''
def perplexity(cross_entropy):

    perplexity = 2**cross_entropy
    return perplexity


if __name__ == "__main__":
    delete_interpolation(0, 0)