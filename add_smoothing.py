import numpy as np

from common import *


def add_one_smoothing(word2id,flist):
    lec=len(word2id)
    add_one = np.zeros((lec, lec)) + 1e-8
    for sentence in flist:
        sentence = [word2id[w] for w in sentence]
        for i in range(1, len(sentence)):
            add_one[[sentence[i - 1]], [sentence[i]]] += 1

    for i in range(lec):
        add_one[i] = (add_one[i]+1) / (add_one[i].sum()+add_one.sum())

    return add_one


def main():
    flist = f_original_shape('train_LM.txt')
    counter=generate_gram_list(flist)
    word2id=get_word2id(counter)
    unigram=get_unigram(counter)
    add_one=add_one_smoothing(word2id,flist)
    test_txt='i am working'
    test_pre=ngram_generator(test_txt,1)
    res = prob_bigram(test_pre, word2id, unigram, add_one)  # test bigram 概率
    print(res)

if __name__ == "__main__":
    main()