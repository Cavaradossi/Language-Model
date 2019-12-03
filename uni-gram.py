# -*- coding: utf-8 -*-
from ngram_generator import ngram_generator
from collections import defaultdict
import re
import math


dictionary = set()              #文档词典

postings = defaultdict(dict)    #存放词频信息的postings

total_num = 0                   #总词频数

num_dic = 0                     #总词个数


def main():
    global postings,total_num,num_dic,dictionary
    
    get_dic()
    cal_probability()
    num_dic = len(dictionary)
    
    print("Total number of train words:"+str(total_num))
    print("number of dictionary:"+str(num_dic))
'''
    ppl = test()
   
    print("the test PPL score is:"+str(round(ppl,5)))
'''

def get_dic():
    global dictionary,total_num,postings
    f = open('train_LM.txt','r',encoding='utf-8',errors='ignore')
    lines = f.readlines()
    f.close()

    for line in lines:
        terms = line_token(line)
        #print(terms)
        d_tnum=len(terms)#预处理后每篇文档的总词数
        #print(d_tnum)
        unique_terms = set(terms)
        dictionary = dictionary.union(unique_terms)#并入总词典
        total_num += d_tnum
        
        for term in unique_terms:
            
            c_term=terms.count(term)
            if term in postings:
                postings[term][0] += c_term
            else:
                postings[term][0] = c_term

       
    
def line_token(document):
    return ngram_generator(document, 1)

def cal_probability():
    global postings,total_num
    
    for term in postings:
        postings[term][1] = postings[term][0]/total_num
       
def get_pw_of_absent(newTerm):
    #根据在测试集中新出现的词来更新语料库的词概率信息（加 1 法）
    
    global total_num,num_dic
    
    return 1/(total_num + num_dic)
    

'''
def test():
    global postings
    log_PT = 0
    f = open('test_LM.txt','r',encoding='utf-8',errors='ignore')
    document = f.read()
    f.close()
    test_wNum = 0
    words = line_token(document)
    for expected_str in words:
        test_wNum += 1
        #加 1 法平滑
        if expected_str in postings:
            log_PT += math.log(postings[expected_str][1],2)
        else:
#            print("update_posting!!!")
#            update_postings_dic(expected_str)
#            log_PT += math.log(postings[expected_str][1],2)
            print("one not in posting!!!")
            temp = get_pw_of_absent(str)
            log_PT += math.log(temp,2)
    print("log_PT:"+str(log_PT))
    print("test_num:"+str(test_wNum))
    PPL = pow(2,-log_PT/test_wNum)  
    return PPL
'''

if __name__ == "__main__":
    main()
    print(dictionary)
    input('press enter')
