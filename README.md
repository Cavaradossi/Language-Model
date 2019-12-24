# Language_Model
# 乔姆老斯基 语言模型



## smooth.py
author: 卢裕铭  
实现了几个平滑方法的2gram模型，包括：古德-图灵平滑、Katz平滑、绝对减值平滑、线性减值平滑、Kneser-Ney平滑

## Kneser_Ney_ngram.py
author: 卢裕铭  
实现了ngram的Kneser-Ney平滑

## jisuan.py
author: 卢裕铭  
网页后端访问该文件时读取数据，计算结果

## delete interpolation
author:王齐茁
  
直接调用delete_interpolation.py即可。
本方法将使用train.txt生成初始数据，使用test.txt更新参数
运行后会生成bigram概率矩阵，结果存放在new_bigram.txt


## RNN_Language_Model
author:王齐茁
  
直接调用run.py将对ptb/test进行测试
运行后困惑度结果存放在log/ppl_result，预测词概率结果存放在log/prob_result
将run.py中参数TRAIN设置为True可使用ptb/train进行训练
将run.py中参数VERBOSE设置为True可在测试时打印每句话的困惑度
run.py中函数test(sentence)可用于测试单个句子并输出结果


## baseline-srilm
author:余政泽
下载srilm toolkit http://www.speech.sri.com/projects/srilm

 ```
 cd $SRILM
 git clone https://github.com/nuance1979/srilm-python
 ```
 然后
 ```
 cd srilm-python
make
```
运行示例
```
./example.sh
```
实验结果
```
./example.sh 3 wsj/dict wsj/text.00-20 wsj/text.21-22 wsj/text.23-24 2>/dev/null
Ngram LM with Good-Turing discount:
file /home/yisu/Work/data/wsj/text.23-24: 3761 sentences, 78669 words, 0 OOVs
0 zeroprobs, logprob= -182850 ppl= 165.292 ppl1= 211.009
Ngram LM with Witten-Bell discount:
file /home/yisu/Work/data/wsj/text.23-24: 3761 sentences, 78669 words, 0 OOVs
0 zeroprobs, logprob= -183187 ppl= 166.851 ppl1= 213.095
Ngram LM with Kneser-Ney discount:
file /home/yisu/Work/data/wsj/text.23-24: 3761 sentences, 78669 words, 0 OOVs
0 zeroprobs, logprob= -179528 ppl= 150.64 ppl1= 191.454
Ngram LM with Chen-Goodman discount:
file /home/yisu/Work/data/wsj/text.23-24: 3761 sentences, 78669 words, 0 OOVs
0 zeroprobs, logprob= -178963 ppl= 148.283 ppl1= 188.316
Ngram LM with Jelinek-Mercer smoothing:
file /home/yisu/Work/data/wsj/text.23-24: 3761 sentences, 78669 words, 0 OOVs
0 zeroprobs, logprob= -184712 ppl= 174.115 ppl1= 222.826
MaxEnt LM:
file /home/yisu/Work/data/wsj/text.23-24: 3761 sentences, 78669 words, 0 OOVs
0 zeroprobs, logprob= -178757 ppl= 147.433 ppl1= 187.185
```
example.py
```
./example.py --order 3 --vocab wsj/dict --train wsj/text.00-20 --heldout wsj/text.21-22 --test wsj/text.23-24 2>/dev/null
Ngram LM with Good-Turing discount: logprob = -182850.498553 denom = 82430.0 ppl = 165.291999209
Ngram LM with Witten-Bell discount: logprob = -183186.586563 denom = 82430.0 ppl = 166.851104561
Ngram LM with Kneser-Ney discount: logprob = -179527.687043 denom = 82430.0 ppl = 150.64028419
Ngram LM with Chen-Goodman discount: logprob = -178963.100995 denom = 82430.0 ppl = 148.283165135
Ngram LM with Jelinek-Mercer smoothing: logprob = -184712.194621 denom = 82430.0 ppl = 174.115329327
MaxEnt LM: logprob = -178740.10768 denom = 82430.0 ppl = 147.362371816
```

API示例
```
python3
...
>>> import srilm
>>> help(srilm.vocab.Vocab)
```



## common.py

author:朱靖雯

运行示例
```
run common.py
```

实现内容(按照函数先后顺序的功能描述):

1. 调用预处理函数,输入数据文件,得到预处理后的文件flist,以list格式存储.
2. 统计f_list中各单词的个数,以list格式输出,counter:[(word,times),(word,times),...,(word,times)]
3. 输入counter,输出词与id的对应word2id,以set格式输出
4. 输入word2id,输出id与词的对应,以set格式输出
5. 输入word2id,flist,输出由两两词同时出现的次数的二维数组
6. 输入word2id,flist,输出两两词同时出现概率(bigram)的二维数组
7. 输入counter,输出unigram的1维数组
8. 输入测试句子,训练集的word2id,训练集得到的unigram,输出测试句子的unigram概率的log值
9. 输入输入测试句子,训练集的word2id,训练集得到的unigram,bigram,输出测试句子的bigram概率的log值
10. 输入测试语料库,训练集的word2id,unigram,bigram,输出测试语料库的概率的log值
11. 输入文件f,输出文件f中总共的单词数(不包含"\n")
12. 输入总单词数,语料库概率的log值,输出语料库的交叉熵
13. 输入语料库的交叉熵,输出语料库的困惑度
14. 计算bigram概率矩阵每一行的概率概率是否为1,输入bigram概率文件f,返回0,打印每行的概率和(用于测试生成的bigram是否合理)
15. 读取bigram概率矩阵文件,输入bigram概率矩阵文件,返回bigram概率矩阵
16. main函数是各函数的使用示例



## add smooting

author:朱靖雯

运行示例
```
run add_smoothing.py
```


实现内容(按照函数先后顺序的功能描述):

1. add_one_smoothing函数,实现加一算法,输入word2id,flist,输出加一算法平滑后的bigram矩阵二维数组
2. main函数,加一算法从训练集到测试集困惑熵的计算示例
