# baseline-srilm

下载srilm toolkit http://www.speech.sri.com/projects/srilm
```
$ cd $SRILM
$ git clone https://github.com/nuance1979/srilm-python
```
然后
```
$ cd srilm-python
$ make
```
运行示例
```
$ ./example.sh
```
实验结果
```
$ ./example.sh 3 wsj/dict wsj/text.00-20 wsj/text.21-22 wsj/text.23-24 2>/dev/null
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
$ ./example.py --order 3 --vocab wsj/dict --train wsj/text.00-20 --heldout wsj/text.21-22 --test wsj/text.23-24 2>/dev/null
Ngram LM with Good-Turing discount: logprob = -182850.498553 denom = 82430.0 ppl = 165.291999209
Ngram LM with Witten-Bell discount: logprob = -183186.586563 denom = 82430.0 ppl = 166.851104561
Ngram LM with Kneser-Ney discount: logprob = -179527.687043 denom = 82430.0 ppl = 150.64028419
Ngram LM with Chen-Goodman discount: logprob = -178963.100995 denom = 82430.0 ppl = 148.283165135
Ngram LM with Jelinek-Mercer smoothing: logprob = -184712.194621 denom = 82430.0 ppl = 174.115329327
MaxEnt LM: logprob = -178740.10768 denom = 82430.0 ppl = 147.362371816
```
API示例
```
$ python3
...
>>> import srilm
>>> help(srilm.vocab.Vocab)
```


