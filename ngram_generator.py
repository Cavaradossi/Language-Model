import re

from nltk import WordNetLemmatizer, pos_tag, word_tokenize

from nltk.corpus import wordnet

def ngram_generator(s, n):
    if (len(s.split()) < n):
        return "ERROR, NUMBER OF GRAM EXCEED TEXT!"

    # Convert to lowercases
    s = s.lower()

    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)

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
