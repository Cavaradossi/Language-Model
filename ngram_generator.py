import re

from nltk import WordNetLemmatizer, pos_tag, word_tokenize


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
    lemmatizer = WordNetLemmatizer()
    tagged = nltk.pos_tag(token)
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
