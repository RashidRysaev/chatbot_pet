import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return PorterStemmer().stem(word=word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(word) for word in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)

    for indx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[indx] = 1.0

    return bag
