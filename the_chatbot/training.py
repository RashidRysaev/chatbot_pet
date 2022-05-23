import json
import numpy as np

from nltk_utils import tokenize, stem, bag_of_words


with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy_data = []


for intent in intents.get('intents'):
    tag = intent.get('tag')
    tags.append(tag)
    for pattern in intent.get('patterns'):
        pattern_tokenized = tokenize(pattern)
        all_words.extend(pattern_tokenized)
        xy_data.append((pattern_tokenized, tag))

ignore_marks = ['!', '?', ';', '.', ',']

all_words = [stem(word) for word in all_words if word not in ignore_marks]
all_words = sorted(set(all_words))

tags = sorted(set(tags))

x_train = []
y_train = []

for (pattern_sencence, tag) in xy_data:
    bag = bag_of_words(pattern_sencence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)


x_train = np.array(x_train)
y_train = np.array(y_train)
