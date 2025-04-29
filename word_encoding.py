# For AI 680 students only
# Word encoding with word2vec

from scipy.spatial.distance import cosine

import math
import numpy as np

with open("words.txt", "r", encoding="utf-8") as wf: 
    words = dict()
    for line in wf: 
        row = line.split()
        word = row[0]
        vector = np.array([float(x) for x in row[1:]])
        words[word] = vector


def distance(w1, w2):
    return cosine(w1, w2)


def closest_words(embedding):
    distances = {
        w: distance(embedding, words[w])
        for w in words
    }
    return sorted(distances, key=lambda w: distances[w])[:10]


def closest_word(embedding):
    return closest_words(embedding)[0]


print(distance(words["lunch"], words["book"]))
print(distance(words["lunch"], words["dinner"]))

print(closest_word( words["king"] - words["man"] + words["woman"] ))
 
