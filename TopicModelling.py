
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import glob
import os
import string
import nltk
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import scipy.stats as stats
import json



corpus = []

input_file=open('C:/Users/Arpitha Somayaji/Desktop/Fall 2017/DATA SCIENCE/Homework/Last-Try/Tweets.json', 'r')

json_decode=json.load(input_file)

for item in json_decode:
    corpus.append(item.get('text'))



vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 2)
dtm = vectorizer.fit_transform(corpus)
# names = [fn[:fn.find(".")] for fn in files] #get file names without .txt
from sklearn import decomposition

num_topics = 10
num_top_words = 20
clf = decomposition.NMF(n_components = num_topics, random_state=1)
doctopic = clf.fit_transform(dtm)
topic_words = []
for topic in clf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([corpus[i] for i in word_idx])
for t in range(len(topic_words)):
    print("Topic {}: {}".format(t, ' '.join(topic_words[t][:15])))

