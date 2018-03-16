import string
import numpy as np
import pandas as pd
import json
import nltk
import csv
from textblob import TextBlob

input_file=open('C:/Users/Arpitha Somayaji/Desktop/Fall 2017/DATA SCIENCE/Homework/Last-Try/Tweets.json', 'r')
result = []
json_decode=json.load(input_file)
#pick only the colums required like Text,Username and place
#split the places to get only states  for the states present in USA
for item in json_decode:
    my_dict={}
    my_dict['text']=item.get('text')
    my_dict['name']=item.get('user').get('name')
    if item.get('place')!= None and str(item.get('place').get('country'))=="United States":
        location=item.get('place').get('full_name').split(',')
        my_dict['state']=location[1]
    else:
        my_dict['state']="Null"
    result.append(my_dict)
df=pd.DataFrame(result)
# print(df.items)
word = []
#preprocessing requirement
p = string.punctuation
d = string.digits
table_p = str.maketrans(p, len(p) * " ")
table_d = str.maketrans(d, len(d) * " ")
stopwords = nltk.corpus.stopwords.words("english")

state_text = pd.DataFrame(columns=('State','Tweets'))
# for every row, if state in already present in the dictionary, append the list otherwise add the state and add the text
for index,row in df.iterrows():
    if((state_text['State']) == row[1]).any():
        word = row[2]
        word=word.lower()
        word=word.translate(table_p)
        word=word.translate(table_d)
        word = nltk.word_tokenize(word)
        word = [w for w in word if w not in stopwords]
        state_text['Tweets'].values[0].extend(word)
    else:
        word = row[2]
        word=word.lower()
        word=word.translate(table_p)
        word=word.translate(table_d)
        word = nltk.word_tokenize(word)
        word = [w for w in word if w not in stopwords]
        state_text.loc[state_text.size] = [row[1],word]

sentiment=[]
#compute the sentiment for each row using textblob
for index,row in state_text.iterrows():
    str=' '.join(word for word in row[1])
    row[1]=str
    sentiment.append(TextBlob(row[1]).sentiment.polarity)
#remove twitter text and add sentiment column to dataframe
print(sentiment)
state_text=state_text.assign(Sentiment=sentiment)
state_text=state_text.drop('Tweets',axis=1)
#convert dataframe into csv
state_text.to_csv('C:/Users/Arpitha Somayaji/Desktop/Fall 2017/DATA SCIENCE/Homework/Last-Try/twt_sentiment.csv')
