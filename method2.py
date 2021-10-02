# machine learning approach

import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# kaggle dataset link https://www.kaggle.com/brijeshgzp05/sentence-types-question-command-and-statement
ndf = pd.read_csv('file1.csv')  

# encoding the data
ndf['type'] = [1 if t == 'question' else 0 for t in ndf['type']]
X_train, X_test, y_train, y_test = train_test_split(ndf['statement'], ndf['type'], test_size=0.2, random_state=0)

ndf = pd.concat([X_train, y_train], axis=1)
ndf.reset_index(drop=True, inplace=True)

# function to clean the data
def sentence_clean(sentences):
    nsentences = []
    for sentence in sentences:
        sentence = re.sub("[^a-zA-Z]"," ",sentence)

        words = sentence.lower().split()

        nsentences.append(" ".join(words))
    return nsentences

# transforming the X_train
ndf['statement'] = sentence_clean(ndf['statement'])

# creating the bag of words
vectorizer = CountVectorizer(analyzer='word',
        lowercase=True,
        tokenizer=None,
        stop_words=None,
        max_features=500,
        ngram_range=(1,2))

train_features = vectorizer.fit_transform(ndf['statement'])
train_features = train_features.toarray()

# used naive bayes classifier
model = MultinomialNB(fit_prior=False,class_prior=None,alpha=0.5)
model.fit(train_features, ndf['type'])

# below are code for testing the model

# tdf = pd.concat([X_test, y_test], axis=1)
# tdf.reset_index(drop=True, inplace=True)
# print(tdf)
# test_transform = vectorizer.transform(tdf['statement'])
# test_transform = test_transform.toarray()
# predict = model.predict(test_transform)
# print(classification_report(predict , y_test))         # Got accuracy 0.90, f1-score 0.87 
# tdf['ptype'] = predict


# using the model, predicting the sentence given in text file
file = open("text.txt", "r")
sentences = file.readlines()
sentences = [sentence.strip() for sentence in sentences]

df = pd.DataFrame(sentences, columns=['sentence'])

test_transform = vectorizer.transform(df['sentence'])
test_transform = test_transform.toarray()

predict = model.predict(test_transform)

# adding the predict result to the dataframe
df['type'] = predict
# encoding it back to is_question: yes or no
df['type'] = ['yes' if x == 1 else 'no' for x in df['type']]


# df[df['type'] == 1].count()   
# total 1523 inquiry detected

# saving it to csv file
df.to_csv("result2.csv",index=False)
