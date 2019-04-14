import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# DATASET HAS BEEN TAKEN FROM  - https://www.kaggle.com/uciml/sms-spam-collection-dataset
df = pd.read_csv('spam.csv',encoding='ISO-8859-1')
df = df[['v1','v2']]
df.columns = ['labels', 'data']
df['labels'] = pd.get_dummies(df['labels'],drop_first=True) #making ham as 0 and spam as 1

count_vectorizer = CountVectorizer(decode_error='ignore')
X = count_vectorizer.fit_transform(df['data'])
Y = df['labels'].values

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
# print(df.head())

model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))

# visualize the data
def visualize(label,df):
    words = ''
    for msg in df[df['labels'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    if(label==1):
        plt.title('spam')
    else:
        plt.title('ham')
    plt.axis('off')
    plt.show()

visualize(1,df)
visualize(0,df)


# see what we're getting wrong
df['predictions'] = model.predict(X)

# things that should be spam
sneaky_spam = df[(df['predictions'] == 0) & (df['labels'] == 1)]['data']
for msg in sneaky_spam:
  print('spam marked as ham-------->',msg)
print('\n----------------------------------------------------------------------------------------------------\n')
# things that should not be spam
not_actually_spam = df[(df['predictions'] == 1) & (df['labels'] == 0)]['data']
for msg in not_actually_spam:
  print('ham marked as spam--------->',msg)
