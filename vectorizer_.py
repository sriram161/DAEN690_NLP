import nltk
import yaml
import pandas as pd

with open('C:/Users/notme/Documents/Development/DAEN690_NPL/data_prep/data.yml', 'r') as infile:
    word_lists = yaml.load(infile)

tag_df = pd.read_csv('C:/Users/notme/Documents/Development/DAEN690_NPL/app/data/tagme.csv')
word_lists['tag'] = list(set(tag_df['Word'].tolist()))

df = pd.read_csv('C:/Users/notme/Documents/Development/DAEN690_NPL/app/data/Management Growth Buzz - Data_preprocessed.csv')

df_new = df[['Sentences', 'Relevance', 'Sentiment']]

# Talha word list relevance.
relevance = []
for i in range(len(df_new['Relevance'])):
    if (df_new['Relevance'][i] != "non_relevant"):
        relevance.append("relevant")
    else:
        relevance.append("non_relevant")

df_new['new_relevance'] = relevance

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def donald_frame(words, df):
    cv = CountVectorizer(vocabulary=words)
    X = cv.fit_transform(df['Sentences'])
    # np.savetxt("foo.csv", X.toarray(), delimiter=",")
    d = pd.DataFrame(X.toarray())
    d.columns = cv.get_feature_names()
    relavent = d[list(set(d.columns) - {'total', 'Sentences'})]
    l = relavent.sum(axis=0)
    cols = [item[0] for item in zip(list(l.index), l.tolist()) if item[1] != 0]
    return relavent[cols]

pos_df = donald_frame(word_lists['pos'], df_new)
neg_df = donald_frame(word_lists['neg'], df_new)
lit_df = donald_frame(word_lists['lit'], df_new)
unc_df = donald_frame(word_lists['unc'], df_new)
tag_df = donald_frame(word_lists['tag'], df_new)

pos_df['Sentences'] = df_new['Sentences']
neg_df['Sentences'] = df_new['Sentences']
lit_df['Sentences'] = df_new['Sentences']
unc_df['Sentences'] = df_new['Sentences']
tag_df['Sentences'] = df_new['Sentences']

pos_df['Relevance'] = df_new['new_relevance']
neg_df['Relevance'] = df_new['new_relevance']
lit_df['Relevance'] = df_new['new_relevance']
unc_df['Relevance'] = df_new['new_relevance']
tag_df['Relevance'] = df_new['new_relevance']

pos_df['Sentiment'] = df_new['Sentiment']
neg_df['Sentiment'] = df_new['Sentiment']
lit_df['Sentiment'] = df_new['Sentiment']
unc_df['Sentiment'] = df_new['Sentiment']
tag_df['Sentiment'] = df_new['Sentiment']

pos_df.to_csv('pos.csv')
neg_df.to_csv('neg.csv')
lit_df.to_csv('lit.csv')
unc_df.to_csv('unc.csv')
tag_df.to_csv('tag.csv')

# drop columns from talha word list.
exclude = ['sentiment_negative', 'sentiment_neutral ', 'sentiment_positive', 'sentiment_NA', 'not_relevant', 'relevant', 'very_relevant', 'extremely_relevant']
df = df.drop(exclude, axis=1)
df['Relevance'] = df_new['new_relevance']
df.to_csv('thala.csv')

bag_df = pd.read_csv("C:/Users/notme/Documents/Development/DAEN690_NPL/app/data/bog_features.csv")
bag_df['Relevance'] = df_new['new_relevance']
bag_df.to_csv('bagfw.csv')

