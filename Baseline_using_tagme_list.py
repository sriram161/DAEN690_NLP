# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 00:04:08 2019

@author: Paras Sethi
"""

# Load the Pandas libraries with alias 'pd' 
import pandas as pd
import chardet
from nltk.util import ngrams
import tagme
import chardet
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB

# Set the authorization token for subsequent calls.
tagme.GCUBE_TOKEN = "f2175941-b114-4ad7-9d44-f2e5e7689527-843339462"


with open("Management Stability Buzz - Data_preprocessed.csv", 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large
data=pd.read_csv("Management Stability Buzz - Data_preprocessed.csv", encoding=result['encoding'])

final_sentence=list(data['Sentences'])

#Creating ngrams from unigram to fourgram for each sentence
def word_grams(words, min=1, max=3):
    s = []
    for n in range(min,max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s

ngram_list=[]
for i in range(len(final_sentence)):
    ngram_list.append(word_grams(final_sentence[i].split(' ')))
    
flat_list = list(set([item for sublist in ngram_list for item in sublist]))

word_list=[]
topic=[]
value=[]

for word in flat_list:
    rels = tagme.relatedness_title((word, "Stability"))
    a=rels.relatedness[0].rel
    if a is not None:
        if a>0:
            word_list.append(word)
            topic.append("Stability")
            value.append(a)
            
df_tagme=pd.DataFrame(
    {'Topic': topic,
     'Word': word_list,
     'Value': value
    })
    
df_tagme.to_csv('tagme.csv', index=False)

is_greater =  df_tagme['Value']>=0.25
df_tagme_greater= df_tagme[is_greater]
tagme_wordlist=df_tagme_greater['Word']
tagme_wordlist=list(set(tagme_wordlist))
tagme_wordlist=pd.DataFrame(np.array(tagme_wordlist),columns=["Word"])

df = pd.read_csv('Management Stability Buzz - Data_preprocessed.csv')
df_new = df[['Sentences', 'Relevance', 'Sentiment']]

## Tagme word list relevance.
relevance = []
for i in range(len(df_new['Relevance'])):
    if (df_new['Relevance'][i]=="extremely_relevant" or df_new['Relevance'][i]=="very_relevant" or df_new['Relevance'][i]=="relevant"):
        relevance.append("relevant")
    else:
        relevance.append("non_relevant")
df_new['new_relevance'] = relevance

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

tagme_df = donald_frame(tagme_wordlist['Word'], df_new)

tagme_df['Sentences'] = df_new['Sentences']
cols = tagme_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
tagme_df = tagme_df[cols]

tagme_df['Relevance'] = df_new['new_relevance']

tagme_df['Sentiment'] = df_new['Sentiment']

tagme_df.to_csv('tagme_df.csv', index=False)

final_sentence=list(tagme_df['Sentences'])

new_relevance=list(tagme_df['Relevance'])
        
sentiment=list(tagme_df['Sentiment'])

b=tagme_df.columns.get_loc("Relevance")

data_features=tagme_df[tagme_df.columns[1:b]]

X_train, X_test, y_train, y_test,z_train,z_test,train_data_features,test_data_features = train_test_split(final_sentence, new_relevance, sentiment, data_features, test_size=0.3, stratify=new_relevance)

train_data_features = train_data_features.values
test_data_features = test_data_features.values


##########################
## Logistic Regression ##
##########################
def LogRegression(train_data_features, test_data_features, train):
    maxent = LogisticRegression(solver='sag',max_iter=100,fit_intercept=True)
    maxent=maxent.fit(train_data_features, train)
    result_maxent = maxent.predict(test_data_features)
    return result_maxent

####################
##  Random Forest ##
####################
def RFForest(train_data_features, test_data_features, train):
    forest = RandomForestClassifier(n_estimators = 100) 
    forest = forest.fit(train_data_features,train)
    result_rf = forest.predict(test_data_features)
    return result_rf

###########
##  SVM ##
##########
def SVM(train_data_features, test_data_features, train):
    svclassifier = SVC(kernel='linear') 
    svclassifier = svclassifier.fit(train_data_features,train)
    result_svm = svclassifier.predict(test_data_features)
    return result_svm

###################
##  Naive Bayes ##
###################
def NB(train_data_features, test_data_features, train):
    gnb = GaussianNB()
    gnb = gnb.fit(train_data_features,train)
    result_gnb = gnb.predict(test_data_features)
    return result_gnb

# Calculating the accuracy and making confusion matrix
def calculate_accuracy(test, result):
    correct=0
    wrong=0
    # Calculating the accuracy
    for i in range(len(result)):
        if result[i]==test[i]:
            correct+=1
        else:
            wrong+=1
    accuracy=correct/(correct+wrong)
    print("\nThe accuracy achieved is: "+str("%.2f" % (accuracy*100))+"%\n")
    print("Below is the confusion matrix:\n")
    print(confusion_matrix(test,result))
    print(classification_report(test,result))

print("\nBelow are the results of Logistic Regression Classifier:")
rel_result_maxent=LogRegression(train_data_features,test_data_features,y_train)
calculate_accuracy(y_test,rel_result_maxent)
print("\nBelow are the results of Random Forest Classifier:")
rel_result_rf=RFForest(train_data_features,test_data_features,y_train)
calculate_accuracy(y_test,rel_result_rf)
print("\nBelow are the results of SVM Classifier:")
rel_result_svm=SVM(train_data_features,test_data_features,y_train)
calculate_accuracy(y_test,rel_result_svm)
print("\nBelow are the results of NAive Bayes Classifier:")
rel_result_gnb=NB(train_data_features,test_data_features,y_train)
calculate_accuracy(y_test,rel_result_gnb)

####Identifying the sentiments of the relevant sentences####

###First we need to identify the sentiments from the test data
###for which the relevance has come as relevant
def sent_features(rel_result, test_features):
    sent_test_data_features=[]
    for i in range(len(rel_result)):
        if(rel_result[i]=="relevant"):
            sent_test_data_features.append(test_features[i])
    sent_test_data_features=np.asarray(sent_test_data_features)
    return sent_test_data_features

##Finding the test set for the sentences for which relevance has come out
    ## to be relevant
def sentiment_test(rel_result, test):
    sent_test=[]
    for i in range(len(rel_result)):
        if(rel_result[i]=="relevant"):
            sent_test.append(test[i])
    return sent_test

print("\nBelow are the results of Logistic Regression Classifier for sentiments:")
sent_test_data_features=sent_features(rel_result_maxent, test_data_features)
sent_test=sentiment_test(rel_result_maxent, z_test)
sent_result_maxent=LogRegression(train_data_features,sent_test_data_features,z_train)
calculate_accuracy(sent_test,sent_result_maxent)

print("\nBelow are the results of Random Forest Classifier for sentiments:")
sent_test_data_features=sent_features(rel_result_rf, test_data_features)
sent_result_rf=RFForest(train_data_features,sent_test_data_features,z_train)
sent_test=sentiment_test(rel_result_rf, z_test)
calculate_accuracy(sent_test,sent_result_rf)

print("\nBelow are the results of SVM Classifier for sentiments:")
sent_test_data_features=sent_features(rel_result_svm, test_data_features)
sent_test=sentiment_test(rel_result_svm, z_test)
sent_result_svm=SVM(train_data_features,sent_test_data_features,z_train)
calculate_accuracy(sent_test,sent_result_svm)

print("\nBelow are the results of NAive Bayes Classifier for sentiments:")
sent_test_data_features=sent_features(rel_result_gnb, test_data_features)
sent_test=sentiment_test(rel_result_gnb, z_test)
sent_result_gnb=NB(train_data_features,sent_test_data_features,z_train)
calculate_accuracy(sent_test,sent_result_gnb)