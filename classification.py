# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import chardet
import pandas as pd
from nltk.stem import WordNetLemmatizer
from pandas_ml import ConfusionMatrix
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import numpy as np

with open("Management Growth Buzz - Data_preprocessed.csv", 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large
data=pd.read_csv("Management Growth Buzz - Data_preprocessed.csv", encoding=result['encoding'])

#to determine the index number where the relevance changing from relevance
#to non-relevance
for i in range(len(data['Relevance'])):
    if (data['Relevance'][i]=="non_relevant"):
        a=i
        break

final_sentence=list(data['Sentences'])

relevance=[]
for i in range(len(data['Relevance'])):
    if (data['Relevance'][i]=="extremely_relevant" or data['Relevance'][i]=="very_relevant" or data['Relevance'][i]=="relevant"):
        relevance.append("relevant")
    else:
        relevance.append("non_relevant")
        
sentiment=list(data['Sentiment'])

relevant_sentences=final_sentence[0:a]
non_relevant_sentences=final_sentence[a+1:-1]
relevant_sentiment=sentiment[0:a]
non_relevant_sentiment=sentiment[a+1:-1]
relevant_relevance=relevance[0:a]
non_relevant_relevance=relevance[a+1:-1]

X_train, X_test, y_train, y_test,z_train,z_test = train_test_split(final_sentence, relevance, sentiment, test_size=0.3)

print ("\nCreating the bag of words...\n")
# Initialize bag of words tool from scikit-learn's
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,\
                             preprocessor = None,stop_words = None,max_features = 5000) 
# fit_transform() does two functions: 
# First, it fits the model and learns the vocabulary; 
# second, it transforms our training data into feature vectors. 
# The input to fit_transform should be a list of strings.
train_data_features = vectorizer.fit_transform(X_train)
# Numpy arrays are easy to work with, so convert the result to an array
train_data_features = train_data_features.toarray()
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(X_test)
test_data_features = test_data_features.toarray()
print("\nBag of words is created.")

###################
# Maximum Entropy #
###################
print ("\nTraining the maximum entropy classifier...")
# We'll use LogisticRegressionCV
# As, Logistic Regression CV aka MaxEnt classifier
# Fit the model to the training set, using the bag of words
maxent = LogisticRegression(solver='sag',\
                              max_iter=100,\
                              fit_intercept=True)
maxent=maxent.fit(train_data_features, y_train)
result_maxent = maxent.predict(test_data_features)
print("Maximum Entropy Classifier is trained.\n")

# Calculating the accuracy and making confusion matrix
def calculate_accuracy(result):
    correct=0
    wrong=0
    # Calculating the accuracy
    for i in range(len(result)):
        if result[i]==y_test[i]:
            correct+=1
        else:
            wrong+=1

    accuracy=correct/(correct+wrong)
    print("\nThe accuracy achieved is: "+str("%.2f" % (accuracy*100))+"%\n")
    print("Below is the confusion matrix:\n")
    cm = ConfusionMatrix(y_test, result)
    print(cm)
    cm.print_stats()

print("\nBelow are the results of Maximum Entropy Classifier:")
calculate_accuracy(list(result_maxent))

#sent_test_data_features=[]
#sent_test=[]
#
#for i in range(len(result_maxent)):
#    if(result_maxent[i]=="relevant"):
#        sent_test_data_features.append(test_data_features[i])
#        sent_test.append(z_test[i])
#
#sent_test_data_features=np.asarray(sent_test_data_features) 
## We'll use LogisticRegressionCV
## As, Logistic Regression CV aka MaxEnt classifier
## Fit the model to the training set, using the bag of words
#maxent_sent = LogisticRegression(solver='sag',\
#                              max_iter=100,\
#                              fit_intercept=True)
#maxent_sent=maxent_sent.fit(train_data_features, z_train)
#result_maxent_sent = maxent_sent.predict(sent_test_data_features)
#print("Maximum Entropy Classifier is trained for sentiment.\n")
#
#print("\nBelow are the results of Maximum Entropy Classifier for sentiment:")
#calculate_accuracy(result_maxent_sent, sent_test)

