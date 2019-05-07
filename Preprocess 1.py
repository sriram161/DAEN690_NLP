# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Load the Pandas libraries with alias 'pd' 
import pandas as pd
import chardet
import re
import emoji
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

with open("Management Growth Buzz - Data.csv", 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large
data=pd.read_csv("Management Growth Buzz - Data.csv", encoding=result['encoding'])

# To analyse the special puntuations(emoticons and emojis) and replacing 
# them with the associated words,
# We have taken the code from a github repository
# https://github.com/aritter/twitter_nlp/blob/master/python/emoticons.py
mycompile = lambda pat:  re.compile(pat,  re.UNICODE)
#SMILEY = mycompile(r'[:=].{0,1}[\)dpD]')
#MULTITOK_SMILEY = mycompile(r' : [\)dp]')
NormalEyes = r'[8:=]'
Wink = r'[;*]'
NoseArea = r'(|o|O|-)'   ## rather tight precision, \S might be reasonable...
HappyMouths = r'[D\)\]]'
SadMouths = r'[\(\[]'
Tongue = r'[pP]'
OtherMouths = r'[doO/\\]'  # remove forward slash if http://'s aren't cleaned
Happy_RE =  mycompile( '(\^_\^|' + NormalEyes + NoseArea + HappyMouths + ')')
Sad_RE = mycompile(NormalEyes+ NoseArea + SadMouths )
Wink_RE = mycompile(Wink + NoseArea + HappyMouths)
Tongue_RE = mycompile(NormalEyes + NoseArea + Tongue)

Emoticon = (
    "("+NormalEyes+"|"+Wink+")" +
    NoseArea + 
    "("+Tongue+"|"+OtherMouths+"|"+SadMouths+"|"+HappyMouths+")"
)
Emoticon_RE = mycompile(Emoticon)
#Emoticon_RE = "|".join([Happy_RE,Sad_RE,Wink_RE,Tongue_RE,Other_RE])
#Emoticon_RE = mycompile(Emoticon_RE)
# Till this part, we have taken the work from the above mentioned URL

# Below is the function that we have made using the above code
def analyze_special_punctuations(text):
    text=emoji.demojize(text)
    text=re.sub(Happy_RE, r'Happy', text)
    text=re.sub(Sad_RE, r'Sad', text)
    text=re.sub(Wink_RE, r'Wink', text)
    text=re.sub(Tongue_RE, r'Tongue', text)
    return text

#Creating a stop words list of our own
stops=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "between", "during", "before", "after","from", "up", "down", "in", "to", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each","other", "own", "same", "so", "than", "too", "very", "can", "will", "just", "do", "should", "now"]

#Replace apostrophe/short words
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def cleaning_review(review):
    # 1. Replace % sign with the word
    review_text = review.replace("%"," percent")
    review_text = review.replace("eps","Earnings per share")
    review_text = review.replace("ebitda","Earnings before interest, tax, depreciation and amortization")
    # 2. Replace contraction words
    review_text=decontracted(review_text)
    # 3. Replace emoticons with words
    review_text= analyze_special_punctuations(review_text)
    # 4. Remove Non letters and tokenize
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # 5. Convert words to lower case and split them
    review_text = review_text.lower().split()
    # 6. Removing chosen stopwords
    resultwords  = [word for word in review_text if word not in stops]      
    # 7. Lemmatizing the data
    wordnet_lemmatizer = WordNetLemmatizer()
    words = [wordnet_lemmatizer.lemmatize(i,'v') for i in resultwords]
    # 8. Joining the words to form sentences
    return(" ".join(words))

sentences=list(data['Sentences'])
preprocessed=[]
for i in range(len(sentences)):
   preprocessed.append(cleaning_review(sentences[i]))
   
df1=pd.DataFrame(preprocessed)
df1.columns=['Preprocessed']
data['Sentences']=df1['Preprocessed']
data.to_csv('Management Growth Buzz - Data_preprocessed.csv', index=False)