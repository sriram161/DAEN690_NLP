import PyPDF2
from itertools import chain
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import POS_LIST

pdfFileObj_pos = open('app/data/Mcdonald_wordlist/LM_Positive.pdf', 'rb')
pdfFileObj_neg = open('app/data/Mcdonald_wordlist/LM_Negative.pdf', 'rb')
pdfFileObj_lit = open('app/data/Mcdonald_wordlist/LM_Litigious.pdf', 'rb')
pdfFileObj_unc = open('app/data/Mcdonald_wordlist/LM_Uncertainty.pdf', 'rb')

pdfReader_pos = PyPDF2.PdfFileReader(pdfFileObj_pos)
pdfReader_neg = PyPDF2.PdfFileReader(pdfFileObj_neg)
pdfReader_lit = PyPDF2.PdfFileReader(pdfFileObj_lit)
pdfReader_unc = PyPDF2.PdfFileReader(pdfFileObj_unc)

pdf_obj = (pdfReader_pos, pdfReader_unc, pdfReader_neg, pdfReader_lit)

def get_text(pdfReader):
    word_list = []
    for page_no in range(pdfReader.numPages):
        text = pdfReader.getPage(page_no).extractText()
        if page_no == 0:
           word_list.extend(text.split('\n')[:])
        word_list.extend(text.split('\n'))
    word_list = [word.strip(' ').strip('') for word in word_list if len(word.strip(' ').strip('')) > 0 and ' ' not in word and word != "Loughran"]
    return word_list


word_dict = dict(zip(('pos', 'unc', 'neg', 'lit'), [
                 get_text(obj) for obj in pdf_obj]))

wordnet_lemmatizer = WordNetLemmatizer()
lemma_paper_words = {key: sorted(list(set([wordnet_lemmatizer.lemmatize(word.lower()) for word in word_list]))) for key, word_list in word_dict.items()}

import yaml
with open('data.yml', 'w') as outfile:
    yaml.dump(lemma_paper_words, outfile, default_flow_style=False)

# count = CountVectorizer(vocabulary=myvocab)
# X_vectorized = count.transform(X_train)
