import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from textstat import flesch_kincaid_grade
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import pickle


model = pickle.load(open('Pickled/ethics_model.pickle', 'rb'))

t_vectorizer = pickle.load(open("Pickled/t_vectorizer.pickle", "rb"))

t_bigrams = pickle.load(open("Pickled/t_bigrams.pickle", "rb"))

a_vectorizer = pickle.load(open("Pickled/a_vectorizer.pickle", "rb"))

a_bigrams = pickle.load(open("Pickled/a_bigrams.pickle", "rb"))

thresh = 0.571886953922783



def preprocess(s):
    """This function converts a string to lowercase, removes punctuation, and removes opening/trailing whitespaces"""
    # convert string s to lowercase
    out = s.lower()
    # remove punctuation
    out = re.sub(r'[^\w\s]','',out)
    # remove opening/trailing whitespace
    out = out.strip()
    return out


def lemmatize(text):
    """This function lemmatizes the string text"""
    lemmatizer = WordNetLemmatizer()
    tokenized = word_tokenize(text)
    lemma = [lemmatizer.lemmatize(word) for word in tokenized]
    lemmatized_text = ' '.join(lemma)
    return lemmatized_text
    
def proba_rescale(proba):
	minimum = 0.060509
	maximum = 0.914831
	if proba >= thresh:
		rescale = 50 + 100 * (proba - thresh)/(maximum-thresh)
	else:
		rescale = -50 + 100 * (proba - minimum)/(thresh-minimum)
	rescale = max(0, rescale)
	rescale = min(100, rescale)
	return rescale


def predict_ethics(advice):
    """Function returns prediction on whether the string variable advice
    is ethical or not based the trained RandomForestClassifier estimator."""
    
    # preprocess and lemmatize the string
    
    original_advice = ''.join([c for c in advice])
    
    advice = preprocess(advice)
    advice = lemmatize(advice)
    
    
    # obtain necessary series
    title = pd.Series([advice])
    alltext = pd.Series([advice])
    
    # vectorize using the trained vectorizers
    vectorized_title = t_vectorizer.transform(title)
    bigram_title = t_bigrams.transform(title)
    vectorized_alltext = a_vectorizer.transform(alltext)
    bigram_alltext = a_bigrams.transform(alltext)
    
    # obtain reading level
    RL = np.array([flesch_kincaid_grade(advice)]).reshape(-1,1)
    
    # form input matrix
    input_matrix = hstack([vectorized_title,bigram_title,vectorized_alltext,bigram_alltext,RL])
    
    proba = model.predict_proba(input_matrix)[0,1]
    
    if proba >= thresh:
        return True, 'Advice: %s'%(original_advice), 'This advice is unethical with Unethical Score %s%%.'%(round(proba_rescale(proba),1))
    else:
        return False, 'Advice: %s'%(original_advice), 'This advice is not unethical with Unethical Score %s%%.'%(round(proba_rescale(proba),1))





