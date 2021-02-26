import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from textstat import flesch_kincaid_grade
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import requests
import time
import json

import prediction_script

import pickle

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input_text = [x for x in request.form.values()][0]
	
    _, advice, prediction_text = prediction_script.predict_ethics(input_text)

    return render_template('index.html', advice=advice, prediction_text=prediction_text)

@app.route('/predict_lpt',methods=['POST'])
def predict_lpt():
	'''
	For retrieving a recent LPT post
	'''
	end = time.time()
	data_list_lpt = []
	df = pd.DataFrame()
	length = 10
	while df.shape[0]<length:
		url = 'https://api.pushshift.io/reddit/search/submission/?size=%s&subreddit=LifeProTips&sort=desc&sort_type=created_utc&before=%d'%(length,end)
		r = requests.get(url)
		try:
			data = r.json()
		except json.decoder.JSONDecodeError:
			break
		data_list_lpt += data['data']
		df = pd.DataFrame(data_list_lpt)
		df = df[df['title']!='LPT']
		df = df[~(df['title'].str.contains('request')) & ~(df['title'].str.contains('Request')) & ~(df['title'].str.contains('REQUEST'))]
		try:
			df = df[df['removed_by_category']!='moderator']
		except:
			pass
		try:
			df = df[df['banned_by']!='moderators']
		except:
			pass
		end = data['data'][-1]['created_utc']
			
	length = df.shape[0]
    	
	try:
		spot = np.random.randint(length)
	except ValueError:
		return render_template('index.html', advice='An error in retrieving the Reddit data ocurred, please try again.', prediction_text='')
    
	input_text = df['title'].values[spot]
    
	post_id = df['id'].values[spot]
    
	_, advice, prediction_text = prediction_script.predict_ethics(input_text)
    
	return render_template('index.html', advice='<a href="https://www.reddit.com/%s">%s</a>'%(post_id,advice), prediction_text=prediction_text)

@app.route('/predict_ulpt',methods=['POST'])
def predict_ulpt():
	'''
	For retrieving a recent ULPT post
	'''
	end = time.time()
	data_list_lpt = []
	df = pd.DataFrame()
	length = 10
	while df.shape[0]<length:
		url = 'https://api.pushshift.io/reddit/search/submission/?size=%s&subreddit=UnethicalLifeProTips&sort=desc&sort_type=created_utc&before=%d'%(length,end)
		r = requests.get(url)
		try:
			data = r.json()
		except json.decoder.JSONDecodeError:
			break
		data_list_lpt += data['data']
		df = pd.DataFrame(data_list_lpt)
		df = df[df['title']!='ULPT']
		df = df[~(df['title'].str.contains('request')) & ~(df['title'].str.contains('Request')) & ~(df['title'].str.contains('REQUEST'))]
		try:
			df = df[df['removed_by_category']!='moderator']
		except:
			pass
		try:
			df = df[df['banned_by']!='moderators']
		except:
			pass
		end = data['data'][-1]['created_utc']
		
	length = df.shape[0]
		
	
	try:
		spot = np.random.randint(length)
	except ValueError:
		return render_template('index.html', advice='An error in retrieving the Reddit data ocurred, please try again.', prediction_text='')
    
	input_text = df['title'].values[spot]
    
	post_id = df['id'].values[spot]
    
	_, advice, prediction_text = prediction_script.predict_ethics(input_text)
    
	return render_template('index.html', advice='<a href="https://www.reddit.com/%s">%s</a>'%(post_id,advice), prediction_text=prediction_text)

@app.route('/predict_r',methods=['POST'])
def predict_r():
	'''
	For retrieving a recent ULPT post
	'''
	if [x for x in request.form.values()]==[]:
		return render_template('index.html', advice='Please select a subreddit', prediction_text='')
	
	subreddit =  [x for x in request.form.values()][0]
	end = time.time()
	data_list_lpt = []
	df = pd.DataFrame()
	length = 10
	while df.shape[0]<length:
		url = 'https://api.pushshift.io/reddit/search/submission/?size=%s&subreddit=%s&sort=desc&sort_type=created_utc&before=%d'%(length,subreddit,end)
		r = requests.get(url)
		try:
			data = r.json()
		except json.decoder.JSONDecodeError:
			break
		data_list_lpt += data['data']
		df = pd.DataFrame(data_list_lpt)
		df = df[df['title']!='ULPT']
		df = df[~(df['title'].str.contains('request')) & ~(df['title'].str.contains('Request')) & ~(df['title'].str.contains('REQUEST'))]
		try:
			df = df[df['removed_by_category']!='moderator']
		except:
			pass
		try:
			df = df[df['banned_by']!='moderators']
		except:
			pass
		end = data['data'][-1]['created_utc']
		
	length = df.shape[0]
		
	
	try:
		spot = np.random.randint(length)
	except ValueError:
		return render_template('index.html', advice='An error in retrieving the Reddit data ocurred, please try again.', prediction_text='')
    
	input_text = df['title'].values[spot]
    
	post_id = df['id'].values[spot]
    
	_, advice, prediction_text = prediction_script.predict_ethics(input_text)
    
	return render_template('index.html', advice='<a href="https://www.reddit.com/%s">%s<br>(Obtained from r/%s; follow link to see original post)</a>'%(post_id,advice,subreddit), prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)