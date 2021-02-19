import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from textstat import flesch_kincaid_grade
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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
	
    advice = prediction_script.predict_ethics(input_text)[1]
    
    prediction_text = prediction_script.predict_ethics(input_text)[2]

    return render_template('index.html', advice=advice, prediction_text=prediction_text)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)