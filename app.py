from flask import Flask, request, jsonify, render_template
from keras.models import Sequential, load_model
from nltk.tokenize import word_tokenize
from Intent_classification_final import uni, padding_doc
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
model = load_model("Newmodel12.h5")
i2 = uni()[1]
i3 = uni()[2]
ti = uni()[0]
lem=WordNetLemmatizer()
results = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text=[x for x in request.form.values()][0]
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
    test_word = word_tokenize(clean)
    test_word = [lem.lemmatize(w.lower()) for w in test_word]
    test_ls = i2.texts_to_sequences(test_word)

    # Check for unknown words
    if [] in test_ls:
        test_ls = list(filter(None, test_ls))
    test_ls = np.array(test_ls).reshape(1, len(test_ls))
    x = padding_doc(test_ls, i3)
    pred = model.predict_proba(x)

    predictions = pred[0]
    classes = np.array(ti)
    ids = np.argsort(-predictions)
    classes = classes[ids]
    predictions = -np.sort(-predictions)

    tyu = pd.DataFrame(predictions, index=classes)

    doc = nlp(text)
    ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]

    output1 = tyu[0].idxmax()
    output2=ents
    results = {"Intent":output1,"Entity":output2}

    return render_template('index.html', prediction_text='{}'.format(results))


if __name__ == "__main__":
    app.run(debug=True)