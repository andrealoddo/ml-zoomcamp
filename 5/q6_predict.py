from flask import Flask
from flask import request
from flask import jsonify

import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def load(filename):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)

dv = load('dv.bin')
model = load('model2.bin')

app = Flask('CreditCard')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    result = {
        'card_probability': float(y_pred),
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
