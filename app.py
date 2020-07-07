from flask import Flask, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route('/')
def index():
    # Load ML model
    regression_model = joblib.load('./notebooks/regr_new.pkl')
    tree_model = joblib.load('./notebooks/tree_model.pkl')

    features = ['BEDS', 'BATHS', 'SQFT', 'AGE', 'LOTSIZE', 'GARAGE', 'DOM']
    feature_subs = [4, 2.5, 3005, 15, 17903.0, 1, 40]
    regression_prediction = regression_model.predict([[4, 2.5, 3005, 15, 17903.0, 1, 40]])[0][0].round(1)
    regression_prediction = str(regression_prediction)

    #tree_prediction = tree_model.predict([[4, 2.5, 3005, 15, 17903.0, 1, 40]])[0][0].round(1)
    #tree_prediction = str(tree_prediction)

    return render_template('index.html', features=features, feature_subs=feature_subs, regr_pred=regression_prediction)
