from flask import Flask, render_template
import joblib

app = Flask(__name__)


@app.route('/')
def index():
    # Load ML model
    model = joblib.load('./notebooks/regr_new.pkl')

    # Make prediction - features = ['BEDS', 'BATHS', 'SQFT', 'AGE', 'LOTSIZE', 'GARAGE']
    prediction = model.predict([[4, 2.5, 3005, 15, 17903.0, 1]])[0][0].round(1)
    prediction = str(prediction)


    return render_template('index.html', pred=prediction)
