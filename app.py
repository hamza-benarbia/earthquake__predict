from flask import Flask, render_template, request
from datetime import datetime
import joblib
import pandas as pd

# Load your pre-trained model using joblib
model = joblib.load('random_forest_model.h5')

app = Flask(__name__, static_url_path='/static', static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    date = request.form['date']
    time = request.form['time']
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])

    # Preprocess the input data
    try:
        ts = datetime.strptime(date + ' ' + time, '%m/%d/%Y %H:%M:%S')
        timestamp = ts.timestamp()
    except ValueError:
        return "Invalid input"

    input_data = pd.DataFrame({'Timestamp': [timestamp], 'Latitude': [latitude], 'Longitude': [longitude]})

    prediction = model.predict(input_data)
    
    magnitude, depth = prediction[0]

    return render_template('index.html', magnitude=round(magnitude, 2), depth=round(depth, 2))

if __name__ == '__main__':
    app.run(debug=True)
