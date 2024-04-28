from flask import Flask, render_template, request
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model = joblib.load("traffic_model.pkl")

# Load the dataset
df = pd.read_csv('./dataset/traffic.csv')

# Define home page route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    hour = int(request.form['hour'])  # Convert to int
    junction = int(request.form['junction'])  # Convert to int
    day = int(request.form['day'])  # Convert to int
    month = int(request.form['month'])  # Convert to int

    # Create a DataFrame with input values
    data = pd.DataFrame({'Junction': [junction],
                         'Month': [month],
                         'Day': [day],
                         'Hour': [hour]})

    # Make prediction
    prediction = model.predict(data)

    # Calculate traffic for all other junctions
    other_junctions_traffic = {}
    for j in range(1, 5):
        if j != junction:
            other_data = pd.DataFrame({'Junction': [j],
                                       'Month': [month],
                                       'Day': [day],
                                       'Hour': [hour]})
            other_prediction = model.predict(other_data)
            other_junctions_traffic[f"Junction {j}"] = round(other_prediction[0])

    # Return prediction and other junctions traffic to user
    return render_template('result.html', prediction=round(prediction[0]), hour=hour, junction=junction, day=day, month=month, other_junctions=other_junctions_traffic)




if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
