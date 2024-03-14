from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model from the .pkl file
with open('RG.pkl', 'rb') as f:
    model = pickle.load(f)

def process_data(data, request):
    # Process the data into a list
    data_list = [request.args.get(item) for item in data]
    return data_list

@app.route('/predict', methods=['GET'])
def predict():
    # Data to fetch from the URL parameters
    data_to_fetch = [
        "Owned_Realty",
        "Total_Children",
        "Total_Income",
        "Education_Type",
        "Family_Status",
        "Housing_Type",
        "Applicant_Age",
        "Years_of_Working",
        "Total_Bad_Debt",
        "Total_Good_Debt",
    ]
    
    # Create a DataFrame from the fetched data
    newdata = pd.DataFrame([process_data(data_to_fetch, request)], columns=data_to_fetch)

    # Make predictions using the model
    prediction = model.predict(newdata)

    # Get prediction result
    if prediction == 1:
        result = "Maaf Pengajuan Anda di Approved"
    else:
        result = "Selamat Pengajuan Anda di Rejected"
    
    return hasil(result)

@app.route('/')
def index():
    return render_template('form.html')

def hasil(result):
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
