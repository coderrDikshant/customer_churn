from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model, encoders, and scaler
model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("encoders.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Define feature columns for consistent ordering
feature_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from the form
    data = {col: request.form[col] for col in feature_columns}

    # Convert input data to DataFrame for consistency
    input_data = pd.DataFrame([data])

    # Encode categorical variables with error handling for unseen labels
    for col, le in label_encoders.items():
        if col in input_data.columns:
            # Handle unseen labels by assigning them a default value (e.g., the first class)
            input_data[col] = input_data[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            input_data[col] = le.transform(input_data[col])

    # Scale numerical features
    input_data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        input_data[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Return result
    return render_template('index.html', prediction_text=f'Churn Prediction: {"Yes" if prediction == 1 else "No"}')

if __name__ == "__main__":
    app.run(debug=True)
