from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

app = Flask(__name__)

# Load the trained model and encoders once
model = joblib.load('model/diabetes_predictive_model.joblib')
one_hot_encoder = joblib.load('model/one_hot_encoder.joblib')


# Define preprocessing function
def preprocess_input(data):
    # Convert input data to a DataFrame
    df = pd.DataFrame(data, index=[0])

    # One-Hot Encoding for categorical features
    one_hot_features = [
        'gender',
       'smoking_history'
    ]
    if all(feature in df.columns for feature in one_hot_features):
        # Use the handle_unknown='ignore' parameter to ignore unknown categories
        one_hot_encoded = one_hot_encoder.transform(df[one_hot_features])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_features))

        # Drop the original one-hot encoded columns and concatenate the new ones
        df = df.drop(columns=one_hot_features)
        
        # Ensure the concatenation aligns properly
        df = pd.concat([df.reset_index(drop=True), one_hot_encoded_df.reset_index(drop=True)], axis=1)

    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        # Check for required fields
        required_fields = ['gender', 'age', 'hypertension', 'heart_disease', 
        'smoking_history','bmi', 'HbA1c_level', 
        'blood_glucose_level']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return render_template('index.html', prediction=f"Missing required fields: {', '.join(missing_fields)}")

        preprocessed_data = preprocess_input(data)
        prediction = model.predict(preprocessed_data)
        prediction_result = prediction[0]
        if prediction_result == 0:
            prediction_result = "Negative"
        else:
            prediction_result = "Positive"

        return render_template('index.html', prediction=f'Patient is likely to be diabetes: {prediction_result}')
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
