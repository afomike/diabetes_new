from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

app = Flask(__name__)

# Load the trained model and encoders once
model = joblib.load('model/degree_class_predictive_model.joblib')
one_hot_encoder = joblib.load('model/one_hot_encoder.joblib')
label_encoders = joblib.load('model/label_encoders.joblib')

# Define preprocessing function
def preprocess_input(data):
    # Convert input data to a DataFrame
    df = pd.DataFrame(data, index=[0])

    # Convert relevant columns to numeric, fill missing values with 0
    df['PREV_GPA'] = pd.to_numeric(df['PREV_GPA'], errors='coerce').fillna(0)
    df['GPA'] = pd.to_numeric(df['GPA'], errors='coerce').fillna(0)

    # Label Encoding for categorical features
    label_encode_features = ['AVG_GRADE_HS']
    for column in label_encode_features:
        if column in df.columns:
            le = label_encoders[column]
            known_classes = set(le.classes_)
            df[column] = df[column].apply(lambda x: le.transform([x])[0] if x in known_classes else -1)  # Using -1 for unknown

    # One-Hot Encoding for categorical features
    one_hot_features = [
        'SCHOOL_TYPE', 'GAP_BEFORE_DEGREE', 
        'MAJOR', 'STUDY_SCHEDULE',
        'PART_TIME_JOB', 'MOTIVATION', 'STRESS_MANAGEMENT'
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
        required_fields = [
            'PREV_GPA', 'GPA', 'SCHOOL_TYPE', 'GAP_BEFORE_DEGREE', 
            'MAJOR', 'STUDY_SCHEDULE', 'PART_TIME_JOB', 
            'MOTIVATION', 'STRESS_MANAGEMENT', 'AVG_GRADE_HS'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return render_template('index.html', prediction=f"Missing required fields: {', '.join(missing_fields)}")

        preprocessed_data = preprocess_input(data)
        prediction = model.predict(preprocessed_data)
        prediction_result = prediction[0]
        return render_template('index.html', prediction=f'Predicted Degree Class: {prediction_result}')
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
