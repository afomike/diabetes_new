from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and encoders
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

    # # Example additional preprocessing steps
    # # Convert study hours to numeric
    # df['HOURS_STUDY'] = pd.to_numeric(df['HOURS_STUDY'], errors='coerce').fillna(0)
    
    # df['EXTRA_ACTIVITIES_HOURS'] = pd.to_numeric(df['EXTRA_ACTIVITIES_HOURS'], errors='coerce').fillna(0)

    
        # Label Encoding for categorical features
    label_encode_features = ['AVG_GRADE_HS']
    for column in label_encode_features:
        df[column] = label_encoders[column].transform(df[column])

    # One-Hot Encoding for categorical features
    one_hot_features = [
        'SCHOOL_TYPE', 'GAP_BEFORE_DEGREE', 
        'MAJOR', 'STUDY_SCHEDULE',
        'PART_TIME_JOB', 'MOTIVATION', 'STRESS_MANAGEMENT'
    ]
    one_hot_encoded = one_hot_encoder.transform(df[one_hot_features])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_features))

    # Drop the original one-hot encoded columns and concatenate the new ones
    df = df.drop(columns=one_hot_features)
    df = pd.concat([df, one_hot_encoded_df], axis=1)


    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    preprocessed_data = preprocess_input(data)
    prediction = model.predict(preprocessed_data)
    prediction_result ={prediction[0]}
    return render_template('index.html', prediction=prediction_result)
    # return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
