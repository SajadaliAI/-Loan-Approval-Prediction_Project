from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained loan approval model
model = joblib.load('model.pkl')

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        gender = int(request.form['gender'])            # 0 = Female, 1 = Male
        married = int(request.form['married'])          # 0 = No, 1 = Yes
        dependents = int(request.form['dependents'])    # Number of dependents
        education = int(request.form['education'])      # 0 = Graduate, 1 = Not Graduate
        self_employed = int(request.form['self_employed']) # 0 = No, 1 = Yes
        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount = float(request.form['loan_amount'])
        loan_amount_term = float(request.form['loan_amount_term'])
        credit_history = int(request.form['credit_history']) # 0 or 1
        property_area = int(request.form['property_area'])    # 0 = Rural, 1 = Semiurban, 2 = Urban

        # Prepare data for prediction
        data = np.array([[gender, married, dependents, education, self_employed,
                          applicant_income, coapplicant_income, loan_amount,
                          loan_amount_term, credit_history, property_area]])

        # Make prediction
        prediction = model.predict(data)[0]

        if prediction == 1:
            result = "✅ Loan Approved"
        else:
            result = "❌ Loan Not Approved"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
