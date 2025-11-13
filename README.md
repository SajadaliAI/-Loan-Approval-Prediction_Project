# 💰 Loan Approval Prediction using Machine Learning

This project predicts whether a loan application will be **approved or rejected** based on applicant information and financial history.

---

## 🎯 Objective
To build a machine learning model that can **predict loan approval**, helping banks and financial institutions automate the decision-making process and reduce manual effort.

---

## 📊 Dataset Overview
Dataset: `loan_data.csv`  

**Common Columns include:**  
- ApplicantIncome, CoapplicantIncome, LoanAmount  
- Loan_Amount_Term, Credit_History, Gender, Married  
- Dependents, Education, Self_Employed, Property_Area  
- Loan_Status (Target variable: Y/N)  

---

## 🧩 Steps Performed

### 🧹 1. Data Preprocessing
- Handled missing values  
- Encoded categorical features (LabelEncoder / OneHotEncoder)  
- Scaled numerical features (StandardScaler / MinMaxScaler)  

### 📊 2. Exploratory Data Analysis (EDA)
- Checked distributions of applicant income, loan amount, credit history  
- Visualized relationships between features and loan status  
- Analyzed impact of gender, education, property area on loan approval  

### 🤖 3. Model Training
Implemented and compared:
- Logistic Regression  
 

### 📈 4. Model Evaluation
- Accuracy, Precision, Recall, F1-Score  
- Confusion Matrix visualization  
- Selected the best performing model and saved it  

### 💾 5. Model Saving
```python
import joblib
joblib.dump(best_model, 'models/loan_model.pkl')
