# 💳 CreditWise — Loan Approval Intelligence

<div align="center">

![CreditWise Banner](https://img.shields.io/badge/CreditWise-Loan%20Intelligence-00e5ff?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0yMCA0SDRjLTEuMSAwLTIgLjktMiAydjEyYzAgMS4xLjkgMiAyIDJoMTZjMS4xIDAgMi0uOSAyLTJWNmMwLTEuMS0uOS0yLTItMnptMCAxNEg0di02aDEwdi0ySDR2LTJoMTZ2MTB6Ii8+PC9zdmc+)

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-creditwise--loan.streamlit.app-00e5ff?style=for-the-badge)](https://creditwise-loan.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Core-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**An end-to-end Machine Learning pipeline for credit risk assessment, powered by Gaussian Naive Bayes, custom class balancing, and an interactive real-time Streamlit dashboard.**

</div>

-----

## 📌 Overview

CreditWise is a complete ML-powered loan approval system built from scratch. It takes applicant financial data, processes it through a robust pipeline, and returns a real-time approval decision with model confidence, probability breakdown, and key factor analysis.

Built as a **1st year CSE student** — self-learned, no tutorials copy-pasted.

-----

## 🚀 Live Demo

👉 **[creditwise-loan.streamlit.app](https://creditwise-loan.streamlit.app/)**

-----

## ✨ Features

|Feature                   |Description                                      |
|--------------------------|-------------------------------------------------|
|💳 **Real-time Prediction**|Instant loan approval/rejection with confidence %|
|📊 **Model Comparison**    |Naive Bayes vs Logistic Regression vs KNN        |
|⚖️ **Class Balancing**     |Custom oversampling for 70/30 imbalanced dataset |
|🔍 **Feature Correlations**|Visual analysis of top predictors                |
|🎯 **Confusion Matrix**    |Full evaluation with TP/TN/FP/FN breakdown       |
|📈 **Probability Charts**  |Plotly gauge + bar charts for confidence         |
|🌙 **Dark UI**             |Custom CSS dark theme with cyan accents          |

-----

## 🧠 ML Pipeline

```
Raw Data (1,000 samples, 19 features)
        ↓
Null Handling (SimpleImputer)
        ↓
Label Encoding (Education, Target)
        ↓
One-Hot Encoding (6 categorical cols)
        ↓
Feature Engineering
  → DTI_Ratio² · Credit_Score² · log(Income)
        ↓
StandardScaler
        ↓
Oversampling (minority class balancing)
        ↓
Model Training (GaussianNB)
        ↓
Real-time Prediction + Confidence Score
```

-----

## 📊 Model Performance

|Model          |Accuracy      |Precision     |F1 Score      |Recall        |
|---------------|--------------|--------------|--------------|--------------|
|**Naive Bayes**|displayed live|displayed live|displayed live|displayed live|
|Logistic Reg.  |displayed live|displayed live|displayed live|displayed live|
|KNN            |displayed live|displayed live|displayed live|displayed live|


> *Metrics computed live on app — toggle oversampling on/off to see the impact*

-----

## 🔍 Top Predictors

```
Credit Score    ████████████████████  +0.451  (strongest positive)
DTI Ratio       ████████████████████  -0.445  (strongest negative)
Applicant Income ████████░░░░░░░░░░░  +0.120
Employer MNC    ████░░░░░░░░░░░░░░░░  +0.069
Loan Amount     ████████░░░░░░░░░░░░  -0.126
```

-----

## 🛠️ Tech Stack

```python
language    = "Python 3.11+"
ml          = ["scikit-learn", "pandas", "numpy"]
deployment  = ["Streamlit", "Plotly"]
styling     = "Custom CSS (dark theme)"
balancing   = "Custom oversampling (no imblearn)"
```

-----

## 📁 Repository Structure

```
creditwise-loan-system/
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
├── runtime.txt             # Python 3.11 for Streamlit Cloud
└── README.md               # You are here
```

-----

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/mehranmushtaq/creditwise-loan-system.git
cd creditwise-loan-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

-----

## 📦 Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.4.0
plotly>=5.18.0
```

-----

## 🗂️ Input Features

**Financial:** Applicant Income, Co-applicant Income, Loan Amount, Loan Term, Savings, Collateral Value

**Credit & Risk:** Credit Score, DTI Ratio, Existing Loans, Dependents

**Demographics:** Age, Gender, Marital Status, Education Level

**Employment:** Employment Status, Employer Category, Property Area, Loan Purpose

-----

## 👨‍💻 Author

**Mehran Mushtaq**

- 🎓 1st Year CSE Student
- 🔥 Self-learned ML & Deep Learning
- 📍 Kashmir, India
- 🐙 [GitHub](https://github.com/mehranmushtaq)
- 💻 [LeetCode](https://leetcode.com/u/mehraan1/)

-----

## 🔗 Related Projects

- [Deep Learning — ANN/CNN with PyTorch](https://github.com/mehranmushtaq/deep-learning)
- [Machine Learning from Scratch & Sklearn](https://github.com/mehranmushtaq/Machine-Learning-with-scikit-learn-and-from-scratch)
- [Exploratory Data Analysis](https://github.com/mehranmushtaq/exploratory-data-analysis)

-----

⭐ **Star this repo if you find it helpful!**
