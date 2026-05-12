<div align="center">

<img src="https://img.shields.io/badge/CreditWise-Loan%20Intelligence-00e5ff?style=for-the-badge&logo=databricks&logoColor=white" alt="CreditWise"/>

# 💳 CreditWise — Loan Approval Intelligence System

**An end-to-end Machine Learning pipeline for credit risk assessment, powered by Gaussian Naive Bayes, SMOTE class balancing, and an interactive real-time Streamlit dashboard.**

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-creditwise--loan.streamlit.app-00e5ff?style=flat-square)](https://creditwise-loan.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Core-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-10b981?style=flat-square)]()

<br/>


> *“Transforming complex financial data into instant, interpretable loan decisions — no black boxes.”*

</div>

-----

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [ML Pipeline Deep Dive](#-ml-pipeline-deep-dive)
- [Model Performance](#-model-performance)
- [Feature Importance](#-feature-importance)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How It Works](#-how-it-works)
- [Data & Class Imbalance](#-data--class-imbalance)
- [Roadmap](#-roadmap)
- [Internship & Academic Context](#-internship--academic-context)
- [Contributing](#-contributing)
- [Author](#-author)
- [License](#-license)

-----

## 🎯 Overview

**CreditWise** is a production-deployed machine learning application that automates loan eligibility assessment using probabilistic classification. It is built on a **real-world structured dataset** (`loan_approval_data.csv`) of 1,000 loan applications, taking it through the complete ML lifecycle — EDA, missing value handling, encoding, feature engineering, multi-model benchmarking, and real-time interactive prediction — all deployed as a polished web app.

The project demonstrates a **rigorous, structured ML workflow**: Exploratory Data Analysis with Seaborn and Matplotlib, systematic imputation of missing values, model comparison across three classifiers, and selection of the best model based on a domain-appropriate metric — **Precision** — since in lending, false approvals are more costly than false rejections.

**Problem Statement:** Traditional loan approval processes are slow, opaque, and prone to human bias. CreditWise replaces subjective judgment with a transparent, data-driven model that provides instant decisions with confidence scores and interpretable feature breakdowns.

-----

## Live Demo

👉 **[creditwise-loan.streamlit.app](https://creditwise-loan.streamlit.app/)**

The live app allows you to:

- Input applicant financial and demographic details through an interactive form
- Receive an instant loan approval or rejection decision
- View model confidence via a real-time gauge chart
- Explore probability distributions, key risk factors, and confusion matrices

-----

## Key Features

|Feature                        |Description                                                    |
|-------------------------------|---------------------------------------------------------------|
|🧠 **Real-Time Prediction**     |Instant loan approval/rejection using Gaussian Naive Bayes     |
|⚖️ **SMOTE Integration**        |Toggle class balancing on/off to handle 70.2% / 29.8% imbalance|
|📊 **Multi-Model Benchmarking** |Compares Naive Bayes, Logistic Regression, and KNN side-by-side|
|🎯 **Confidence Scoring**       |Probabilistic output with approval confidence percentage       |
|📈 **Feature Correlation Chart**|Reveals each feature’s predictive weight toward loan approval  |
|🔲 **Confusion Matrix**         |TP / TN / FP / FN breakdown for the primary classifier         |
|🎨 **Dark UI / Design System**  |Syne + DM Mono typography, CSS grid background, cyan accent    |
|☁️ **Cloud Deployed**           |Live on Streamlit Community Cloud — zero-install access        |

-----

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CreditWise Pipeline                        │
├──────────────────┬───────────────────────┬──────────────────────┤
│   Data Layer     │   Processing Layer    │   Inference Layer    │
│                  │                       │                      │
│ loan_approval_   │ → Missing value imp.  │ → GaussianNB.predict │
│ data.csv         │   (mean / mode)       │ → predict_proba()    │
│ 1000 rows        │ → Label encoding      │ → Confidence gauge   │
│ 20 columns       │ → One-Hot encoding    │ → Binary decision    │
│ ~50 missing vals │ → Feature engineering │   (Approved/Rejected)│
│ per column       │ → StandardScaler      │                      │
│                  │ → SMOTE (optional)    │                      │
└──────────────────┴───────────────────────┴──────────────────────┘
                               │
               ┌───────────────▼───────────────┐
               │      Streamlit Frontend       │
               │  Form → Predict → Visualize   │
               └───────────────────────────────┘
```

-----

## 🔬 ML Pipeline Deep Dive

### 1. Dataset

The project uses a **real CSV dataset** (`loan_approval_data.csv`) — not synthetically generated data. It contains **1,000 loan applications**, **20 columns**, and approximately **50 missing values per column** (950 non-null per feature).

**Features:**

|Column            |Type   |Description                                     |
|------------------|-------|------------------------------------------------|
|Applicant_ID      |float64|Unique identifier (dropped before modelling)    |
|Applicant_Income  |float64|Primary applicant monthly income                |
|Coapplicant_Income|float64|Co-applicant monthly income                     |
|Employment_Status |object |Salaried / Self-employed / Unemployed / Contract|
|Age               |float64|Applicant age                                   |
|Marital_Status    |object |Single / Married / Divorced                     |
|Dependants        |float64|Number of dependants                            |
|Credit_Score      |float64|Credit score (550–800 range)                    |
|Existing_Loans    |float64|Number of active loans                          |
|DTI_Ratio         |float64|Debt-to-income ratio                            |
|Savings           |float64|Total savings                                   |
|Collateral_Value  |float64|Value of collateral offered                     |
|Loan_Amount       |float64|Requested loan amount                           |
|Loan_Term         |float64|Loan repayment term (months)                    |
|Loan_Purpose      |object |Home / Car / Personal / Education / Business    |
|Property_Area     |object |Urban / Semiurban / Rural                       |
|Education_Level   |object |Graduate / Not Graduate                         |
|Gender            |object |Male / Female                                   |
|Employer_Category |object |Private / Government / MNC / Unemployed         |
|Loan_Approved     |object |**Target** — Yes / No                           |

### 2. Exploratory Data Analysis (EDA)

A thorough EDA was conducted before any modelling:

- **Class balance:** Pie chart confirming 70.2% rejected / 29.8% approved — flagging the need for SMOTE
- **Categorical distributions:** Bar charts — 621 Male / 379 Female; Graduate majority
- **Income histograms:** Applicant_Income and Coapplicant_Income show roughly uniform distributions
- **Box plots:** Credit_Score and DTI_Ratio show strong visual separation between approved and rejected applicants; Savings, Age, Loan_Amount show minimal separation
- **Credit Score histplot by class:** Approved applicants clearly skew toward 700–800; rejected skew toward 550–700
- **Full 28×28 Correlation Heatmap:** Computed across all numerically encoded features to identify multicollinearity and feature-target relationships

### 3. Preprocessing

```
Raw CSV (with ~50 missing values per column)
   │
   ├─ Numerical → SimpleImputer(strategy="mean")
   ├─ Categorical → SimpleImputer(strategy="most_frequent")
   ├─ Applicant_ID → dropped (df.drop("Applicant_ID", axis=1))
   ├─ Education_Level + Loan_Approved → LabelEncoder
   └─ Employment_Status, Marital_Status, Loan_Purpose,
      Property_Area, Gender, Employer_Category
         → OneHotEncoder(drop="first", sparse_output=False)
```

### 4. Model Benchmarking (Pre-Engineering)

Three classifiers were trained and evaluated on the same 67/33 train-test split **before** feature engineering to identify the strongest base model:

- Logistic Regression
- KNN (n_neighbors=5)
- Gaussian Naive Bayes

**Naive Bayes was selected as the best model on the basis of Precision.**

### 5. Feature Engineering

After identifying Naive Bayes as the best base model, feature engineering was applied to further improve performance:

```python
df["DTI_Ratio_sq"]         = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"]      = df["Credit_Score"] ** 2
df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

# Drop originals — replaced by engineered versions
X = df.drop(columns=["DTI_Ratio", "Credit_Score", "Loan_Approved", "Applicant_Income"])
```

- `DTI_Ratio_sq` — amplifies the penalty for high debt ratios non-linearly
- `Credit_Score_sq` — rewards excellent credit scores disproportionately
- `Applicant_Income_log` — compresses right skew in income distribution

### 6. Train-Test Split & Scaling

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

-----

## 📊 Model Performance

### Before Feature Engineering

|Model              |Precision|Accuracy |F1 Score |Recall   |
|-------------------|---------|---------|---------|---------|
|Logistic Regression|84.6%    |86.7%    |77.8%    |71.96%   |
|KNN (k=5)          |72.7%    |76.7%    |55.5%    |44.9%    |
|**Naive Bayes** ⭐  |**85.7%**|**85.8%**|**75.4%**|**67.3%**|


> ✅ **Best model on Precision → Naive Bayes selected**

### After Feature Engineering (Final)

|Model              |Precision|Accuracy |F1 Score |Recall   |
|-------------------|---------|---------|---------|---------|
|**Naive Bayes** ⭐  |**88.5%**|**87.9%**|**79.4%**|**71.9%**|
|Logistic Regression|85.6%    |88.5%    |81.4%    |77.8%    |

Feature engineering improved Naive Bayes Precision by **+2.8 percentage points**.

**Why Precision over Accuracy?**
In credit risk, a false positive (approving a loan to a defaulter) is far more damaging than a false negative (rejecting a qualified applicant). Precision directly measures how often the model’s approvals are genuinely correct — making it the right optimisation target for this domain.

-----

## 🔍 Feature Importance

From the actual correlation analysis in the notebook (`num_cols.corr()["Loan_Approved"].sort_values(ascending=False)`):

|Feature                     |Correlation|
|----------------------------|-----------|
|Credit_Score                |**+0.451** |
|Applicant_Income            |+0.120     |
|Employer_Category_MNC       |+0.120     |
|Loan_Purpose_Personal       |+0.034     |
|Marital_Status_Single       |+0.030     |
|Property_Area_Urban         |+0.026     |
|Collateral_Value            |+0.021     |
|…                           |…          |
|Loan_Amount                 |−0.086     |
|Loan_Term                   |−0.054     |
|Education_Level             |−0.044     |
|Existing_Loans              |−0.035     |
|Employer_Category_Government|−0.039     |
|Employment_Status_Unemployed|−0.041     |
|Gender_Male                 |−0.054     |
|DTI_Ratio                   |**−0.445** |

**Key insight:** Credit Score (+0.451) and DTI Ratio (−0.445) are by far the dominant predictors — mirroring real-world lending practice.

-----

## 🛠️ Tech Stack

```
Language        Python 3.11+
ML Core         scikit-learn (GaussianNB, LogisticRegression,
                KNeighborsClassifier, StandardScaler,
                LabelEncoder, OneHotEncoder, SimpleImputer,
                train_test_split, accuracy_score, precision_score,
                f1_score, recall_score, confusion_matrix)
Resampling      imbalanced-learn (SMOTE)
Data            pandas, numpy
EDA             seaborn, matplotlib
Visualisation   Plotly (Streamlit app — gauge, bar, heatmap)
Frontend        Streamlit + custom CSS3
Typography      Google Fonts — Syne, DM Mono, DM Sans
Deployment      Streamlit Community Cloud
```

-----

## 📁 Project Structure

```
creditwise-loan-system/
│
├── app.py                   # Streamlit web application (full pipeline + UI)
├── loan_approval_data.csv   # Real dataset — 1000 rows, 20 columns
├── loan_approval.ipynb      # Jupyter Notebook — EDA, modelling, analysis
├── requirements.txt         # Pinned dependencies
└── README.md                # This file
```

-----

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/mehranmushtaq/creditwise-loan-system.git
cd creditwise-loan-system
```

**2. (Optional) Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the app**

```bash
streamlit run app.py
```

**5. Open in browser**

```
http://localhost:8501
```

-----

## 🧭 How It Works

```
User fills form (18 applicant fields)
         │
         ▼
predict() function called
         │
         ├─ LabelEncoder → Education_Level
         ├─ OneHotEncoder → 6 categorical features
         ├─ Compute: DTI_sq, Credit_sq, Income_log
         ├─ Drop: DTI_Ratio, Credit_Score, Applicant_Income
         ├─ Align columns to training feature set
         └─ StandardScaler.transform()
                   │
                   ▼
         GaussianNB.predict()       →  0 (Rejected) / 1 (Approved)
         GaussianNB.predict_proba() →  [P(Rejected), P(Approved)]
                   │
                   ▼
         Render: decision card + confidence gauge + probability bars + key factors
```

-----

## 📉 Data & Class Imbalance

The dataset reflects a realistic loan approval rate:

```
Rejected  ████████████████████████████  70.2%
Approved  ████████████                  29.8%
```

A model trained on this imbalanced data without correction would learn to over-predict rejections, achieving deceptively high accuracy by simply ignoring approvals. **SMOTE** corrects this by synthetically generating new minority-class examples using k-nearest-neighbour interpolation — balancing the training distribution without duplicating existing records.

-----

## 🗺️ Roadmap

- [ ] Add Random Forest and XGBoost to model comparison
- [ ] Integrate SHAP explainability for per-prediction feature attribution
- [ ] Add ROC-AUC and Precision-Recall curve visualisations
- [ ] Export prediction as a downloadable PDF report
- [ ] Wrap model in a FastAPI REST endpoint for programmatic access
- [ ] Add unit tests and CI/CD via GitHub Actions

-----

## 🎓 Internship & Academic Context

This project was developed as part of a **B.Tech Computer Science Engineering** programme and serves as a complete, deployable portfolio piece demonstrating the following industry-relevant competencies:

### Skills Demonstrated

**Machine Learning & Data Science**

- End-to-end ML pipeline from raw CSV to a live deployed application
- EDA using Seaborn (bar plots, histograms, box plots, heatmaps) and Matplotlib
- Systematic missing value imputation (mean for numerical, mode for categorical)
- Multi-model benchmarking and selection using domain-appropriate metric (Precision)
- Feature engineering with polynomial and logarithmic transformations
- Class imbalance handling with SMOTE
- Full evaluation suite: Accuracy, Precision, F1, Recall, Confusion Matrix

**Software Engineering**

- Clean, modular Python code with Streamlit caching for performance
- Robust pipeline with consistent train/inference column alignment
- Single-file Streamlit architecture optimised for cloud deployment

**Product & UI**

- Custom Streamlit theming with CSS3 injection
- Real-time interactive charting with Plotly
- Responsive two-column layout (input form + results panel)

**Deployment**

- Streamlit Community Cloud deployment
- Dependency management via `requirements.txt`

### Relevant for Internship Roles at

|Role                            |Relevance                                                                 |
|--------------------------------|--------------------------------------------------------------------------|
|Machine Learning Engineer Intern|Full ML pipeline, model selection, evaluation metrics, feature engineering|
|Data Science Intern             |EDA with Seaborn/Matplotlib, imputation, correlation analysis, SMOTE      |
|AI/ML Product Intern            |End-to-end product from raw CSV to deployed interactive app               |
|Software Engineering Intern     |Python, clean architecture, cloud deployment                              |
|FinTech / Risk Analytics Intern |Credit scoring domain, DTI analysis, precision-optimised lending model    |

### Academic Alignment

This project maps directly to coursework in **Pattern Recognition**, **Data Mining**, **Probability & Statistics**, **Software Engineering**, and **Database Systems**. The structured Jupyter notebook (`loan_approval.ipynb`) demonstrates the ability to communicate analytical findings clearly through well-documented, reproducible, step-by-step code — a key skill in both research and industry settings.

-----

## 🤝 Contributing

Contributions are welcome and encouraged!

1. Fork the repository
1. Create a feature branch: `git checkout -b feature/your-feature-name`
1. Commit your changes: `git commit -m "feat: add SHAP explainability"`
1. Push to the branch: `git push origin feature/your-feature-name`
1. Open a Pull Request

Please follow conventional commit style and keep PRs focused on a single improvement.

-----

## 👤 Author

**Mehran Mushtaq**
B.Tech Computer Science Engineering Student

[![GitHub](https://img.shields.io/badge/GitHub-mehranmushtaq-181717?style=flat-square&logo=github)](https://github.com/mehranmushtaq)

-----

## 📄 License

This project is licensed under the **MIT License** — see the <LICENSE> file for details.

-----

<div align="center">

Made with 🧠 and Python · Deployed on Streamlit Cloud

**[⭐ Star this repo](https://github.com/mehranmushtaq/creditwise-loan-system)** if you found it useful!

</div>
