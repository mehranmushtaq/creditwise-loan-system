# creditwise-loan-system
# 💳 CreditWise Loan Intelligence System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://creditwise-loan.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning solution designed to automate credit risk assessment. This system uses a Gaussian Naive Bayes architecture to predict loan eligibility with a high degree of confidence, featuring a real-time interactive dashboard for financial diagnostics.

[**Explore the Live Demo »**](https://creditwise-loan.streamlit.app/)

---

###  Key Features
* **Predictive Intelligence:** Real-time loan approval classification using optimized Naive Bayes.
* **Interactive Dashboard:** Built with **Streamlit** for a seamless user experience.
* **Financial Diagnostics:** Includes confidence scoring and class imbalance handling (SMOTE integration).
* **Data Visualization:** Dynamic charts powered by **Plotly** to visualize credit trends.

###  Tech Stack
* **Language:** Python
* **ML Libraries:** Scikit-learn, Pandas, NumPy
* **Frontend/Deployment:** Streamlit, CSS3
* **Visuals:** Plotly, Matplotlib

###  Installation & Setup
1. **Clone the Repo**
   ```bash
   git clone [https://github.com/mehranmushtaq/creditwise-loan-system.git](https://github.com/mehranmushtaq/creditwise-loan-system.git)
   cd creditwise-loan-system
2. **Set up Environment**
   ```
   pip install -r requirements.txt
   ```
3. **Run Locally**
   ```
   streamlit run app.py
   ```
 
### Model Architecture

The system processes raw financial data through a robust pipeline:

1.Preprocessing: Handling missing values and categorical encoding.

2.Scaling: Robust scaling for financial numerical features.

3.Classification: GaussianNB implementation for probabilistic approval modeling.

4.Evaluation: Monitored via Precision-Recall curves and Confusion Matrices.

### Repository Structure
```
├── app.py              # Streamlit Web Interface
├── requirements.txt    # Project Dependencies
└── README.md
```

### Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Developed by **Mehran Mushtaq**
```
https://github.com/mehranmushtaq
```
B.Tech Computer Science Engineering Student

