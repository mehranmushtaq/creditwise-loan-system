import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score,
    recall_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="CreditWise · Loan Approval",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS (dark financial aesthetic)
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'Syne', sans-serif; letter-spacing: -0.5px; }

/* ── Grid background ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,229,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,255,0.025) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Cards ── */
.cw-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
}

/* ── Imbalance banner ── */
.imbalance-banner {
    background: linear-gradient(135deg, rgba(245,158,11,0.08), rgba(245,158,11,0.02));
    border: 1px solid rgba(245,158,11,0.35);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 24px;
}

/* ── Result approved ── */
.result-approved {
    background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(16,185,129,0.03));
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}

/* ── Result rejected ── */
.result-rejected {
    background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.03));
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}

/* ── Metric pill ── */
.metric-pill {
    background: #1a2235;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
}

/* ── Section label ── */
.section-tag {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #00e5ff;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.8;
    margin-bottom: 10px;
}

/* ── Inputs ── */
.stNumberInput input, .stSelectbox select, .stSlider {
    background: #1a2235 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00e5ff, #0099cc) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 14px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,229,255,0.25) !important;
}

/* ── Logo header ── */
.cw-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 20px;
    border-bottom: 1px solid #1e2d45;
    margin-bottom: 32px;
}
.cw-logo {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.cw-logo span { color: #00e5ff; }
.cw-badge {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #00e5ff;
    border: 1px solid #00e5ff;
    padding: 4px 10px;
    border-radius: 4px;
    opacity: 0.75;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="cw-header">
  <div class="cw-logo">Credit<span>Wise</span></div>
  <div class="cw-badge">Naive Bayes · v1.0</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='font-family:Syne,sans-serif;font-size:2.6rem;font-weight:800;
           letter-spacing:-1.5px;margin-bottom:6px;'>
  Loan Approval <span style='background:linear-gradient(90deg,#00e5ff,#7c3aed);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>Intelligence</span>
</h1>
<p style='color:#64748b;font-size:14px;margin-bottom:28px;'>
  ML-powered loan assessment · 1,000 applications · Naive Bayes + Feature Engineering
</p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# IMBALANCE BANNER
# ─────────────────────────────────────────────

st.markdown("""
<div class="imbalance-banner">
  <div style="font-size:18px;margin-bottom:6px;">⚠️ <strong style="color:#f59e0b;
       font-family:Syne,sans-serif;font-size:13px;letter-spacing:1px;
       text-transform:uppercase;">Class Imbalance Notice</strong></div>
  <p style="font-size:13px;color:#94a3b8;line-height:1.6;margin-bottom:10px;">
    Training data is <strong>imbalanced</strong>: 70.2% rejections vs 29.8% approvals.
    The raw model may lean toward predicting rejection.
    Enable <strong>SMOTE</strong> below to oversample the minority class for fairer predictions.
  </p>
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
    <span style="font-family:DM Mono,monospace;font-size:11px;color:#64748b;width:28px;">No</span>
    <div style="flex:1;height:8px;background:#1a2235;border-radius:4px;overflow:hidden;">
      <div style="width:70.2%;height:100%;background:#ef4444;border-radius:4px;"></div>
    </div>
    <span style="font-family:DM Mono,monospace;font-size:11px;color:#ef4444;">70.2%</span>
  </div>
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-family:DM Mono,monospace;font-size:11px;color:#64748b;width:28px;">Yes</span>
    <div style="flex:1;height:8px;background:#1a2235;border-radius:4px;overflow:hidden;">
      <div style="width:29.8%;height:100%;background:#10b981;border-radius:4px;"></div>
    </div>
    <span style="font-family:DM Mono,monospace;font-size:11px;color:#10b981;">29.8%</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SMOTE TOGGLE
# ─────────────────────────────────────────────

use_smote = st.toggle("⚖️ Apply SMOTE (balance classes before training)", value=True)

# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────

@st.cache_data
def generate_dataset(seed=42):
    np.random.seed(seed)
    n = 1000
    df = pd.DataFrame()
    df["Applicant_ID"]        = np.arange(n)
    df["Applicant_Income"]    = np.random.uniform(1000, 20000, n)
    df["Coapplicant_Income"]  = np.random.choice([0, np.random.uniform(500, 10000)], n)
    df["Employment_Status"]   = np.random.choice(["Salaried", "Self-employed", "Unemployed"], n, p=[0.55, 0.30, 0.15])
    df["Age"]                 = np.random.randint(22, 62, n)
    df["Marital_Status"]      = np.random.choice(["Single", "Married", "Divorced"], n, p=[0.4, 0.45, 0.15])
    df["Dependents"]          = np.random.randint(0, 6, n)
    df["Credit_Score"]        = np.random.uniform(550, 800, n)
    df["Existing_Loans"]      = np.random.randint(0, 6, n)
    df["DTI_Ratio"]           = np.random.uniform(0.05, 0.6, n)
    df["Savings"]             = np.random.uniform(0, 20000, n)
    df["Collateral_Value"]    = np.random.uniform(0, 40000, n)
    df["Loan_Amount"]         = np.random.uniform(5000, 50000, n)
    df["Loan_Term"]           = np.random.choice([12, 24, 36, 48, 60], n)
    df["Loan_Purpose"]        = np.random.choice(["Home", "Car", "Personal", "Education"], n)
    df["Education_Level"]     = np.random.choice(["Graduate", "Not Graduate"], n, p=[0.72, 0.28])
    df["Gender"]              = np.random.choice(["Male", "Female"], n, p=[0.62, 0.38])
    df["Property_Area"]       = np.random.choice(["Urban", "Semiurban", "Rural"], n)
    df["Employer_Category"]   = np.random.choice(["Private", "Government", "MNC", "Unemployed"], n, p=[0.4, 0.25, 0.2, 0.15])

    score = (
         0.45 * (df["Credit_Score"] - 550) / 250
        -0.44 * df["DTI_Ratio"]
        +0.12 * df["Applicant_Income"] / 20000
        -0.13 * df["Loan_Amount"] / 50000
        +0.07 * (df["Employer_Category"] == "MNC").astype(int)
        +0.03 * (df["Loan_Purpose"] == "Personal").astype(int)
        + np.random.normal(0, 0.12, n)
    )
    threshold = np.percentile(score, 70)
    df["Loan_Approved"] = np.where(score > threshold, "Yes", "No")
    return df

df_raw = generate_dataset()

# ─────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

@st.cache_resource
def build_pipeline(use_smote_flag):
    df = df_raw.copy()
    df = df.drop("Applicant_ID", axis=1)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c != "Loan_Approved"]

    num_imp = SimpleImputer(strategy="mean")
    df[num_cols] = num_imp.fit_transform(df[num_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    le_edu = LabelEncoder()
    le_tgt = LabelEncoder()
    df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])
    df["Loan_Approved"]   = le_tgt.fit_transform(df["Loan_Approved"])

    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False)
    encoded = ohe.fit_transform(df[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols))
    df = pd.concat([df.drop(columns=ohe_cols).reset_index(drop=True), encoded_df], axis=1)

    df["DTI_Ratio_sq"]        = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"]     = df["Credit_Score"] ** 2
    df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

    X = df.drop(["Loan_Approved", "DTI_Ratio", "Credit_Score"], axis=1)
    y = df["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    if use_smote_flag:
        sm = SMOTE(random_state=42)
        X_train_sc, y_train = sm.fit_resample(X_train_sc, y_train)

    nb = GaussianNB(); lr = LogisticRegression(max_iter=1000); knn = KNeighborsClassifier(n_neighbors=5)
    nb.fit(X_train_sc, y_train); lr.fit(X_train_sc, y_train); knn.fit(X_train_sc, y_train)

    def metrics(model):
        y_pred = model.predict(X_test_sc)
        return {
            "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 1),
            "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 1),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0) * 100, 1),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 1),
            "cm":        confusion_matrix(y_test, y_pred).tolist(),
        }

    return {
        "nb": nb, "lr": lr, "knn": knn, "scaler": scaler, "ohe": ohe, "ohe_cols": ohe_cols,
        "le_edu": le_edu, "le_tgt": le_tgt, "feature_cols": list(X.columns),
        "metrics": {"Naive Bayes": metrics(nb), "Logistic Reg.": metrics(lr), "KNN": metrics(knn)}
    }

with st.spinner("Training models..."):
    pipeline = build_pipeline(use_smote)

nb_model = pipeline["nb"]; scaler = pipeline["scaler"]; ohe = pipeline["ohe"]; ohe_cols = pipeline["ohe_cols"]
le_edu = pipeline["le_edu"]; feat_cols = pipeline["feature_cols"]; model_metrics = pipeline["metrics"]

# ─────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict(inputs: dict):
    row = pd.DataFrame([inputs])
    row["Education_Level"] = le_edu.transform(row["Education_Level"])
    encoded = ohe.transform(row[ohe_cols])
    enc_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols))
    row = pd.concat([row.drop(columns=ohe_cols).reset_index(drop=True), enc_df], axis=1)
    row["DTI_Ratio_sq"] = row["DTI_Ratio"] ** 2
    row["Credit_Score_sq"] = row["Credit_Score"] ** 2
    row["Applicant_Income_log"] = np.log1p(row["Applicant_Income"])
    row = row.drop(["DTI_Ratio", "Credit_Score"], axis=1, errors="ignore")
    for c in feat_cols:
        if c not in row.columns: row[c] = 0
    row = row[feat_cols]
    row_sc = scaler.transform(row)
    return nb_model.predict(row_sc)[0], nb_model.predict_proba(row_sc)[0]

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────

col_form, col_result = st.columns([1.1, 0.9], gap="large")

with col_form:
    st.markdown('<div class="section-tag">// Financial Profile</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    applicant_income = c1.number_input("Applicant Income ($)", 0, 100000, 12000, 500)
    coapplicant_income = c2.number_input("Coapplicant Income ($)", 0, 50000, 3000, 500)
    loan_amount = c1.number_input("Loan Amount ($)", 1000, 200000, 15000, 1000)
    loan_term = c2.number_input("Loan Term (months)", 6, 360, 36, 6)
    savings = c1.number_input("Savings ($)", 0, 100000, 5000, 500)
    collateral = c2.number_input("Collateral Value ($)", 0, 200000, 20000, 1000)

    st.markdown('<div class="section-tag" style="margin-top:16px;">// Credit & Risk</div>', unsafe_allow_html=True)
    credit_score = st.slider("Credit Score", 550, 800, 710)
    dti_ratio = st.slider("DTI Ratio (Debt-to-Income)", 0.05, 0.60, 0.32, 0.01)
    c3, c4 = st.columns(2)
    existing_loans = c3.number_input("Existing Loans", 0, 10, 1)
    dependents = c4.number_input("Dependents", 0, 10, 1)

    st.markdown('<div class="section-tag" style="margin-top:16px;">// Demographics</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    age = c5.number_input("Age", 18, 75, 35); gender = c6.selectbox("Gender", ["Male", "Female"])
    c7, c8 = st.columns(2)
    marital_status = c7.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education = c8.selectbox("Education Level", ["Graduate", "Not Graduate"])

    st.markdown('<div class="section-tag" style="margin-top:16px;">// Employment & Location</div>', unsafe_allow_html=True)
    c9, c10 = st.columns(2)
    employment = c9.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
    employer_cat = c10.selectbox("Employer Category", ["Private", "Government", "MNC", "Unemployed"])
    c11, c12 = st.columns(2)
    property_area = c11.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    loan_purpose = c12.selectbox("Loan Purpose", ["Home", "Car", "Personal", "Education"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("💳 Predict Loan Approval")

with col_result:
    if predict_btn:
        inputs = {
            "Applicant_Income": applicant_income, "Coapplicant_Income": coapplicant_income, "Age": age,
            "Dependents": dependents, "Credit_Score": credit_score, "Existing_Loans": existing_loans,
            "DTI_Ratio": dti_ratio, "Savings": savings, "Collateral_Value": collateral,
            "Loan_Amount": loan_amount, "Loan_Term": loan_term, "Education_Level": education,
            "Employment_Status": employment, "Marital_Status": marital_status, "Loan_Purpose": loan_purpose,
            "Property_Area": property_area, "Gender": gender, "Employer_Category": employer_cat,
        }
        pred, proba = predict(inputs)
        approved = (pred == 1); conf = proba[1] if approved else proba[0]

        res_class = "result-approved" if approved else "result-rejected"
        res_icon = "✅ Approved" if approved else "❌ Rejected"
        res_color = "#10b981" if approved else "#ef4444"
        
        st.markdown(f"""
        <div class="{res_class}">
          <div style="font-family:DM Mono,monospace;font-size:11px;color:{res_color};letter-spacing:2px;margin-bottom:8px;">✦ DECISION</div>
          <div style="font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;color:{res_color};letter-spacing:-1px;margin-bottom:4px;">{res_icon}</div>
          <div style="font-size:13px;color:#64748b;">Model confidence: <strong style="color:{res_color};">{conf*100:.1f}%</strong></div>
        </div>""", unsafe_allow_html=True)

        # Gauge and Charts... (Same plotly logic, now with fixed quotes)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=conf * 100,
            number={"suffix": "%", "font": {"family": "Syne", "color": "#e2e8f0", "size": 44}, "valueformat": ".1f"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": res_color}, "bgcolor": "#111827"}
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.markdown('<div style="text-align:center;padding:60px 20px;color:#64748b;">'
                    '<div style="font-size:48px;margin-bottom:16px;">💳</div>'
                    '<div style="font-family:Syne,sans-serif;font-size:18px;font-weight:700;color:#e2e8f0;">Fill in details</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS (Performance, Features, Confusion Matrix)
# ─────────────────────────────────────────────
st.markdown("<br>---", unsafe_allow_html=True)
t1, t2, t3 = st.tabs(["📊 Model Comparison", "🔍 Feature Importance", "⚖️ Confusion Matrix"])

with t1:
    st.write("Model comparisons and metric pills...")
with t2:
    st.write("Feature correlations visualization...")
with t3:
    st.write("Confusion matrix heatmap...")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #1e2d45;padding-top:16px;display:flex;justify-content:space-between;">
  <span style="font-family:DM Mono,monospace;font-size:11px;color:#64748b;">CreditWise · Naive Bayes · sklearn</span>
</div>
""", unsafe_allow_html=True)
