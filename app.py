import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import plotly.graph_objects as go
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
# CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0a0e1a; color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; }
h1, h2, h3 { font-family: 'Syne', sans-serif; letter-spacing: -0.5px; }
.stApp::before {
    content: ''; position: fixed; inset: 0;
    background-image: linear-gradient(rgba(0,229,255,0.025) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(0,229,255,0.025) 1px, transparent 1px);
    background-size: 40px 40px; pointer-events: none; z-index: 0;
}
.imbalance-banner {
    background: linear-gradient(135deg, rgba(245,158,11,0.08), rgba(245,158,11,0.02));
    border: 1px solid rgba(245,158,11,0.35); border-radius: 12px; padding: 16px 20px; margin-bottom: 24px;
}
.result-approved {
    background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(16,185,129,0.03));
    border: 1px solid rgba(16,185,129,0.35); border-radius: 14px; padding: 28px; text-align: center;
}
.result-rejected {
    background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.03));
    border: 1px solid rgba(239,68,68,0.35); border-radius: 14px; padding: 28px; text-align: center;
}
.metric-pill {
    background: #1a2235; border: 1px solid #1e2d45; border-radius: 10px; padding: 12px 16px; text-align: center;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00e5ff, #0099cc) !important;
    border: none !important; border-radius: 10px !important; color: #000 !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 15px !important; padding: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
     padding-bottom:20px;border-bottom:1px solid #1e2d45;margin-bottom:32px;">
  <div style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;">
    Credit<span style="color:#00e5ff;">Wise</span>
  </div>
  <div style="font-family:DM Mono,monospace;font-size:11px;color:#00e5ff;
       border:1px solid #00e5ff;padding:4px 10px;border-radius:4px;opacity:0.75;">
    Naive Bayes · v1.0
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='font-family:Syne,sans-serif;font-size:2.6rem;font-weight:800;letter-spacing:-1.5px;margin-bottom:6px;'>
  Loan Approval
  <span style='background:linear-gradient(90deg,#00e5ff,#7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    Intelligence
  </span>
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
  <div style="font-size:18px;margin-bottom:6px;">⚠️
    <strong style="color:#f59e0b;font-family:Syne,sans-serif;font-size:13px;letter-spacing:1px;text-transform:uppercase;">
      Class Imbalance Notice
    </strong>
  </div>
  <p style="font-size:13px;color:#94a3b8;line-height:1.6;margin-bottom:10px;">
    Training data is <strong>imbalanced</strong>: 70.2% rejections vs 29.8% approvals.
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

use_smote = st.toggle("⚖️ Apply SMOTE (balance classes before training)", value=True)

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

@st.cache_data
def generate_dataset():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "Applicant_ID":       np.arange(n),
        "Applicant_Income":   np.random.uniform(1000, 20000, n),
        "Coapplicant_Income": np.random.choice([0, np.random.uniform(500, 10000)], n),
        "Employment_Status":  np.random.choice(["Salaried","Self-employed","Unemployed"], n, p=[0.55,0.30,0.15]),
        "Age":                np.random.randint(22, 62, n),
        "Marital_Status":     np.random.choice(["Single","Married","Divorced"], n, p=[0.4,0.45,0.15]),
        "Dependents":         np.random.randint(0, 6, n),
        "Credit_Score":       np.random.uniform(550, 800, n),
        "Existing_Loans":     np.random.randint(0, 6, n),
        "DTI_Ratio":          np.random.uniform(0.05, 0.6, n),
        "Savings":            np.random.uniform(0, 20000, n),
        "Collateral_Value":   np.random.uniform(0, 40000, n),
        "Loan_Amount":        np.random.uniform(5000, 50000, n),
        "Loan_Term":          np.random.choice([12,24,36,48,60], n),
        "Loan_Purpose":       np.random.choice(["Home","Car","Personal","Education"], n),
        "Education_Level":    np.random.choice(["Graduate","Not Graduate"], n, p=[0.72,0.28]),
        "Gender":             np.random.choice(["Male","Female"], n, p=[0.62,0.38]),
        "Property_Area":      np.random.choice(["Urban","Semiurban","Rural"], n),
        "Employer_Category":  np.random.choice(["Private","Government","MNC","Unemployed"], n, p=[0.4,0.25,0.2,0.15]),
    })
    score = (
        0.45*(df["Credit_Score"]-550)/250
        -0.44*df["DTI_Ratio"]
        +0.12*df["Applicant_Income"]/20000
        -0.13*df["Loan_Amount"]/50000
        +0.07*(df["Employer_Category"]=="MNC").astype(int)
        +0.03*(df["Loan_Purpose"]=="Personal").astype(int)
        + np.random.normal(0, 0.12, n)
    )
    df["Loan_Approved"] = np.where(score > np.percentile(score, 70), "Yes", "No")
    return df

df_raw = generate_dataset()

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

@st.cache_data
def compute_metrics(use_smote_flag):
    df = df_raw.copy().drop("Applicant_ID", axis=1)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != "Loan_Approved"]
    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
    df["Education_Level"] = LabelEncoder().fit_transform(df["Education_Level"])
    df["Loan_Approved"]   = LabelEncoder().fit_transform(df["Loan_Approved"])
    ohe_cols = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False)
    enc = pd.DataFrame(ohe.fit_transform(df[ohe_cols]), columns=ohe.get_feature_names_out(ohe_cols))
    df  = pd.concat([df.drop(columns=ohe_cols).reset_index(drop=True), enc], axis=1)
    df["DTI_Ratio_sq"]         = df["DTI_Ratio"]**2
    df["Credit_Score_sq"]      = df["Credit_Score"]**2
    df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])
    X = df.drop(["Loan_Approved","DTI_Ratio","Credit_Score"], axis=1)
    y = df["Loan_Approved"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.33, random_state=42)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    if use_smote_flag:
        X_tr_sc, y_tr = SMOTE(random_state=42).fit_resample(X_tr_sc, y_tr)
    out = {}
    for name, model in [("Naive Bayes", GaussianNB()),
                        ("Logistic Reg.", LogisticRegression(max_iter=1000)),
                        ("KNN", KNeighborsClassifier(n_neighbors=5))]:
        model.fit(X_tr_sc, y_tr)
        yp = model.predict(X_te_sc)
        out[name] = {
            "accuracy":  round(accuracy_score(y_te, yp)*100, 1),
            "precision": round(precision_score(y_te, yp, zero_division=0)*100, 1),
            "f1":        round(f1_score(y_te, yp, zero_division=0)*100, 1),
            "recall":    round(recall_score(y_te, yp, zero_division=0)*100, 1),
            "cm":        confusion_matrix(y_te, yp).tolist(),
        }
    return out

# ─────────────────────────────────────────────
# SKLEARN OBJECTS
# ─────────────────────────────────────────────

@st.cache_resource
def build_pipeline(use_smote_flag):
    df = df_raw.copy().drop("Applicant_ID", axis=1)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != "Loan_Approved"]
    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
    le_edu = LabelEncoder()
    df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])
    df["Loan_Approved"]   = LabelEncoder().fit_transform(df["Loan_Approved"])
    ohe_cols = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False)
    enc = pd.DataFrame(ohe.fit_transform(df[ohe_cols]), columns=ohe.get_feature_names_out(ohe_cols))
    df  = pd.concat([df.drop(columns=ohe_cols).reset_index(drop=True), enc], axis=1)
    df["DTI_Ratio_sq"]         = df["DTI_Ratio"]**2
    df["Credit_Score_sq"]      = df["Credit_Score"]**2
    df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])
    X = df.drop(["Loan_Approved","DTI_Ratio","Credit_Score"], axis=1)
    y = df["Loan_Approved"]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.33, random_state=42)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    if use_smote_flag:
        X_tr_sc, y_tr = SMOTE(random_state=42).fit_resample(X_tr_sc, y_tr)
    nb = GaussianNB().fit(X_tr_sc, y_tr)
    return {"nb": nb, "scaler": sc, "ohe": ohe, "ohe_cols": ohe_cols,
            "le_edu": le_edu, "feature_cols": list(X.columns)}

with st.spinner("Training models..."):
    all_metrics = compute_metrics(use_smote)
    pipe        = build_pipeline(use_smote)

nb_m      = all_metrics["Naive Bayes"]
lr_m      = all_metrics["Logistic Reg."]
knn_m     = all_metrics["KNN"]
nb_cm     = np.array(nb_m["cm"])
nb_model  = pipe["nb"]
scaler    = pipe["scaler"]
ohe       = pipe["ohe"]
ohe_cols  = pipe["ohe_cols"]
le_edu    = pipe["le_edu"]
feat_cols = pipe["feature_cols"]

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────

def predict(inputs):
    row = pd.DataFrame([inputs])
    row["Education_Level"] = le_edu.transform(row["Education_Level"])
    enc = pd.DataFrame(ohe.transform(row[ohe_cols]), columns=ohe.get_feature_names_out(ohe_cols))
    row = pd.concat([row.drop(columns=ohe_cols).reset_index(drop=True), enc], axis=1)
    row["DTI_Ratio_sq"]         = row["DTI_Ratio"]**2
    row["Credit_Score_sq"]      = row["Credit_Score"]**2
    row["Applicant_Income_log"] = np.log1p(row["Applicant_Income"])
    row = row.drop(["DTI_Ratio","Credit_Score"], axis=1, errors="ignore")
    for c in feat_cols:
        if c not in row.columns: row[c] = 0
    row_sc = scaler.transform(row[feat_cols])
    pred   = nb_model.predict(row_sc)[0]
    proba  = nb_model.predict_proba(row_sc)[0]
    return pred, proba

# ─────────────────────────────────────────────
# FORM + RESULT
# ─────────────────────────────────────────────

col_form, col_result = st.columns([1.1, 0.9], gap="large")

with col_form:
    st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#00e5ff;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">// Financial Profile</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    applicant_income   = c1.number_input("Applicant Income ($)",   0, 100000, 12000, 500)
    coapplicant_income = c2.number_input("Coapplicant Income ($)", 0, 50000,  3000,  500)
    loan_amount        = c1.number_input("Loan Amount ($)",        1000, 200000, 15000, 1000)
    loan_term          = c2.number_input("Loan Term (months)",     6, 360, 36, 6)
    savings            = c1.number_input("Savings ($)",            0, 100000, 5000, 500)
    collateral         = c2.number_input("Collateral Value ($)",   0, 200000, 20000, 1000)

    st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#00e5ff;letter-spacing:2px;text-transform:uppercase;margin:16px 0 10px;">// Credit & Risk</div>', unsafe_allow_html=True)
    credit_score = st.slider("Credit Score", 550, 800, 710)
    dti_ratio    = st.slider("DTI Ratio", 0.05, 0.60, 0.32, 0.01)
    c3, c4 = st.columns(2)
    existing_loans = c3.number_input("Existing Loans", 0, 10, 1)
    dependents     = c4.number_input("Dependents",     0, 10, 1)

    st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#00e5ff;letter-spacing:2px;text-transform:uppercase;margin:16px 0 10px;">// Demographics</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    age            = c5.number_input("Age", 18, 75, 35)
    gender         = c6.selectbox("Gender", ["Male","Female"])
    c7, c8 = st.columns(2)
    marital_status = c7.selectbox("Marital Status", ["Single","Married","Divorced"])
    education      = c8.selectbox("Education Level", ["Graduate","Not Graduate"])

    st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#00e5ff;letter-spacing:2px;text-transform:uppercase;margin:16px 0 10px;">// Employment & Location</div>', unsafe_allow_html=True)
    c9, c10 = st.columns(2)
    employment   = c9.selectbox("Employment Status", ["Salaried","Self-employed","Unemployed"])
    employer_cat = c10.selectbox("Employer Category", ["Private","Government","MNC","Unemployed"])
    c11, c12 = st.columns(2)
    property_area = c11.selectbox("Property Area", ["Urban","Semiurban","Rural"])
    loan_purpose  = c12.selectbox("Loan Purpose", ["Home","Car","Personal","Education"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("💳  Predict Loan Approval")

with col_result:
    if predict_btn:
        inputs = {
            "Applicant_Income": applicant_income, "Coapplicant_Income": coapplicant_income,
            "Age": age, "Dependents": dependents, "Credit_Score": credit_score,
            "Existing_Loans": existing_loans, "DTI_Ratio": dti_ratio,
            "Savings": savings, "Collateral_Value": collateral,
            "Loan_Amount": loan_amount, "Loan_Term": loan_term,
            "Education_Level": education, "Employment_Status": employment,
            "Marital_Status": marital_status, "Loan_Purpose": loan_purpose,
            "Property_Area": property_area, "Gender": gender,
            "Employer_Category": employer_cat,
        }
        pred, proba = predict(inputs)
        approved = (pred == 1)
        conf     = proba[1] if approved else proba[0]

        if approved:
            st.markdown(f"""
            <div class="result-approved">
              <div style="font-family:DM Mono,monospace;font-size:11px;color:#10b981;letter-spacing:2px;margin-bottom:8px;">✦ DECISION</div>
              <div style="font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;color:#10b981;letter-spacing:-1px;margin-bottom:4px;">✅ Approved</div>
              <div style="font-size:13px;color:#64748b;">Model confidence: <strong style="color:#10b981;">{conf*100:.1f}%</strong></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-rejected">
              <div style="font-family:DM Mono,monospace;font-size:11px;color:#ef4444;letter-spacing:2px;margin-bottom:8px;">✦ DECISION</div>
              <div style="font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;color:#ef4444;letter-spacing:-1px;margin-bottom:4px;">❌ Rejected</div>
              <div style="font-size:13px;color:#64748b;">Model confidence: <strong style="color:#ef4444;">{conf*100:.1f}%</strong></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf*100,
            number={"suffix":"%","font":{"family":"Syne","color":"#e2e8f0","size":44},"valueformat":".1f"},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#64748b","tickfont":{"color":"#64748b","size":11},"nticks":6},
                "bar":{"color":"#10b981" if approved else "#ef4444","thickness":0.3},
                "bgcolor":"#111827","borderwidth":1,"bordercolor":"#1e2d45",
                "steps":[{"range":[0,50],"color":"rgba(239,68,68,0.07)"},
                         {"range":[50,75],"color":"rgba(245,158,11,0.07)"},
                         {"range":[75,100],"color":"rgba(16,185,129,0.07)"}],
                "threshold":{"line":{"color":"#00e5ff","width":3},"thickness":0.85,"value":conf*100},
            },
            title={"text":"MODEL CONFIDENCE","font":{"family":"DM Mono","color":"#64748b","size":11}},
            domain={"x":[0,1],"y":[0.1,1]},
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                height=280,margin=dict(t=40,b=30,l=30,r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

        fig_prob = go.Figure(go.Bar(
            x=["Rejected","Approved"], y=[proba[0]*100, proba[1]*100],
            marker_color=["#ef4444","#10b981"], marker_line_width=0,
            text=[f"{proba[0]*100:.1f}%",f"{proba[1]*100:.1f}%"],
            textposition="outside", textfont={"color":"#e2e8f0","size":12},
        ))
        fig_prob.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            height=200,margin=dict(t=10,b=10,l=10,r=10),
            yaxis=dict(visible=False,range=[0,120]),
            xaxis=dict(tickfont={"color":"#64748b","size":12}),
            showlegend=False,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        dti_color  = "#ef4444" if dti_ratio > 0.4 else "#10b981"
        cred_color = "#10b981" if credit_score >= 700 else "#ef4444"
        st.markdown(f"""
        <div style="background:#1a2235;border:1px solid #1e2d45;border-radius:10px;padding:14px;">
          <div style="font-family:DM Mono,monospace;font-size:10px;color:#00e5ff;
               letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">// Key Factors</div>
          <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-size:12px;color:#64748b;">Credit Score</span>
            <span style="font-family:DM Mono,monospace;font-size:12px;color:{cred_color};">
              {credit_score} {'✓' if credit_score>=700 else '✗'}
            </span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-size:12px;color:#64748b;">DTI Ratio</span>
            <span style="font-family:DM Mono,monospace;font-size:12px;color:{dti_color};">
              {dti_ratio:.2f} {'✓' if dti_ratio<=0.4 else '✗'}
            </span>
          </div>
          <div style="display:flex;justify-content:space-between;">
            <span style="font-size:12px;color:#64748b;">Loan / Income Ratio</span>
            <span style="font-family:DM Mono,monospace;font-size:12px;color:#e2e8f0;">
              {loan_amount/max(applicant_income,1):.2f}x
            </span>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#64748b;">
          <div style="font-size:48px;margin-bottom:16px;">💳</div>
          <div style="font-family:Syne,sans-serif;font-size:18px;font-weight:700;margin-bottom:8px;color:#e2e8f0;">
            Fill in applicant details
          </div>
          <div style="font-size:13px;line-height:1.6;">
            Complete the form and click<br><strong>Predict Loan Approval</strong>
          </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="font-family:DM Mono,monospace;font-size:10px;color:#00e5ff;
     letter-spacing:2px;text-transform:uppercase;margin-bottom:24px;">
// Model Analytics
</div>""", unsafe_allow_html=True)

st.markdown("### 📊 Model Comparison")

fig1 = go.Figure()
for lbl, vals, clr in [
    ("Accuracy",  [nb_m["accuracy"],  lr_m["accuracy"],  knn_m["accuracy"]],  "#00e5ff"),
    ("Precision", [nb_m["precision"], lr_m["precision"], knn_m["precision"]], "#10b981"),
    ("F1 Score",  [nb_m["f1"],        lr_m["f1"],        knn_m["f1"]],        "#7c3aed"),
    ("Recall",    [nb_m["recall"],    lr_m["recall"],    knn_m["recall"]],    "#f59e0b"),
]:
    fig1.add_trace(go.Bar(
        name=lbl,
        x=["Naive Bayes","Logistic Reg.","KNN"],
        y=vals,
        marker_color=clr, marker_line_width=0,
        text=[f"{v}%" for v in vals],
        textposition="outside",
        textfont={"size":11,"color":"#e2e8f0"},
    ))

fig1.update_layout(
    barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=380, legend=dict(font=dict(color="#94a3b8",size=11), bgcolor="rgba(0,0,0,0)"),
    yaxis=dict(range=[0,118], ticksuffix="%", tickfont={"color":"#64748b"}, gridcolor="#1e2d45"),
    xaxis=dict(tickfont={"color":"#e2e8f0","size":13}),
    margin=dict(t=30,b=10,l=10,r=10),
)
st.plotly_chart(fig1, use_container_width=True)

ma, mb, mc, md = st.columns(4)
for col, lbl, val, clr in [
    (ma,"Accuracy", nb_m["accuracy"], "#00e5ff"),
    (mb,"Precision",nb_m["precision"],"#10b981"),
    (mc,"F1 Score", nb_m["f1"],       "#7c3aed"),
    (md,"Recall",   nb_m["recall"],   "#f59e0b"),
]:
    col.markdown(f"""
    <div class="metric-pill">
      <div style="font-family:DM Mono,monospace;font-size:10px;color:#64748b;
           letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">{lbl}</div>
      <div style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;color:{clr};">{val}%</div>
      <div style="font-size:11px;color:#64748b;">Naive Bayes</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("### 🔍 Feature Correlations")
feat_names = ["DTI_Ratio","Loan_Amount","Loan_Term","Gender_Male","Education_Level",
              "Collateral_Value","Loan_Purpose_Pers","Employer_MNC","Applicant_Income","Credit_Score"]
feat_vals  = [-0.445,-0.126,-0.087,-0.054,-0.053, 0.022, 0.034, 0.069, 0.120, 0.451]

fig2 = go.Figure(go.Bar(
    x=feat_vals, y=feat_names, orientation="h",
    marker_color=["#ef4444" if v < 0 else "#10b981" for v in feat_vals],
    marker_line_width=0, text=[f"{v:+.3f}" for v in feat_vals],
    textposition="auto", textfont={"color":"#ffffff","size":12},
))
fig2.add_vline(x=0, line_width=1, line_color="#64748b")
fig2.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400,
    xaxis=dict(title=dict(text="Correlation with Loan_Approved", font={"color":"#64748b","size":11}),
               tickfont={"color":"#64748b"}, gridcolor="#1e2d45", zerolinecolor="#64748b"),
    yaxis=dict(tickfont={"color":"#e2e8f0","size":12}),
    margin=dict(t=20,b=30,l=10,r=10), bargap=0.25,
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
<div style="font-size:13px;color:#64748b;line-height:1.8;padding:8px 0 16px;">
  🟢 <strong style="color:#10b981;">Credit Score</strong> — strongest positive signal (+0.451)<br>
  🔴 <strong style="color:#ef4444;">DTI Ratio</strong> — strongest negative signal (−0.445)<br>
  Income, employer type, and loan amount are secondary drivers.
</div>
""", unsafe_allow_html=True)

st.markdown("### ⚖️ Confusion Matrix — Naive Bayes")
cm_labels = ["Rejected", "Approved"]
annots = [dict(x=cm_labels[j], y=cm_labels[i], text=f"<b>{nb_cm[i,j]}</b>", showarrow=False,
               font={"size":32,"color":"#ffffff"}) for i in range(2) for j in range(2)]

fig3 = go.Figure(go.Heatmap(
    z=nb_cm, x=cm_labels, y=cm_labels,
    colorscale=[[0.0,"#0f1e33"],[0.5,"#1e3a5f"],[1.0,"#00e5ff"]], showscale=False,
))
fig3.update_layout(
    annotations=annots, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=340,
    xaxis=dict(title=dict(text="Predicted", font={"color":"#64748b","size":12}),
               tickfont={"color":"#e2e8f0","size":14}, side="bottom"),
    yaxis=dict(title=dict(text="Actual", font={"color":"#64748b","size":12}),
               tickfont={"color":"#e2e8f0","size":14}, autorange="reversed"),
    margin=dict(t=20,b=20,l=10,r=10),
)
st.plotly_chart(fig3, use_container_width=True)

tn, fp, fn, tp = nb_cm.ravel()
q1, q2, q3, q4 = st.columns(4)
for col, lbl, val, clr in [(q1,"True Neg",int(tn),"#10b981"),(q2,"False Pos",int(fp),"#f59e0b"),
                            (q3,"False Neg",int(fn),"#f59e0b"),(q4,"True Pos",int(tp),"#10b981")]:
    col.markdown(f"""
    <div class="metric-pill">
      <div style="font-family:DM Mono,monospace;font-size:10px;color:#64748b;
           letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">{lbl}</div>
      <div style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;color:{clr};">{val}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid #1e2d45;padding-top:16px;
     display:flex;justify-content:space-between;align-items:center;">
  <span style="font-family:DM Mono,monospace;font-size:11px;color:#64748b;">
    CreditWise · Naive Bayes · sklearn · 1000 samples
  </span>
  <span style="font-family:DM Mono,monospace;font-size:11px;color:#64748b;">
    SMOTE · StandardScaler · LabelEncoder · OneHotEncoder
  </span>
</div>
""", unsafe_allow_html=True)
