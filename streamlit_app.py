# ==========================================================
# SELF-LAUNCHER
# This block lets the file be run directly -- e.g. pressing F5 / "Run
# Module" in IDLE, double-clicking the .py file, or `python
# streamlit_app.py` -- and have it automatically relaunch itself the
# proper way (`streamlit run streamlit_app.py`), which opens the
# dashboard in the browser. No .bat file or terminal command needed.
# ==========================================================

import sys
import os

import streamlit.runtime as st_runtime

if not st_runtime.exists():
    import streamlit.web.cli as st_cli

    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    sys.exit(st_cli.main())

# ==========================================================
# NORMAL APP CODE (only reached once running under Streamlit)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Smart Banking & Loan Approval System",
    page_icon="🏦",
    layout="wide",
)

# ==========================================================
# COLOR THEME
# A navy-and-gold "vault" palette -- deliberately not the default
# Streamlit blue, to feel more like a banking product.
# ==========================================================

COLORS = {
    "bg": "#0B1220",
    "surface": "#141B2D",
    "border": "#263150",
    "gold": "#C9A227",
    "gold_hover": "#E0BB3C",
    "emerald": "#2F9E68",
    "rust": "#C1543C",
    "slate_blue": "#5B7FBF",
    "text": "#EDEAE3",
    "text_muted": "#9CA8BD",
}

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background-color: {COLORS["bg"]};
        color: {COLORS["text"]};
    }}

    h1, h2, h3 {{
        font-family: 'Fraunces', serif;
        color: {COLORS["text"]};
    }}

    h1 {{
        color: {COLORS["gold"]};
    }}

    [data-testid="stSidebar"] {{
        background-color: {COLORS["surface"]};
        border-right: 1px solid {COLORS["border"]};
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: {COLORS["gold"]};
    }}

    .stButton > button,
    .stFormSubmitButton > button {{
        background-color: {COLORS["gold"]};
        color: {COLORS["bg"]};
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }}

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {{
        background-color: {COLORS["gold_hover"]};
        color: {COLORS["bg"]};
    }}

    div[data-testid="stExpander"],
    div[data-testid="stMetric"] {{
        background-color: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
    }}

    [data-testid="stAlert"] {{
        border-radius: 8px;
    }}

    hr {{
        border-color: {COLORS["border"]};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_dark_fig(fig, ax):
    """Apply the navy/gold theme to a matplotlib figure so charts blend
    in with the rest of the dashboard instead of showing a default
    white plot background."""
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["surface"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["border"])
    ax.tick_params(colors=COLORS["text_muted"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    ax.grid(color=COLORS["border"], alpha=0.4, linewidth=0.5)
    return fig, ax

st.title("🏦 Smart Banking & Loan Approval System")
st.caption(
    "Upload your loan dataset to compare classification models, "
    "explore customer segments, and predict approval for a new applicant."
)

# ==========================================================
# LOAD DATA
# ==========================================================

DEFAULT_DATA_PATH = "train.csv"

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV (optional)",
    type="csv",
    help=f"If you skip this, the app will automatically use '{DEFAULT_DATA_PATH}' from the app's folder.",
)


@st.cache_data
def load_data(file_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
    df.columns = df.columns.str.strip()
    return df


if uploaded_file is not None:
    data_raw = load_data(uploaded_file)
    st.sidebar.success("Using uploaded file")
elif os.path.exists(DEFAULT_DATA_PATH):
    data_raw = load_data(DEFAULT_DATA_PATH)
    st.sidebar.info(f"Using local '{DEFAULT_DATA_PATH}'")
else:
    st.info(
        f"Place a file named `{DEFAULT_DATA_PATH}` in the same folder as this app, "
        "or upload a CSV from the sidebar to get started."
    )
    st.stop()

with st.expander("📄 Dataset preview", expanded=False):
    st.dataframe(data_raw.head(10), use_container_width=True)
    st.caption(f"{data_raw.shape[0]} rows × {data_raw.shape[1]} columns")

REQUIRED_COLS = {
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area", "Loan_Status",
}
missing_cols = REQUIRED_COLS - set(data_raw.columns)
if missing_cols:
    st.error(f"Your CSV is missing required column(s): {', '.join(sorted(missing_cols))}")
    st.stop()

# ==========================================================
# PREPROCESSING + ENCODING (cached)
# ==========================================================


@st.cache_data
def preprocess(df: pd.DataFrame):
    df = df.copy()

    for col in ["Gender", "Married", "Dependents", "Self_Employed"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

    categorical = [
        "Gender", "Married", "Dependents", "Education",
        "Self_Employed", "Property_Area", "Loan_Status",
    ]
    encoders = {}
    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    return df, encoders


data, encoders = preprocess(data_raw)

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]
feature_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# TRAIN CLASSIFICATION MODELS (cached)
# ==========================================================


@st.cache_resource
def train_models(X_train_scaled, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    for model in models.values():
        model.fit(X_train_scaled, y_train)
    return models


models = train_models(X_train_scaled, y_train)

st.subheader("📊 Model Accuracy Comparison")

accuracy_rows = []
for name, model in models.items():
    pred = model.predict(X_test_scaled)
    accuracy_rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred)})

results_df = pd.DataFrame(accuracy_rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)

col1, col2 = st.columns([1, 1.3])
with col1:
    st.dataframe(
        results_df.style.format({"Accuracy": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )
    best_row = results_df.iloc[0]
    st.success(f"Best model: **{best_row['Model']}** — {best_row['Accuracy']:.2%} accuracy")
with col2:
    fig_acc, ax_acc = plt.subplots(figsize=(5, 3.2))
    bar_colors = [
        COLORS["emerald"] if m == best_row["Model"] else COLORS["gold"]
        for m in results_df["Model"]
    ]
    ax_acc.bar(results_df["Model"], results_df["Accuracy"], color=bar_colors)
    ax_acc.set_ylim(0, 1)
    ax_acc.set_ylabel("Accuracy")
    style_dark_fig(fig_acc, ax_acc)
    plt.setp(ax_acc.get_xticklabels(), rotation=20, ha="right")
    fig_acc.tight_layout()
    st.pyplot(fig_acc)

# ==========================================================
# LINEAR REGRESSION — PREDICT LOAN AMOUNT
# ==========================================================

st.subheader("💰 Loan Amount Prediction (Linear Regression)")

X_lr = data.drop("LoanAmount", axis=1)
y_lr = data["LoanAmount"]
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_lr, y_lr, test_size=0.20, random_state=42
)
lr = LinearRegression()
lr.fit(X_train_lr, y_train_lr)
loan_pred = lr.predict(X_test_lr)

lr_compare = pd.DataFrame({
    "Actual Loan Amount": y_test_lr.values[:10],
    "Predicted Loan Amount": np.round(loan_pred[:10], 1),
})
st.dataframe(lr_compare, use_container_width=True, hide_index=True)

# ==========================================================
# K-MEANS CUSTOMER SEGMENTATION
# ==========================================================

st.subheader("👥 Customer Segmentation (K-Means)")

cluster_data = data[["ApplicantIncome", "LoanAmount", "Credit_History"]]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(cluster_data)

cluster_palette = ListedColormap([COLORS["gold"], COLORS["emerald"], COLORS["slate_blue"]])

col3, col4 = st.columns(2)
with col3:
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        cluster_data["ApplicantIncome"],
        cluster_data["LoanAmount"],
        c=clusters,
        cmap=cluster_palette,
        alpha=0.85,
        edgecolors="none",
    )
    ax.set_xlabel("Applicant Income")
    ax.set_ylabel("Loan Amount")
    ax.set_title("Customer Groups by Income & Loan Amount")
    style_dark_fig(fig, ax)
    st.pyplot(fig)

# ==========================================================
# PCA
# ==========================================================

with col4:
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(cluster_data)

    fig2, ax2 = plt.subplots()
    ax2.scatter(
        reduced[:, 0], reduced[:, 1],
        c=clusters,
        cmap=cluster_palette,
        alpha=0.85,
        edgecolors="none",
    )
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("PCA Projection (colored by customer group)")
    style_dark_fig(fig2, ax2)
    st.pyplot(fig2)

st.caption(f"Original shape: {cluster_data.shape}  →  Reduced shape: {reduced.shape}")

# ==========================================================
# NEW APPLICANT PREDICTION (SIDEBAR FORM)
# ==========================================================

st.sidebar.divider()
st.sidebar.header("🧮 New Applicant Prediction")

with st.sidebar.form("applicant_form"):
    gender = st.selectbox("Gender", list(encoders["Gender"].classes_))
    married = st.selectbox("Married", list(encoders["Married"].classes_))
    dependents = st.selectbox("Dependents", list(encoders["Dependents"].classes_))
    education = st.selectbox("Education", list(encoders["Education"].classes_))
    self_employed = st.selectbox("Self Employed", list(encoders["Self_Employed"].classes_))
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=2000, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=120, step=10)
    loan_term = st.number_input("Loan Term (days)", min_value=0, value=360, step=30)
    credit_history = st.selectbox("Credit History", ["Has credit history", "No credit history"])
    property_area = st.selectbox("Property Area", list(encoders["Property_Area"].classes_))
    model_choice = st.selectbox("Model to use", list(models.keys()), index=list(models.keys()).index("Random Forest"))

    submitted = st.form_submit_button("Predict Loan Status", use_container_width=True)

if submitted:
    input_dict = {
        "Gender": encoders["Gender"].transform([gender])[0],
        "Married": encoders["Married"].transform([married])[0],
        "Dependents": encoders["Dependents"].transform([dependents])[0],
        "Education": encoders["Education"].transform([education])[0],
        "Self_Employed": encoders["Self_Employed"].transform([self_employed])[0],
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": 1 if credit_history == "Has credit history" else 0,
        "Property_Area": encoders["Property_Area"].transform([property_area])[0],
    }

    sample_df = pd.DataFrame([input_dict])[feature_columns]
    sample_scaled = scaler.transform(sample_df)

    chosen_model = models[model_choice]
    prediction = chosen_model.predict(sample_scaled)[0]

    st.session_state["prediction_result"] = {
        "model": model_choice,
        "approved": bool(prediction == 1),
    }
    if hasattr(chosen_model, "predict_proba"):
        proba = chosen_model.predict_proba(sample_scaled)[0]
        st.session_state["prediction_result"]["probability"] = float(proba[1])
    else:
        st.session_state["prediction_result"]["probability"] = None

# ==========================================================
# SHOW PREDICTION RESULT
# ==========================================================

if "prediction_result" in st.session_state:
    result = st.session_state["prediction_result"]
    st.subheader("🧾 Prediction Result")

    if result["approved"]:
        st.success(f"✅ Loan Status: **APPROVED**  (model: {result['model']})")
    else:
        st.error(f"❌ Loan Status: **REJECTED**  (model: {result['model']})")

    if result["probability"] is not None:
        st.progress(result["probability"])
        st.caption(f"Estimated approval probability: {result['probability']:.1%}")

st.sidebar.divider()
st.sidebar.caption("Smart Banking System • built with Streamlit")
