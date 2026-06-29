# ==========================================================
# SMART BANKING & LOAN APPROVAL SYSTEM
# ==========================================================

import pandas as pd
import numpy as np

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
# LOAD DATASET
# ==========================================================

data = pd.read_csv("train.csv")

# Remove spaces from column names
data.columns = data.columns.str.strip()

print("Dataset Loaded Successfully")
print(data.head())

# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

data["Gender"] = data["Gender"].fillna(data["Gender"].mode()[0])
data["Married"] = data["Married"].fillna(data["Married"].mode()[0])
data["Dependents"] = data["Dependents"].fillna(data["Dependents"].mode()[0])
data["Self_Employed"] = data["Self_Employed"].fillna(data["Self_Employed"].mode()[0])
data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].median())
data["Loan_Amount_Term"] = data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0])
data["Credit_History"] = data["Credit_History"].fillna(data["Credit_History"].mode()[0])

# ==========================================================
# ENCODE CATEGORICAL DATA
# ==========================================================

encoder = LabelEncoder()

categorical = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in categorical:
    data[col] = encoder.fit_transform(data[col])

# Remove Loan ID
data = data.drop("Loan_ID", axis=1)

# ==========================================================
# FEATURES AND TARGET
# ==========================================================

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# Keep the feature column order/names so we can reuse them later
# for the new-customer prediction (fixes the "X does not have valid
# feature names" warning).
feature_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# LOGISTIC REGRESSION
# ==========================================================

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, log_pred))

# ==========================================================
# DECISION TREE
# ==========================================================

print("\n==============================")
print("DECISION TREE")
print("==============================")

tree = DecisionTreeClassifier(random_state=42)

tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)

print("Accuracy :", accuracy_score(y_test, tree_pred))

# ==========================================================
# RANDOM FOREST
# ==========================================================

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest.fit(X_train, y_train)

forest_pred = forest.predict(X_test)

print("Accuracy :", accuracy_score(y_test, forest_pred))

# ==========================================================
# KNN
# ==========================================================

print("\n==============================")
print("K NEAREST NEIGHBOR")
print("==============================")

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

print("Accuracy :", accuracy_score(y_test, knn_pred))

# ==========================================================
# LINEAR REGRESSION
# Predict Loan Amount
# ==========================================================

print("\n==============================")
print("LINEAR REGRESSION")
print("==============================")

X_lr = data.drop("LoanAmount", axis=1)

y_lr = data["LoanAmount"]

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_lr,
    y_lr,
    test_size=0.20,
    random_state=42
)

lr = LinearRegression()

lr.fit(X_train_lr, y_train_lr)

loan_pred = lr.predict(X_test_lr)

print("Sample Predicted Loan Amounts")

print(loan_pred[:10])

# ==========================================================
# K-MEANS CLUSTERING
# ==========================================================

print("\n==============================")
print("K-MEANS CLUSTERING")
print("==============================")

cluster_data = data[[
    "ApplicantIncome",
    "LoanAmount",
    "Credit_History"
]]

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(cluster_data)

data["Customer_Group"] = clusters

print(data[[
    "ApplicantIncome",
    "LoanAmount",
    "Customer_Group"
]].head())

# ==========================================================
# PCA
# ==========================================================

print("\n==============================")
print("PCA")
print("==============================")

pca = PCA(n_components=2)

reduced = pca.fit_transform(cluster_data)

print("Original Shape :", cluster_data.shape)

print("Reduced Shape :", reduced.shape)

# ==========================================================
# NEW CUSTOMER PREDICTION
# ==========================================================

print("\n==============================")
print("NEW CUSTOMER LOAN PREDICTION")
print("==============================")

sample = np.array([[

1,      # Gender
1,      # Married
0,      # Dependents
1,      # Education
0,      # Self Employed
5000,   # Applicant Income
2000,   # Coapplicant Income
120,    # Loan Amount
360,    # Loan Term
1,      # Credit History
2       # Property Area

]])

# FIX: wrap the raw array in a DataFrame using the same column
# names/order the scaler was originally fitted on. This removes the
# "X does not have valid feature names, but StandardScaler was fitted
# with feature names" warning, and also protects you from silently
# feeding columns in the wrong order in the future.
sample_df = pd.DataFrame(sample, columns=feature_columns)

sample_scaled = scaler.transform(sample_df)

prediction = forest.predict(sample_scaled)

if prediction[0] == 1:
    print("\nLoan Status : APPROVED")
else:
    print("\nLoan Status : REJECTED")

print("\n===================================")
print("SMART BANKING SYSTEM COMPLETED")
print("===================================")

