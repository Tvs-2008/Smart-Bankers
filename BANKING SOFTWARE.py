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
data = pd.read_csv("train.csv")
data.columns = data.columns.str.strip()
print("Dataset Loaded Successfully")
print(data.head())
data["Gender"] = data["Gender"].fillna(data["Gender"].mode()[0])
data["Married"] = data["Married"].fillna(data["Married"].mode()[0])
data["Dependents"] = data["Dependents"].fillna(data["Dependents"].mode()[0])
data["Self_Employed"] = data["Self_Employed"].fillna(data["Self_Employed"].mode()[0])
data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].median())
data["Loan_Amount_Term"] = data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0])
data["Credit_History"] = data["Credit_History"].fillna(data["Credit_History"].mode()[0])
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
data = data.drop("Loan_ID", axis=1)
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]
feature_columns = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
print("Accuracy :", accuracy_score(y_test, log_pred))
print("\n==============================")
print("DECISION TREE")
print("==============================")
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
print("Accuracy :", accuracy_score(y_test, tree_pred))
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
print("\n==============================")
print("K NEAREST NEIGHBOR")
print("==============================")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print("Accuracy :", accuracy_score(y_test, knn_pred))
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
print("\n==============================")
print("PCA")
print("==============================")
pca = PCA(n_components=2)
reduced = pca.fit_transform(cluster_data)
print("Original Shape :", cluster_data.shape)
print("Reduced Shape :", reduced.shape)
print("\n==============================")
print("NEW CUSTOMER LOAN PREDICTION")
print("==============================")
sample = np.array([[
1,      
1,     
0,      
1,      
0,      
5000,   
2000,   
120,    
360,    
1,      
2       
]])
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
