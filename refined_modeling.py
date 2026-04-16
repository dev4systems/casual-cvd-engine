import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# 1. Load Data
data = pd.read_csv("framingham.csv")

# 2. Basic Cleaning (Dropping rows with critical missing targets or high nulls if necessary)
# For this demonstration, we'll keep it clean but handle imputation properly
# Drop rows where target is missing (none in this dataset usually)
data = data.dropna(subset=['TenYearCHD'])

# 3. Train-Test Split FIRST (Fixes Data Leakage)
X = data.drop('TenYearCHD', axis=1)
y = data['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. KNN Imputation (Fitted only on Training Data)
imputer = KNNImputer(n_neighbors=5)
# Note: We fit on X_train and transform both to prevent information from test set leaking into train
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

print("Imputation Complete. No more missing values in training set:", X_train_imputed.isnull().sum().sum())

# 5. Oversampling ONLY the Training Set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_imputed, y_train)

print(f"Oversampling Complete. Original class distribution: {np.bincount(y_train)}")
print(f"New class distribution: {np.bincount(y_train_res)}")

# 6. Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test_imputed)

# 7. Model Training (Random Forest Example)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train_res)

# 8. Evaluation
y_pred = rf.predict(X_test_scaled)
print("\n--- Model Evaluation (Refined) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Save the cleaned and imputed training data for the causal engine
X_train_res['TenYearCHD'] = y_train_res
X_train_res.to_csv("cleaned_training_data.csv", index=False)
