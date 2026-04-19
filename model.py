# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 13:20:26 2025

@author: munna
"""

# 1. Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap
import matplotlib.pyplot as plt
import joblib  # ✅ added for saving model and encoder

# 2. Load dataset
df = pd.read_csv(r"C:\embro_quality\DATA_SET\data_after_EDA.csv")

# 3. Define features
features = ['FSH(mIU/mL)', 'LH(mIU/mL)', 'Age (yrs)', 'AMH(ng/mL)', 'BMI', 'AFC']

# 4. Create multi-class target based on AFC
def classify_afc(afc):
    if afc <= 8:
        return 'Poor'
    elif 9 <= afc <= 12:
        return 'Fair'
    else:
        return 'Good'

df['Embryo_Quality'] = df['AFC'].apply(classify_afc)

# 5. Prepare features and target
X = df[features]
y = df['Embryo_Quality']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 7. Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# ✅ Save model and label encoder for Streamlit app
joblib.dump(model, "xgb_embryo_model.pkl")
joblib.dump(le, "label_encoder.pkl")

# 8. Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Feature Importance
print("\n📌 Feature Importances:")
importances = model.feature_importances_
for col, imp in zip(features, importances):
    print(f"{col}: {imp:.4f}")

# 10. SHAP Explainability
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
