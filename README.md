# 🧬 Embryo Quality Prediction using Machine Learning

## 📌 Overview

This project predicts **Embryo Quality (Poor / Fair / Good)** using clinical parameters from IVF patients.
It combines **Exploratory Data Analysis (EDA)**, **XGBoost classification**, and a **Streamlit web app** for real-time predictions.

---

## 🚀 Features

* 📊 Data preprocessing & cleaning (handling missing values, feature engineering)
* 📈 Exploratory Data Analysis (EDA) with visualizations
* 🤖 Machine Learning model using **XGBoost**
* 🔍 Feature importance + SHAP explainability
* 🌐 Interactive **Streamlit web application**
* 💾 Model persistence using `joblib`

---

## 🧪 Input Features

The model uses the following clinical parameters:

* FSH (mIU/mL)
* LH (mIU/mL)
* Age (years)
* AMH (ng/mL)
* BMI
* AFC (Antral Follicle Count)

---

## 🎯 Target Variable

Embryo Quality is classified based on AFC:

* **Poor** → AFC ≤ 8
* **Fair** → 9 ≤ AFC ≤ 12
* **Good** → AFC > 12

---

## 📂 Project Structure

```
embro_quality/
│
├── DATA_SET/
│   ├── data without infertility _final.csv
│   └── data_after_EDA.csv
│
├── xgb_embryo_model.pkl
├── label_encoder.pkl
│
├── eda.py
├── model.py
├── app.py
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/embryo-quality-prediction.git
cd embryo-quality-prediction
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Exploratory Data Analysis

Run:

```bash
python eda.py
```

This performs:

* Data cleaning
* Feature engineering (AFC creation)
* Missing value handling
* Distribution plots
* Correlation heatmap
* Skewness analysis

---

## 🤖 Model Training

Run:

```bash
python model.py
```

Outputs:

* Trained **XGBoost model**
* Saved files:

  * `xgb_embryo_model.pkl`
  * `label_encoder.pkl`

---

## 🌐 Run Streamlit App

```bash
streamlit run app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## 📸 App Preview

*(Add screenshot here after uploading to GitHub)*

---

## 📈 Model Performance

* Accuracy: ~ (add your value)
* Evaluation metrics:

  * Confusion Matrix
  * Classification Report
  * Feature Importance

---

## 🔍 Explainability

* SHAP values used to interpret model predictions
* Identifies most influential clinical features

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.
It should **NOT be used as a substitute for professional medical advice**.

---

## 👨‍💻 Author

**Munna**

---

## ⭐ Contributing

Pull requests are welcome!
For major changes, please open an issue first.

---

## 📜 License

This project is open-source and available under the **MIT License**.
