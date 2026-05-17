# 🚗 Car Price Prediction using Random Forest Regressor

## 📌 Project Overview
This project predicts the selling price of a used car using the Random Forest Regressor algorithm.  
The application is built using Streamlit and allows users to enter car details to get the predicted resale price.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

---

# 📂 Dataset

Dataset used:
- CarDekho Car Price Dataset

Features used:
- Brand
- Year
- Present Price
- Kilometers Driven
- Fuel Type
- Seller Type
- Transmission
- Owner

Target:
- Selling_Price

---

# ⚙️ Preprocessing Steps

The following preprocessing steps were performed:

1. Brand extraction from `Car_Name`
2. Dropped `Car_Name` column
3. Label Encoding for categorical columns
4. Removed extreme outliers in `Kms_Driven`
5. Train-Test Split

---

# 🤖 Machine Learning Algorithm

## Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

# 📊 Model Performance

| Metric | Value |
|---|---|
| Mean Absolute Error | 0.61 |
| Mean Squared Error | 0.80 |
| R² Score | 0.96 |
| Training Score | 0.98 |
| Testing Score | 0.96 |

---

# 🚀 Features of the Application

- Predicts used car selling price
- User-friendly Streamlit interface
- Handles categorical data
- Uses real-world dataset
- Fast and accurate predictions

---

# ▶️ How to Run the Project

## Step 1: Install Requirements

```bash
pip install -r requirements.txt
