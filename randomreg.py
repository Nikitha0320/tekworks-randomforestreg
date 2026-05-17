import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Title
# -----------------------------
st.title("🚗 Car Price Prediction using Random Forest Regressor")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("cardekho_data.csv")

# -----------------------------
# Brand Extraction
# -----------------------------
def get_brand(name):
    name = name.lower()

    if any(x in name for x in ["city", "brio", "amaze", "jazz", "honda", "activa"]):
        return "Honda"

    elif any(x in name for x in ["verna", "i20", "i10", "eon", "grand", "creta", "xcent", "elantra"]):
        return "Hyundai"

    elif any(x in name for x in ["corolla", "fortuner", "innova", "etios", "camry"]):
        return "Toyota"

    elif any(x in name for x in ["swift", "ciaz", "alto", "ertiga", "sx4", "ritz", "wagon",
                                 "dzire", "baleno", "vitara", "omni", "ignis", "800",
                                 "suzuki", "s "]):
        return "Maruti"

    elif "bajaj" in name:
        return "Bajaj"

    elif "royal" in name:
        return "Royal Enfield"

    elif "hero" in name:
        return "Hero"

    elif "yamaha" in name:
        return "Yamaha"

    elif "tvs" in name:
        return "TVS"

    elif "ktm" in name:
        return "KTM"

    elif "mahindra" in name:
        return "Mahindra"

    elif "land" in name:
        return "Land Rover"

    elif "hyosung" in name:
        return "Hyosung"

    elif "um" in name:
        return "UM"

    else:
        return name.split()[0].capitalize()


df["Brand"] = df["Car_Name"].apply(get_brand)
df.drop("Car_Name", axis=1, inplace=True)

# Optional outlier handling
df = df[df["Kms_Driven"] < 300000]

# -----------------------------
# Encoding
# -----------------------------
le_brand = LabelEncoder()
le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_transmission = LabelEncoder()

df["Brand"] = le_brand.fit_transform(df["Brand"])
df["Fuel_Type"] = le_fuel.fit_transform(df["Fuel_Type"])
df["Seller_Type"] = le_seller.fit_transform(df["Seller_Type"])
df["Transmission"] = le_transmission.fit_transform(df["Transmission"])

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model Training
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Car Details")

brand = st.selectbox("Brand", le_brand.classes_)

year = st.number_input("Year", min_value=2000, max_value=2026, value=2018)

present_price = st.number_input(
    "Present Price in Lakhs",
    min_value=0.0,
    max_value=100.0,
    value=5.0
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=300000,
    value=30000
)

fuel_type = st.selectbox("Fuel Type", le_fuel.classes_)

seller_type = st.selectbox("Seller Type", le_seller.classes_)

transmission = st.selectbox("Transmission", le_transmission.classes_)

owner = st.number_input(
    "Number of Previous Owners",
    min_value=0,
    max_value=5,
    value=0
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Selling Price"):

    brand_encoded = le_brand.transform([brand])[0]
    fuel_encoded = le_fuel.transform([fuel_type])[0]
    seller_encoded = le_seller.transform([seller_type])[0]
    transmission_encoded = le_transmission.transform([transmission])[0]

    input_data = pd.DataFrame({
        "Year": [year],
        "Present_Price": [present_price],
        "Kms_Driven": [kms_driven],
        "Fuel_Type": [fuel_encoded],
        "Seller_Type": [seller_encoded],
        "Transmission": [transmission_encoded],
        "Owner": [owner],
        "Brand": [brand_encoded]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Selling Price: ₹{prediction:.2f} Lakhs")