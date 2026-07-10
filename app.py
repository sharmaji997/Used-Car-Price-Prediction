import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

h1{
    color:#1f77b4;
    text-align:center;
}

.stButton>button{
    width:100%;
    height:50px;
    background:#1f77b4;
    color:white;
    font-size:18px;
    border-radius:10px;
}

.stButton>button:hover{
    background:#0d5ea8;
}

.metric-container{
    border-radius:15px;
    padding:20px;
}

</style>
""",unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------

with open("car_price_model.pkl","rb") as f:
    model = pickle.load(f)

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("cardekho_dataset.csv")

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0",axis=1,inplace=True)

# -----------------------------
# Header
# -----------------------------

st.title("🚗 AI Used Car Price Prediction")

st.markdown("""
Predict the resale value of a used car using a Machine Learning model trained on 15,000+ cars.
""")

st.divider()

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("🚘 Enter Car Details")

# Dataset Information

st.sidebar.markdown("### Dataset Information")

st.sidebar.write(f"Cars : **{len(df)}**")

st.sidebar.write(f"Brands : **{df['brand'].nunique()}**")

st.sidebar.write(f"Models : **{df['model'].nunique()}**")

st.sidebar.divider()

# -----------------------------
# Dynamic Brand & Model
# -----------------------------

brand = st.sidebar.selectbox(
    "Brand",
    sorted(df["brand"].unique())
)

available_models = sorted(
    df[df["brand"]==brand]["model"].unique()
)

model_name = st.sidebar.selectbox(
    "Model",
    available_models
)

# -----------------------------
# Remaining Inputs
# -----------------------------

car_name = st.sidebar.selectbox(
    "Car Name",
    sorted(
        df[
            (df["brand"]==brand) &
            (df["model"]==model_name)
        ]["car_name"].unique()
    )
)

seller_type = st.sidebar.selectbox(
    "Seller Type",
    sorted(df["seller_type"].unique())
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    sorted(df["fuel_type"].unique())
)

transmission_type = st.sidebar.selectbox(
    "Transmission",
    sorted(df["transmission_type"].unique())
)

vehicle_age = st.sidebar.slider(
    "Vehicle Age",
    0,
    20,
    5
)

km_driven = st.sidebar.number_input(
    "Kilometers Driven",
    0,
    500000,
    40000
)

mileage = st.sidebar.number_input(
    "Mileage (km/l)",
    5.0,
    40.0,
    18.0
)

engine = st.sidebar.number_input(
    "Engine (CC)",
    500,
    5000,
    1200
)

max_power = st.sidebar.number_input(
    "Max Power (BHP)",
    20.0,
    500.0,
    90.0
)

seats = st.sidebar.selectbox(
    "Seats",
    sorted(df["seats"].unique())
)

# -----------------------------
# Display Selected Inputs
# -----------------------------

st.subheader("Selected Car Details")

input_df = pd.DataFrame({
    "Car Name":[car_name],
    "Brand":[brand],
    "Model":[model_name],
    "Vehicle Age":[vehicle_age],
    "KM Driven":[km_driven],
    "Seller Type":[seller_type],
    "Fuel Type":[fuel_type],
    "Transmission":[transmission_type],
    "Mileage":[mileage],
    "Engine":[engine],
    "Max Power":[max_power],
    "Seats":[seats]
})

st.dataframe(
    input_df,
    use_container_width=True
)

st.divider()

# -----------------------------
# Prediction Button
# -----------------------------

predict = st.button("🚗 Predict Selling Price")
# -----------------------------
# Prediction
# -----------------------------

if predict:

    try:

        input_data = pd.DataFrame({

            "car_name":[car_name],
            "brand":[brand],
            "model":[model_name],
            "vehicle_age":[vehicle_age],
            "km_driven":[km_driven],
            "seller_type":[seller_type],
            "fuel_type":[fuel_type],
            "transmission_type":[transmission_type],
            "mileage":[mileage],
            "engine":[engine],
            "max_power":[max_power],
            "seats":[seats]

        })

        prediction = model.predict(input_data)

        predicted_price = prediction[0]

        st.divider()

        st.success("Prediction Successful!")

        col1,col2 = st.columns(2)

        with col1:

            st.metric(
                label="Predicted Selling Price",
                value=f"₹ {predicted_price:,.0f}"
            )

        with col2:

            if predicted_price < 300000:
                category = "Budget Car 🚗"

            elif predicted_price < 800000:
                category = "Mid Range Car 🚙"

            elif predicted_price < 1500000:
                category = "Premium Car 🚘"

            else:
                category = "Luxury Car 🏎️"

            st.metric(
                label="Category",
                value=category
            )

        st.divider()

        st.subheader("Prediction Summary")

        st.info(
            f"""
Estimated Market Value

₹ {predicted_price:,.0f}

The value is predicted using a Random Forest Machine Learning model.
            """
        )

        st.progress(100)

        st.subheader("Selected Features")

        st.dataframe(
            input_data,
            use_container_width=True
        )

        st.subheader("Important Factors Affecting Price")

        st.write("✔ Vehicle Age")

        st.write("✔ Kilometers Driven")

        st.write("✔ Brand")

        st.write("✔ Engine Capacity")

        st.write("✔ Maximum Power")

        st.write("✔ Fuel Type")

        st.write("✔ Transmission")

    except Exception as e:

        st.error("Prediction Failed")

        st.code(str(e))

# -----------------------------
# Footer
# -----------------------------

st.divider()

st.markdown(
"""
### About This Project

This application predicts the resale price of a used car using Machine Learning.

**Algorithm Used:** Random Forest Regressor

**Dataset Size:** 15,000+ Used Cars

**Libraries Used:**
- Streamlit
- Pandas
- NumPy
- Scikit-Learn

---
"""
)

