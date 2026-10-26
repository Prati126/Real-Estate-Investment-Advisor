
import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def load_models():
    classifier = joblib.load("classification_model.pkl")
    regressor = joblib.load("regression_model.pkl")

    with open("app_info.json", "r") as f:
        info = json.load(f)

    return classifier, regressor, info

try:
    classifier, regressor, info = load_models()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

st.title("🏠 Real Estate Investment Advisor")
st.write("Analyze a property's investment potential.")

st.info(
    "The five-year estimate assumes fixed 5% annual growth. "
    "It is not a guaranteed market forecast."
)

st.header("Property Details")

with st.form("property_form"):
    col1, col2 = st.columns(2)

    with col1:
        state = st.text_input("State", "West Bengal")
        city = st.text_input("City", "Kolkata")
        locality = st.text_input("Locality", "Central")
        property_type = st.selectbox(
            "Property Type", ["Apartment", "House", "Villa"]
        )
        bhk = st.number_input("BHK", 1, 10, 2)
        size = st.number_input("Size (Sq Ft)", 100, 20000, 1000)
        price = st.number_input(
            "Current Price (Lakhs)", min_value=1.0, value=50.0
        )
        year_built = st.number_input(
            "Year Built", 1950, 2026, 2020
        )
        furnished = st.selectbox(
            "Furnished Status",
            ["Unfurnished", "Semi-Furnished", "Furnished"]
        )
        floor_no = st.number_input("Floor Number", 0, 100, 1)
        total_floors = st.number_input("Total Floors", 1, 100, 5)

    with col2:
        schools = st.number_input("Nearby Schools", 0, 50, 3)
        hospitals = st.number_input("Nearby Hospitals", 0, 50, 2)
        transport = st.selectbox(
            "Public Transport Accessibility", ["Low", "Medium", "High"]
        )
        parking = st.selectbox("Parking Space", ["Yes", "No"])
        security = st.selectbox("Security", ["Yes", "No"])
        amenities = st.selectbox(
            "Amenities", ["Basic", "Standard", "Premium"]
        )
        facing = st.selectbox(
            "Facing", ["North", "South", "East", "West"]
        )
        owner_type = st.selectbox(
            "Owner Type", ["Individual", "Builder", "Dealer"]
        )
        availability = st.selectbox(
            "Availability Status",
            ["Ready_to_Move", "Under_Construction"]
        )

    submitted = st.form_submit_button("Analyze Property")

if submitted:
    age = 2026 - year_built

    property_data = {
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Year_Built": year_built,
        "Furnished_Status": furnished,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Age_of_Property": age,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Amenities": amenities,
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability,
        "Calculated_Age": age,
        "Age_2026": age
    }

    try:
        features = info["classification_features"]
        input_df = pd.DataFrame([property_data]).reindex(
            columns=features
        )

        prediction = classifier.predict(input_df)[0]
        future_price = float(
            regressor.predict(
                pd.DataFrame({"Price_in_Lakhs": [price]})
            )[0]
        )

        st.subheader("Investment Classification")
        st.write(f"**Prediction:** {prediction}")

        st.subheader("Estimated Future Value")
        st.metric("Current Price", f"₹{price:.2f} Lakhs")
        st.metric("Value After 5 Years", f"₹{future_price:.2f} Lakhs")

    except Exception as e:
        st.error(f"Analysis failed: {e}")
