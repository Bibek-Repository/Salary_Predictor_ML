import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load the trained model
# -----------------------------
model = joblib.load("linear_model.pkl")

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="salary predictor by Bibek Baiju",
    page_icon="💼",
    layout="centered"
)

# -----------------------------
# Header Section
# -----------------------------
st.markdown("""
<div style="background-color:#4CAF50;padding:20px;border-radius:10px">
<h1 style="color:white;text-align:center;">salary according to years of experience</h1>
<p style="color:white;text-align:center;">Enter your years of experience to predict your expected salary in NPR.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar for input
# -----------------------------
st.sidebar.header("Input years of experience")
salary = st.sidebar.number_input(
    "Enter your years of experience (0.0 - 20.0):", 
    min_value=0.0, max_value=20.0, value=3.0, step=1.0
)

# -----------------------------
# Predict Button & Output
# -----------------------------
if st.button("Predict Salary"):
    X_new = np.array([[salary]])  # Convert input to 2D
    predicted_Salary = model.predict(X_new)[0]

    # Show result in a styled container
    st.markdown(f"""
    <div style="background-color:#e0f7fa;padding:20px;border-radius:10px;margin-top:20px">
    <h2 style="text-align:center;color:#00796b;">Predicted Salary: {predicted_Salary:.2f} NPR</h2>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Optional: Show model details
# -----------------------------
if st.checkbox("Show Model Details"):
    st.markdown(f"""
    <div style="background-color:#f1f8e9;padding:15px;border-radius:10px;margin-top:10px">
    <h3>Model Coefficients</h3>
    <p>Slope (m): {model.coef_[0]:.2f}</p>
    <p>Intercept (b): {model.intercept_:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div style="text-align:center;margin-top:50px;color:gray;">
<p>Developed with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)