import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

MODEL_PATH = r"C:\Users\pranay\OneDrive\Desktop\CODSOFT\Task 1 - Titanic Survival Prediction\models\titanic_pipeline.sav"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found! Please run train.py first to create titanic_pipeline.sav.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()
 
st.title("🚢 Titanic Survival Prediction App")
st.markdown("### Enter passenger details below to predict if they would have survived.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
        Sex = st.selectbox("Sex", ["male", "female"])
        Age = st.number_input("Age", min_value=0, max_value=100, value=29)
        SibSp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
        Parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
    with col2:
        Fare = st.number_input("Fare Paid", min_value=0.0, value=32.2)
        Embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
        Title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
        CabinDeck = st.selectbox("Cabin Deck", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'])
        IsAlone = st.selectbox("Is Alone?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Survival")

if submitted:

    FamilySize = SibSp + Parch + 1
    IsAlone_val = 1 if IsAlone == "Yes" else 0

    input_df = pd.DataFrame([{
        "Pclass": Pclass,
        "Sex": Sex,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked": Embarked,
        "Title": Title,
        "FamilySize": FamilySize,
        "IsAlone": IsAlone_val,
        "CabinDeck": CabinDeck
    }])

    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        st.success(f"✅ Survival Probability: **{prob*100:.2f}%**")
        if pred == 1:
            st.markdown("🟢 The passenger is **likely to survive**.")
        else:
            st.markdown("🔴 The passenger is **unlikely to survive**.")
    except Exception as e:

        st.error(f"Prediction error: {e}")           
