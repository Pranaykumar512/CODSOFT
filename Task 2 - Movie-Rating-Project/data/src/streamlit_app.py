import streamlit as st
import joblib
import pandas as pd
import os

MODEL_PATH = r"C:\Users\madhukar\OneDrive\Desktop\CODSOFT\Task 2 - Movie-Rating-Project\models\rating_model.sav"

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run training first (python src/train.py).")
    st.stop()

model = joblib.load(MODEL_PATH)

st.title("IMDb India Movie Rating Predictor")


expected_genres = [
    'genre_family', 'genre_musical', 'genre_mystery', 'genre_comedy',
    'genre_horror', 'genre_thriller', 'genre_action', 'genre_drama',
    'genre_adventure', 'genre_crime', 'genre_fantasy', 'genre_romance'
]

with st.form("movie_form"):
    st.write("Provide movie features (approximate):")
    runtime = st.number_input("Runtime (minutes)", min_value=0, value=120)
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2015)
    votes = st.number_input("Votes (approx)", min_value=0, value=1000)
    title = st.text_input("Title", value="Example Movie")
    director = st.text_input("Director", value="other")

    st.write("Select genres:")
    selected_genres = []
    for genre in expected_genres:
       
        checked = st.checkbox(genre.replace('genre_', '').capitalize())
        if checked:
            selected_genres.append(genre)

    submitted = st.form_submit_button("Predict")

    if submitted:
  
        features = {
            'runtime': runtime,
            'year': year,
            'votes': votes,
            'title_len': len(title),
            'director_top': director,
        }

        for g in expected_genres:
            features[g] = 1 if g in selected_genres else 0

        input_df = pd.DataFrame([features])
        pred = model.predict(input_df)[0]
        st.metric("Predicted Rating (IMDb-style)", round(float(pred), 2))
