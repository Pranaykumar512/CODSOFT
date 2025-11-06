import joblib
import pandas as pd
import os

MODEL_PATH = r"C:\Users\madhukar\OneDrive\Desktop\CODSOFT\Task 2 - Movie-Rating-Project\models\rating_model.sav"

def predict_one(features: dict):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train the model first.")
    pipe = joblib.load(MODEL_PATH)
    X = pd.DataFrame([features])
    pred = pipe.predict(X)[0]
    return float(pred)

if __name__ == "__main__":
    sample = {
    'runtime': 150,
    'year': 2015,
    'votes': 5000,
    'title_len': 12,
    'director_top': 'other',
    
    'genre_action': 0,
    'genre_adventure': 0,
    'genre_comedy': 0,
    'genre_crime': 0,
    'genre_drama': 1,        
    'genre_family': 0,
    'genre_fantasy': 0,
    'genre_horror': 0,
    'genre_musical': 0,
    'genre_mystery': 0,
    'genre_romance': 0,
    'genre_thriller': 0
}

    print("Predicted rating:", predict_one(sample))
