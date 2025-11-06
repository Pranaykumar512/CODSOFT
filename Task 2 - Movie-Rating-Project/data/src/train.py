import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = r"C:\Users\madhukar\OneDrive\Desktop\CODSOFT\Task 2 - Movie-Rating-Project\data\processed\movies_features.csv"
MODEL_PATH = r"C:\Users\madhukar\OneDrive\Desktop\CODSOFT\Task 2 - Movie-Rating-Project\models\rating_model.sav"

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def select_features(df):
    
    numeric = [c for c in ['runtime','year','votes','title_len'] if c in df.columns]

    categorical = [c for c in ['director_top'] if c in df.columns]
 
    genres = [c for c in df.columns if c.startswith('genre_')]
    feature_cols = numeric + categorical + genres
    X = df[feature_cols].copy()
    y = df['rating'].astype(float)
    return X, y, numeric, categorical

def build_pipeline(numeric, categorical):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # updated here
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric),
        ('cat', categorical_transformer, categorical)
    ], remainder='passthrough')  # pass genre flags through unchanged

    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    return pipeline

def train():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    df = load_data()
    X, y, numeric, categorical = select_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = build_pipeline(numeric, categorical)
    print("Fitting pipeline...")
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    from math import sqrt

    mse = mean_squared_error(y_test, preds)
    rmse = sqrt(mse)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"RESULTS → RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}")

    # save pipeline in .sav format (joblib)
    joblib.dump(pipe, MODEL_PATH)
    print("Saved pipeline to:", MODEL_PATH)

if __name__ == "__main__":
    train()
