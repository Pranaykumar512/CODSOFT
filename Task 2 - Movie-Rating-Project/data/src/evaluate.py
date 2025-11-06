
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.train import select_features, DATA_PATH, MODEL_PATH

def evaluate():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X, y, _, _ = select_features(df)
    preds = model.predict(X)
    from math import sqrt

    mse = mean_squared_error(y, preds)

    rmse = sqrt(mse)

    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"Evaluate on full data → RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    evaluate()
