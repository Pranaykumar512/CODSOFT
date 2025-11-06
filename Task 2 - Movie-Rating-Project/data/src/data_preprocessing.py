import pandas as pd
import numpy as np
import os

RAW_PATH = r"C:\Users\madhukar\OneDrive\Desktop\CODSOFT\Task 2 - Movie-Rating-Project\data\raw\IMDb Movies India.csv"
OUT_PATH = r"C:\Users\madhukar\OneDrive\Desktop\CODSOFT\Task 2 - Movie-Rating-Project\data\processed\movies_cleaned.csv"

def standardize_colnames(df):
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df

def load_raw(path=RAW_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw dataset not found at: {path}")
    df = pd.read_csv(path, encoding='latin1')

    print("Loaded raw:", df.shape)
    return df

def clean(df):
    df = df.copy()
    df = standardize_colnames(df)

    col_map = {}
    if 'title' not in df.columns:
        for c in df.columns:
            if 'title' in c:
                col_map[c] = 'title'
    if 'genre' not in df.columns:
        for c in df.columns:
            if 'genre' in c:
                col_map[c] = 'genre'
    if 'director' not in df.columns:
        for c in df.columns:
            if 'director' in c:
                col_map[c] = 'director'
    if 'actors' not in df.columns and 'cast' in df.columns:
        col_map['cast'] = 'actors'
    if 'rating' not in df.columns:
        for c in df.columns:
            if c in ['imdb_rating','rating_value','avg_rating','score']:
                col_map[c] = 'rating'
    if 'votes' not in df.columns:
        for c in df.columns:
            if 'vote' in c:
                col_map[c] = 'votes'
    df.rename(columns=col_map, inplace=True)

  
    df.drop_duplicates(inplace=True)

    
    for col in ['rating', 'runtime', 'votes', 'year']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

   
    if 'year' not in df.columns and 'title' in df.columns:
        years = df['title'].astype(str).str.extract(r'\((\d{4})\)')
        if years.dropna().shape[0] > 0:
            df['year'] = pd.to_numeric(years[0], errors='coerce')

   
    if 'genre' in df.columns:
        df['genre'] = df['genre'].astype(str).str.replace('|', ',').str.strip()

    if 'rating' in df.columns:
        df = df.dropna(subset=['rating'])
    else:
        raise KeyError("No 'rating' column found in dataset — cannot continue without target.")

    
    if 'director' in df.columns:
        df = df.dropna(subset=['director'])
    if 'genre' in df.columns:
        df = df.dropna(subset=['genre'])

    return df

def save(df, out_path=OUT_PATH):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Saved cleaned data to:", out_path)

if __name__ == "__main__":
    df_raw = load_raw()
    df_clean = clean(df_raw)
    save(df_clean)
