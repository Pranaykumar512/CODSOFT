import pandas as pd
import os

IN_PATH = r"C:\Users\madhukar\OneDrive\Desktop\CODSOFT\Task 2 - Movie-Rating-Project\data\processed\movies_cleaned.csv"

OUT_PATH = r"C:\Users\madhukar\OneDrive\Desktop\CODSOFT\Task 2 - Movie-Rating-Project\data\processed\movies_features.csv"

def top_k_split(series, k=50):
    top = series.value_counts().head(k).index
    return top

def encode_genres(df, top_n=10):
    df = df.copy()
    if 'genre' not in df.columns:
        return df

    exploded = df['genre'].str.split(',').apply(lambda g: [x.strip().lower() for x in g] if isinstance(g, list) else [])
    
    all_genres = exploded.explode()
    top = all_genres.value_counts().head(top_n).index.tolist()
    for g in top:
        col = f'genre_{g.replace(" ", "_")}'
        df[col] = exploded.apply(lambda lst: int(g in lst))
    return df

def top_director(df, top_k=50):
    df = df.copy()
    if 'director' not in df.columns:
        return df
    top_dirs = top_k_split(df['director'], k=top_k)
    df['director_top'] = df['director'].apply(lambda x: x if x in top_dirs else 'other')
    return df

def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Features input not found: {IN_PATH}")
    df = pd.read_csv(IN_PATH)
    df = encode_genres(df, top_n=12)
    df = top_director(df, top_k=40)
    
    if 'title' in df.columns:
        df['title_len'] = df['title'].astype(str).str.len()
    
    for col in ['runtime','year','votes']:
        if col not in df.columns:
            df[col] = pd.NA
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print("Saved feature-engineered data to:", OUT_PATH)

if __name__ == "__main__":
    main()
