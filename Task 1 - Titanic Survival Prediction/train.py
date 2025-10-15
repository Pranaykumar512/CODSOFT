import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

DATA_PATH = r"C:\Users\madhukar\OneDrive\Desktop\CODSOFT\Task 1 - Titanic Survival Prediction\data\Titanic-Dataset.csv"
OUT_DIR = "models"
OUT_PATH = os.path.join(OUT_DIR, "titanic_pipeline.sav")

def add_features(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r',\s*([^.]*)\.', expand=False).str.strip()
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona','Sir','the Countess'],
        'Rare')
    df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['CabinDeck'] = df['Cabin'].fillna('U').astype(str).str[0]
    df['Fare'] = df['Fare'].fillna(0.0)
    df['Age'] = df['Age']  # will impute later
    return df

def make_ohe():
    # handle different sklearn versions (sparse vs sparse_output)
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown='ignore')

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load
    df = pd.read_csv(DATA_PATH)
    print("Loaded:", DATA_PATH, "shape:", df.shape)

    # Feature engineering
    df = add_features(df)

    # Select features
    FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                'Title', 'FamilySize', 'IsAlone', 'CabinDeck']
    X = df[FEATURES]
    y = df['Survived'].astype(int)

    #  Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    #  Preprocessing pipelines
    num_features = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch']
    cat_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'CabinDeck']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', make_ohe())
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ], remainder='drop')

    #  Full pipeline with classifier
    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    # Train
    print("Training model...")
    pipeline.fit(X_train, y_train)
    print("Training finished.")

    #  Evaluation
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    print("Accuracy:", accuracy_score(y_test, y_pred))
    try:
        print("ROC AUC:", roc_auc_score(y_test, y_prob))
    except Exception:
        pass
    print("Classification report:\n", classification_report(y_test, y_pred))

    # Optional CV (5-fold ROC AUC)
    try:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
        print("CV ROC AUC (5-fold): mean {:.4f} ± {:.4f}".format(cv_scores.mean(), cv_scores.std()))
    except Exception:
        pass

    #  Save pipeline
    joblib.dump(pipeline, OUT_PATH)
    print("Saved pipeline to:", OUT_PATH)

    #  Verify by loading and predicting on one sample
    loaded = joblib.load(OUT_PATH)
    sample = X_test.iloc[[0]]
    print("Sample input:\n", sample.to_dict(orient='records')[0])
    print("Loaded pipeline prediction (probability of survival):", loaded.predict_proba(sample)[:,1][0])
    print("Loaded pipeline predicts:", loaded.predict(sample)[0])

if __name__ == "__main__":
    main()
