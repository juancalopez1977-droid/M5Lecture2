
#!/usr/bin/env python3
"""Entrena un modelo de abandono (dropout) para EduStream.

Uso:
  python train.py --data edustream_dropout.csv --out models/model.pkl
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

def load_data(path):
    df = pd.read_csv(path)
    y = df['dropped_out'].astype(int)
    X = df.drop(columns=['dropped_out'])
    return X, y

def build_pipeline(X):
    numeric_features = X.select_dtypes(include=['int64','float64','int32','float32']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    pre = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    clf = LogisticRegression(max_iter=200, n_jobs=None)
    pipe = Pipeline([('pre', pre), ('clf', clf)])
    return pipe

def main(data_path: str, out_path: str):
    X, y = load_data(data_path)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = build_pipeline(X)
    pipe.fit(X_tr, y_tr)
    # Métricas
    proba = pipe.predict_proba(X_te)[:,1]
    auc = roc_auc_score(y_te, proba)
    print(f"ROC-AUC: {auc:.3f}")
    print(classification_report(y_te, (proba>0.5).astype(int)))
    # Guardar
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)
    print(f"Modelo guardado en {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='edustream_dropout.csv')
    p.add_argument('--out', type=str, default='models/model.pkl')
    args = p.parse_args()
    main(args.data, args.out)
