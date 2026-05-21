
#!/usr/bin/env python3
"""Genera probabilidades de abandono con un modelo entrenado.

Uso:
  python predict.py --model models/model.pkl --input new_data.csv --output predictions.csv
"""
import argparse
import pandas as pd
import joblib

def main(model_path: str, input_csv: str, output_csv: str):
    model = joblib.load(model_path)
    X = pd.read_csv(input_csv)
    proba = model.predict_proba(X)[:,1]
    out = pd.DataFrame({'dropout_proba': proba})
    out.to_csv(output_csv, index=False)
    print(f"Predicciones guardadas en {output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='models/model.pkl')
    p.add_argument('--input', type=str, required=True, help='CSV con mismas columnas que entrenamiento (sin dropped_out)')
    p.add_argument('--output', type=str, default='predictions.csv')
    args = p.parse_args()
    main(args.model, args.input, args.output)
