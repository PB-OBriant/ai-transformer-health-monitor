# classify_sample.py
# Load the trained model and classify rows from a CSV file.

import argparse
import pandas as pd
import joblib
import os

FEATS = ["V_rms", "I_rms", "THD_I", "Temp_C", "Temp_rate_C_per_s", "Loss_W"]

def main():
    parser = argparse.ArgumentParser(description="Classify transformer health from a CSV of feature rows.")
    parser.add_argument("--model", type=str, default="model_random_forest.pkl", help="Path to trained model pickle")
    parser.add_argument("--input", type=str, required=True, help="CSV file with feature columns")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV for predictions")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found at {args.model}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found at {args.input}")

    clf = joblib.load(args.model)
    df = pd.read_csv(args.input)

    missing = [c for c in FEATS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    preds = clf.predict(df[FEATS].values)
    out = df.copy()
    out["Predicted_Condition"] = preds
    out.to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")

if __name__ == "__main__":
    main()
