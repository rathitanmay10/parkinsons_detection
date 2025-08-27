import pandas as pd
import joblib
import argparse

# Load models
clf = joblib.load("models/classifier_rf.pkl")
scaler = joblib.load("models/scaler.pkl")

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Path to CSV file with features")
args = parser.parse_args()

X_new = pd.read_csv(args.input)
X_new_scaled = scaler.transform(X_new)
preds = clf.predict(X_new_scaled)

for i, p in enumerate(preds):
    print(f"Sample {i+1}: {'Parkinsons' if p==1 else 'Healthy'}")
