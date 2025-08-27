# predict_probs.py
import pandas as pd, joblib, argparse, os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()

clf = joblib.load(os.path.join("models","classifier_rf.pkl"))
scaler = joblib.load(os.path.join("models","scaler.pkl"))

df = pd.read_csv(args.input)
for c in ["name","status"]:
    if c in df.columns:
        df = df.drop(columns=[c])

# reorder if scaler has feature_names_in_
if hasattr(scaler, "feature_names_in_"):
    df = df[list(scaler.feature_names_in_)]

Xs = scaler.transform(df)
probs = clf.predict_proba(Xs)[:,1] if hasattr(clf, "predict_proba") else None
preds = clf.predict(Xs)

for i,(p,y) in enumerate(zip(preds, probs if probs is not None else [None]*len(preds))):
    if p==1:
        lab="Parkinsons"
    else:
        lab="Healthy"
    if probs is not None:
        print(f"Sample {i+1}: {lab}  (P={p:.0f}, P(Parkinsons)={y:.3f})")
    else:
        print(f"Sample {i+1}: {lab}")
