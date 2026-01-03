import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

FILES = {
    'arima': 'arima_predictions.csv',
    'gbt': 'model_predictions_gbt.csv',
    'lstm': 'model_predictions2-lstm.csv',
    'gru': 'model_predictions_gru.csv',
}

ACTUAL_CANDIDATES = ['actual', 'actual_label', 'y_true', 'label']
PRED_CANDIDATES = ['predicted', 'prediction', 'predicted_label', 'pred']

os.makedirs('confusion_plots', exist_ok=True)

for name, fname in FILES.items():
    if not os.path.exists(fname):
        print(f"Skipping {name}: file {fname} not found")
        continue
    df = pd.read_csv(fname)
    # find actual and predicted columns
    actual_col = next((c for c in ACTUAL_CANDIDATES if c in df.columns), None)
    pred_col = next((c for c in PRED_CANDIDATES if c in df.columns), None)

    # if predicted probability exists but not hard prediction, try thresholding
    if pred_col is None:
        prob_cols = [c for c in df.columns if 'prob' in c.lower() or 'proba' in c.lower()]
        if prob_cols:
            pred_col = 'derived_pred'
            df[pred_col] = (df[prob_cols[0]] >= 0.5).astype(int)

    if actual_col is None or pred_col is None:
        print(f"Skipping {name}: couldn't find actual/predicted columns. cols={list(df.columns)}")
        continue

    y_true = df[actual_col].astype(int)
    y_pred = df[pred_col].astype(int)

    labels = np.unique(np.concatenate([y_true.unique(), y_pred.unique()]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: {name}')
    out_path = os.path.join('confusion_plots', f'confusion_{name}.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    report = classification_report(y_true, y_pred, zero_division=0)
    txt_path = os.path.join('confusion_plots', f'classification_report_{name}.txt')
    with open(txt_path, 'w') as f:
        f.write(report)

    print(f"Saved {out_path} and {txt_path}")

print('Done')
