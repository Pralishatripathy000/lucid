import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import os

os.makedirs('./models/lucid-layer1', exist_ok=True)

# ── Load activations ──────────────────────────────────
print("Loading activations...")
with open('./data/activations/activations.json', 'r') as f:
    data = json.load(f)

X = np.array(data['features'])
y = np.array(data['labels'])
print(f"Loaded {len(y)} samples, feature dim: {X.shape[1]}")
print(f"Hallucinated: {y.sum()}, Clean: {(y==0).sum()}")

# ── Split ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale ─────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Train Logistic Regression baseline ───────────────
print("\nTraining Logistic Regression probe...")
lr_probe = LogisticRegression(max_iter=1000, random_state=42)
lr_probe.fit(X_train, y_train)

lr_preds = lr_probe.predict(X_test)
lr_probs = lr_probe.predict_proba(X_test)[:, 1]
lr_auroc = roc_auc_score(y_test, lr_probs)
lr_f1    = f1_score(y_test, lr_preds)

print(f"Logistic Regression — AUROC: {lr_auroc:.4f}, F1: {lr_f1:.4f}")

# ── Train MLP probe ───────────────────────────────────
print("\nTraining MLP probe...")
mlp_probe = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
mlp_probe.fit(X_train, y_train)

mlp_preds = mlp_probe.predict(X_test)
mlp_probs = mlp_probe.predict_proba(X_test)[:, 1]
mlp_auroc = roc_auc_score(y_test, mlp_probs)
mlp_f1    = f1_score(y_test, mlp_preds)

print(f"MLP Probe — AUROC: {mlp_auroc:.4f}, F1: {mlp_f1:.4f}")

# ── Full report ───────────────────────────────────────
print("\n── Best Model (MLP) Classification Report ──")
print(classification_report(y_test, mlp_preds, target_names=['Clean', 'Hallucinated']))

# ── Save best model ───────────────────────────────────
print("\nSaving models...")
with open('./models/lucid-layer1/mlp_probe.pkl', 'wb') as f:
    pickle.dump(mlp_probe, f)

with open('./models/lucid-layer1/lr_probe.pkl', 'wb') as f:
    pickle.dump(lr_probe, f)

with open('./models/lucid-layer1/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save metrics
metrics = {
    "logistic_regression": {"auroc": lr_auroc, "f1": lr_f1},
    "mlp_probe":           {"auroc": mlp_auroc, "f1": mlp_f1}
}
with open('./models/lucid-layer1/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nDone! Best AUROC: {max(lr_auroc, mlp_auroc):.4f}")