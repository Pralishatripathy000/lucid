import json
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
import os

os.makedirs('./results', exist_ok=True)

# ── Load saved models and data ────────────────────────
print("Loading models and data...")
with open('./models/lucid-layer1/mlp_probe.pkl', 'rb') as f:
    mlp_probe = pickle.load(f)

with open('./models/lucid-layer1/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('./data/activations/activations.json', 'r') as f:
    data = json.load(f)

with open('./models/lucid-layer1/metrics.json', 'r') as f:
    metrics = json.load(f)

X = np.array(data['features'])
y = np.array(data['labels'])
X_scaled = scaler.transform(X)

probs = mlp_probe.predict_proba(X_scaled)[:, 1]
preds = mlp_probe.predict(X_scaled)

# ── Plot 1: ROC Curve ─────────────────────────────────
fpr, tpr, _ = roc_curve(y, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#2D6A9F', lw=2, label=f'MLP Probe (AUROC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Lucid Layer 1 — ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('./results/roc_curve.png', dpi=150)
plt.close()
print("Saved ROC curve!")

# ── Plot 2: PCA of activation space ──────────────────
print("Generating PCA visualization...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
colors = ['#22C55E' if label == 0 else '#EF4444' for label in y]
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.5, s=20)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('Lucid Layer 1 — Activation Space (Green=Clean, Red=Hallucinated)')

from matplotlib.patches import Patch
legend = [Patch(color='#22C55E', label='Clean'), Patch(color='#EF4444', label='Hallucinated')]
plt.legend(handles=legend)
plt.tight_layout()
plt.savefig('./results/activation_space.png', dpi=150)
plt.close()
print("Saved activation space plot!")

# ── Plot 3: Confusion Matrix ──────────────────────────
cm = confusion_matrix(y, preds)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks([0, 1], ['Clean', 'Hallucinated'])
plt.yticks([0, 1], ['Clean', 'Hallucinated'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Lucid Layer 1 — Confusion Matrix')
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
plt.tight_layout()
plt.savefig('./results/confusion_matrix.png', dpi=150)
plt.close()
print("Saved confusion matrix!")

print(f"\n── Final Metrics ──")
print(f"MLP Probe AUROC : {metrics['mlp_probe']['auroc']:.4f}")
print(f"MLP Probe F1    : {metrics['mlp_probe']['f1']:.4f}")
print(f"LR Probe AUROC  : {metrics['logistic_regression']['auroc']:.4f}")
print(f"LR Probe F1     : {metrics['logistic_regression']['f1']:.4f}")
print("\nAll plots saved to ./results/")