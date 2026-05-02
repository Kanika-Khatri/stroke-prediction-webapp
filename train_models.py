"""
Stroke Prediction - Model Training Script
Trains Decision Tree, Random Forest, SVC, and a GradientBoosting model.
Handles class imbalance via class_weight and threshold tuning for high Recall.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    recall_score, precision_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline

# LOAD REAL DATASET
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = df.drop(columns=["id"])
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
df = df[df["gender"] != "Other"]

print(f"Dataset shape: {df.shape}")
print(f"Stroke rate: {df['stroke'].mean():.2%}")
print(f"Dataset shape: {df.shape}")
print(f"Stroke rate: {df['stroke'].mean():.2%}")



# ──────────────────────────────────────────────
# 2.  PREPROCESSING
# ──────────────────────────────────────────────
le_dict = {}
cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ──────────────────────────────────────────────
# 3.  TRAIN MODELS
# ──────────────────────────────────────────────
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=10,
        class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=42
    ),
    "SVC": SVC(
        kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced", probability=True, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    ),
}

results   = {}
trained   = {}
threshold = 0.30   # Lower threshold → higher Recall

for name, model in models.items():
    print(f"\nTraining {name}...")
    if name == "SVC":
        model.fit(X_train_sc, y_train)
        proba = model.predict_proba(X_test_sc)[:, 1]
    elif name == "Gradient Boosting":
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

    y_pred = (proba >= threshold).astype(int)
    recall    = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, proba)
    cm        = confusion_matrix(y_test, y_pred).tolist()

    results[name] = {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "confusion_matrix": cm,
    }
    trained[name] = (model, proba)
    print(f"  Recall={recall:.3f}  Precision={precision:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

# ──────────────────────────────────────────────
# 4.  SAVE ARTIFACTS
# ──────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
os.makedirs("static/img", exist_ok=True)

best_name = max(results, key=lambda k: results[k]["recall"])
best_model, _ = trained[best_name]

artifacts = {
    "best_model": best_model,
    "best_name":  best_name,
    "scaler":     scaler,
    "le_dict":    le_dict,
    "feature_names": list(X.columns),
    "threshold":  threshold,
}
with open("models/artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

with open("models/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nBest model (highest Recall): {best_name}")

# ──────────────────────────────────────────────
# 5.  GENERATE CHARTS
# ──────────────────────────────────────────────
PALETTE = {
    "bg":     "#0d1117",
    "card":   "#161b22",
    "accent": "#58a6ff",
    "green":  "#3fb950",
    "yellow": "#d29922",
    "red":    "#f85149",
    "text":   "#c9d1d9",
    "muted":  "#8b949e",
}

# ── 5a. Model Comparison Bar Chart ──────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["card"])

metric_keys = ["recall", "precision", "f1", "auc"]
metric_labels = ["Recall", "Precision", "F1 Score", "AUC-ROC"]
model_names   = list(results.keys())
x = np.arange(len(model_names))
w = 0.18
bar_colors = [PALETTE["green"], PALETTE["accent"], PALETTE["yellow"], "#a371f7"]

for i, (mk, ml, bc) in enumerate(zip(metric_keys, metric_labels, bar_colors)):
    vals = [results[m][mk] for m in model_names]
    bars = ax.bar(x + i*w, vals, w, label=ml, color=bc, alpha=0.9, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=7, color=PALETTE["text"], fontweight="bold")

ax.set_xticks(x + w*1.5)
ax.set_xticklabels(model_names, color=PALETTE["text"], fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", color=PALETTE["muted"], fontsize=10)
ax.set_title("Model Performance Comparison", color=PALETTE["text"], fontsize=13, fontweight="bold", pad=12)
ax.tick_params(colors=PALETTE["muted"])
ax.spines[:].set_color(PALETTE["muted"])
ax.spines[:].set_alpha(0.3)
ax.grid(axis="y", color=PALETTE["muted"], alpha=0.15, zorder=0)
ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["muted"],
          labelcolor=PALETTE["text"], fontsize=9)
plt.tight_layout()
plt.savefig("static/img/model_comparison.png", dpi=130, bbox_inches="tight",
            facecolor=PALETTE["bg"])
plt.close()

# ── 5b. ROC Curves ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["card"])
roc_colors = [PALETTE["green"], PALETTE["accent"], PALETTE["yellow"], "#a371f7"]

for (name, (model, proba)), color in zip(trained.items(), roc_colors):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = results[name]["auc"]
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.2f})")

ax.plot([0,1],[0,1], "--", color=PALETTE["muted"], lw=1, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate", color=PALETTE["muted"])
ax.set_ylabel("True Positive Rate", color=PALETTE["muted"])
ax.set_title("ROC Curves", color=PALETTE["text"], fontsize=13, fontweight="bold")
ax.tick_params(colors=PALETTE["muted"])
ax.spines[:].set_color(PALETTE["muted"]); [s.set_alpha(0.3) for s in ax.spines.values()]
ax.grid(color=PALETTE["muted"], alpha=0.12)
ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["muted"],
          labelcolor=PALETTE["text"], fontsize=9)
plt.tight_layout()
plt.savefig("static/img/roc_curves.png", dpi=130, bbox_inches="tight",
            facecolor=PALETTE["bg"])
plt.close()

# ── 5c. Feature Importance (Random Forest) ──────
rf = trained["Random Forest"][0]
importances = rf.feature_importances_
feat_df = pd.DataFrame({"feature": X.columns, "importance": importances})
feat_df = feat_df.sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["card"])
colors = [PALETTE["accent"] if v > feat_df["importance"].median() else PALETTE["muted"]
          for v in feat_df["importance"]]
bars = ax.barh(feat_df["feature"], feat_df["importance"], color=colors, alpha=0.9)
ax.set_xlabel("Importance", color=PALETTE["muted"])
ax.set_title("Feature Importances (Random Forest)", color=PALETTE["text"],
             fontsize=13, fontweight="bold")
ax.tick_params(colors=PALETTE["text"])
ax.spines[:].set_color(PALETTE["muted"]); [s.set_alpha(0.3) for s in ax.spines.values()]
ax.grid(axis="x", color=PALETTE["muted"], alpha=0.12)
plt.tight_layout()
plt.savefig("static/img/feature_importance.png", dpi=130, bbox_inches="tight",
            facecolor=PALETTE["bg"])
plt.close()

# ── 5d. Stroke Distribution Pie ─────────────────
fig, ax = plt.subplots(figsize=(5, 4), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])
counts = df["stroke"].value_counts()
ax.pie(counts, labels=["No Stroke","Stroke"], colors=[PALETTE["green"], PALETTE["red"]],
       autopct="%1.1f%%", startangle=140,
       textprops={"color": PALETTE["text"], "fontsize": 11},
       wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 2})
ax.set_title("Class Distribution", color=PALETTE["text"], fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("static/img/class_dist.png", dpi=130, bbox_inches="tight",
            facecolor=PALETTE["bg"])
plt.close()

print("\n✅ All models trained and charts saved!")
print(f"   Charts: static/img/")
print(f"   Model:  models/artifacts.pkl")
