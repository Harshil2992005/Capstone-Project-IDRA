# Generates the 3D figures embedded in README.md for IDRA Capstone Project 7.
# Run:  python make_3d_figures.py
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

BASE = r"C:\Users\harshil patel\OneDrive\Desktop\IDRA 2\IDRA CAPSTONE PROJECT 7"
RAW = os.path.join(BASE, "p7.csv")
OUT = os.path.join(BASE, "assets")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"

# ---------------------------------------------------------------- data prep (mirrors notebook cells 7-65)
data = pd.read_csv(RAW)
data["Attrition"] = data["Attrition"].map({"Yes": 1, "No": 0})
data["OverTime"] = data["OverTime"].map({"Yes": 1, "No": 0})
data = data.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"])

data["IncomePerYearWorked"] = data["MonthlyIncome"] / (data["TotalWorkingYears"] + 1)
data["LoyaltyRatio"] = data["YearsAtCompany"] / (data["TotalWorkingYears"] + 1)
data["PromotionLag"] = data["YearsSinceLastPromotion"] / (data["YearsAtCompany"] + 1)

cat_cols = ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus", "Gender"]
data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)
X = data_encoded.drop("Attrition", axis=1)
y = data_encoded["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test_s)
y_pred_rf = rf.predict(X_test)
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("LR cm:", cm_lr.tolist())
print("RF cm:", cm_rf.tolist())
print("LR recall(leaver):", round(cm_lr[1, 1] / max(cm_lr[1].sum(), 1), 3))
print("RF recall(leaver):", round(cm_rf[1, 1] / max(cm_rf[1].sum(), 1), 3))

# ---------------------------------------------------------------- fig 1 : 3D architecture
fig = plt.figure(figsize=(11, 6.5), dpi=150)
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=18, azim=-56)

layers = [
    ("DATA", "raw CSV (1470x35)\ncleaned CSV (1470x34)", "#2ca02c", 1.0),
    ("APPLICATION", "clean  b  engineer\nencode  b  split", "#ff7f0e", 2.0),
    ("CORE", "LogisticRegression\nRandomForest(200)", "#1f77b4", 3.0),
    ("ACCESS", "Jupyter notebook\nIDRA PROJECT_7.ipynb", "#d62728", 4.0),
]
w, d = 0.8, 0.8
for i, (name, sub, color, h) in enumerate(layers):
    x, y = i, 0.5
    ax.bar3d(x, y, 0, w, d, h, color=color, alpha=0.90, edgecolor="black", linewidth=0.5, shade=True)
    ax.text(x + w / 2, y + d / 2, h + 0.35, name, ha="center", va="bottom", fontsize=13, fontweight="bold", color="black")
    ax.text(x + w / 2, y + d / 2, 0.30, sub, ha="center", va="bottom", fontsize=8.5, color="white", linespacing=1.4)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)

arr = Arrow3D([3.0, 3.0], [0.5, 0.5], [4.5, 6.0], mutation_scale=22, lw=2.2, arrowstyle="-|>", color="black")
ax.add_artist(arr)
ax.text(3.0, 0.5, 6.15, "read b transform b model b score b export", ha="center", fontsize=10, fontweight="bold", color="black")

ax.set_xlim(-0.6, 4.0); ax.set_ylim(0, 1.6); ax.set_zlim(0, 7)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_xlabel("pipeline stage", labelpad=8)
ax.set_zlabel("layer depth  (bottom = storage, top = user)", labelpad=12)
ax.set_title("IDRA Capstone 7 - architecture in 3D (the whole spine in one stack)", fontsize=14, pad=8)
for s in ("x", "y", "z"):
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "architecture_3d.png"), bbox_inches="tight")

# ---------------------------------------------------------------- fig 2 : 3D risk scatter (real rows)
fig = plt.figure(figsize=(11, 7), dpi=150)
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=16, azim=-132)

stay = data[data["Attrition"] == 0]
left = data[data["Attrition"] == 1]
ax.scatter(stay["MonthlyIncome"], stay["JobSatisfaction"], stay["YearsAtCompany"],
           c="#1f77b4", s=9, alpha=0.45, label="stayed (1233)")
ax.scatter(left["MonthlyIncome"], left["JobSatisfaction"], left["YearsAtCompany"],
           c="#d62728", s=22, alpha=0.9, marker="^", label="left (237)")
ax.set_xlabel("MonthlyIncome ($)", labelpad=10)
ax.set_ylabel("JobSatisfaction (1-4)", labelpad=10)
ax.set_zlabel("YearsAtCompany", labelpad=8)
ax.set_title("3D risk scatter - income x satisfaction x tenure, coloured by attrition", fontsize=13, pad=6)
ax.grid(alpha=0.25)
ax.legend(loc="upper left", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "risk_scatter_3d.png"), bbox_inches="tight")

# ---------------------------------------------------------------- fig 3 : 3D confusion matrices side by side
fig = plt.figure(figsize=(12, 6), dpi=150)
labels = ["TN", "FP", "FN", "TP"]
positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
for sub, model, cm_, color in ((1, "Logistic Regression", cm_lr, "#1f77b4"),
                               (2, "Random Forest", cm_rf, "#d62728")):
    ax = fig.add_subplot(1, 2, sub, projection="3d")
    ax.view_init(elev=24, azim=-58)
    for (xi, yi), lab, val in zip(positions, labels, cm_.ravel()):
        ax.bar3d(xi, yi, 0, 0.35, 0.35, val, color=color, alpha=0.85, edgecolor="black", linewidth=0.4, shade=True)
        ax.text(xi + 0.175, yi + 0.175, val + 6, f"{lab}={val}", ha="center", fontsize=9, fontweight="bold")
    rec = cm_[1, 1] / max(cm_[1].sum(), 1)
    ax.text(0.5, 1.65, 0, f"recall(leaver) = {rec:.3f}", ha="center", fontsize=11, fontstyle="italic", color="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred: stay", "pred: left"], fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["actual: stay", "actual: left"], fontsize=8)
    ax.set_zticks([])
    ax.set_xlim(-0.4, 1.6); ax.set_ylim(-0.4, 2.0)
    ax.set_title(model, fontsize=12)
    ax.grid(alpha=0.25)
fig.suptitle("3D confusion matrices on the same 294 test rows - acc 0.762 vs 0.844, but recall tells the truth",
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "confusion_3d.png"), bbox_inches="tight")

# ---------------------------------------------------------------- fig 4 : full data pipeline in 3D (serpentine)
from matplotlib.lines import Line2D

stages = [
    ("1. Data Collection", "load raw CSV", "#2ca02c"),
    ("2. Data Observation", "shape - dtypes - describe", "#2ca02c"),
    ("3. Data Quality Check", "0 nulls - 0 dupes - const cols", "#2ca02c"),
    ("4. Data Cleaning", "Yes/No to 1/0 - drop const", "#2ca02c"),
    ("5. Outlier Audit (IQR)", "count, keep real salaries", "#2ca02c"),
    ("6. EDA - 5 figures", "target - income - overtime", "#1f77b4"),
    ("7. Grouped Insights", "risk roles - 5 riskiest combos", "#1f77b4"),
    ("8. Stats: centre & spread", "income right-skewed", "#1f77b4"),
    ("9. Probability & Bayes", "P(Attr|OverTime) = 30.5%", "#1f77b4"),
    ("10. Z-Scores / Normal", "top income z = 2.87", "#1f77b4"),
    ("11. Feature Engineering", "3 ratio features", "#ff7f0e"),
    ("12. Encode + Split", "get_dummies - 80/20 stratify", "#ff7f0e"),
    ("13. Scale (train only)", "StandardScaler - no leakage", "#ff7f0e"),
    ("14. Train LR + RF", "balanced - seed 42", "#d62728"),
    ("15. Model Evaluation", "confusion - AUC - recall", "#d62728"),
    ("16. Overfitting Check", "LR gap 1.8% - RF gap 15.7%", "#d62728"),
    ("17. Feature Importance", "income - tenure - ratios", "#d62728"),
    ("18. Export Cleaned CSV", "1470 rows x 34 cols", "#9467bd"),
]
n = len(stages)
cols, rows = 6, 3
x_ = []
y_ = []
for i in range(n):
    r = i // cols
    c = i % cols
    if r % 2 == 1:
        c = cols - 1 - c
    x_.append(c)
    y_.append(r)

fig = plt.figure(figsize=(17, 8.5), dpi=150)
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=16, azim=-64)
blk_w, blk_d, blk_h = 0.62, 0.62, 0.9
centers = []
for i, ((name, sub, color), rx, ry) in enumerate(zip(stages, x_, y_)):
    cx, cy = rx + blk_w / 2 + 0.10, ry + blk_d / 2 + 0.10
    centers.append((cx, cy))
    ax.bar3d(rx + 0.10, ry + 0.10, 0, blk_w, blk_d, blk_h, color=color, alpha=0.92,
             edgecolor="black", linewidth=0.4, shade=True)
    ax.text(cx, cy, blk_h + 0.10, name, ha="center", va="bottom",
            fontsize=7.0, fontweight="bold", color="black")

for i in range(n - 1):
    x0, y0 = centers[i]
    x1, y1 = centers[i + 1]
    arr = Arrow3D([x0, x1], [y0, y1], [blk_h, blk_h],
                  mutation_scale=14, lw=1.6, arrowstyle="-|>", color="black", alpha=0.8)
    ax.add_artist(arr)

ax.set_xlim(-0.3, cols)
ax.set_ylim(-0.3, rows)
ax.set_zlim(0, 2.2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel("pipeline progress", labelpad=6)
ax.set_ylabel("stage row (serpentine)", labelpad=6)
ax.set_zlabel("stage height", labelpad=6)
ax.set_title("Capstone Project 7 - the full data pipeline in 3D (18 stages, notebook order, cell 7 to cell 96)",
             fontsize=14, pad=6)
ax.grid(alpha=0.20)
legend_handles = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#2ca02c", markersize=12, label="collect & clean (1-5)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#1f77b4", markersize=12, label="observe & statistics (6-10)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#ff7f0e", markersize=12, label="prepare features (11-13)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#d62728", markersize=12, label="model & evaluate (14-17)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#9467bd", markersize=12, label="export (18)"),
]
ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.90, 1.0), fontsize=9, framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pipeline_3d.png"), bbox_inches="tight")
print("pipeline done")

print("done")
print("wrote:", os.listdir(OUT))
# ---------------------------------------------------------------- fig 4 : full data pipeline in 3D (serpentine)
from matplotlib.lines import Line2D

stages = [
    ("1. Data Collection", "load raw CSV", "#2ca02c"),
    ("2. Data Observation", "shape � dtypes � describe", "#2ca02c"),
    ("3. Data Quality Check", "0 nulls � 0 dupes � const cols", "#2ca02c"),
    ("4. Data Cleaning", "Yes/No to 1/0 � drop const", "#2ca02c"),
    ("5. Outlier Audit (IQR)", "count, keep real salaries", "#2ca02c"),
    ("6. EDA - 5 figures", "target � income � overtime", "#1f77b4"),
    ("7. Grouped Insights", "risk roles � 5 riskiest combos", "#1f77b4"),
    ("8. Stats: centre & spread", "income right-skewed", "#1f77b4"),
    ("9. Probability & Bayes", "P(Attr|OverTime) = 30.5%", "#1f77b4"),
    ("10. Z-Scores / Normal", "top income z = 2.87", "#1f77b4"),
    ("11. Feature Engineering", "3 ratio features", "#ff7f0e"),
    ("12. Encode + Split", "get_dummies � 80/20 stratify", "#ff7f0e"),
    ("13. Scale (train only)", "StandardScaler - no leakage", "#ff7f0e"),
    ("14. Train LR + RF", "balanced � seed 42", "#d62728"),
    ("15. Model Evaluation", "confusion � AUC � recall", "#d62728"),
    ("16. Overfitting Check", "LR gap 1.8% � RF gap 15.7%", "#d62728"),
    ("17. Feature Importance", "income � tenure � ratios", "#d62728"),
    ("18. Export Cleaned CSV", "1470 rows x 34 cols", "#9467bd"),
]
n = len(stages)
cols, rows = 6, 3
x_ = []; y_ = []
for i in range(n):
    r = i // cols
    c = i % cols
    if r % 2 == 1:
        c = cols - 1 - c
    x_.append(c); y_.append(r)

fig = plt.figure(figsize=(17, 8.5), dpi=150)
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=16, azim=-64)
blk_w, blk_d, blk_h = 0.62, 0.62, 0.9
centers = []
for i, ((name, sub, color), rx, ry) in enumerate(zip(stages, x_, y_)):
    cx, cy = rx + blk_w / 2 + 0.10, ry + blk_d / 2 + 0.10
    centers.append((cx, cy))
    ax.bar3d(rx + 0.10, ry + 0.10, 0, blk_w, blk_d, blk_h, color=color, alpha=0.92,
             edgecolor="black", linewidth=0.4, shade=True)
    ax.text(cx, cols - 1 - ry if ry == 0 else cy, blk_h + 0.12, name, ha="center", va="bottom",
            fontsize=7.2, fontweight="bold", color="black", zorder=5)
    ax.text(cx, cy, blk_h + 0.12, "", ha="center")

for i in range(n - 1):
    x0, y0 = centers[i]; x1, y1 = centers[i + 1]
    arr = Arrow3D([x0, x1], [y0, y1], [blk_h + 0.02, blk_h + 0.02],
                  mutation_scale=14, lw=1.6, arrowstyle="-|>", color="black", alpha=0.8)
    ax.add_artist(arr)

ax.set_xlim(-0.3, cols); ax.set_ylim(-0.3, rows); ax.set_zlim(0, 2.2)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_xlabel("pipeline progress", labelpad=6)
ax.set_ylabel("stage row (serpentine)", labelpad=6)
ax.set_zlabel("stage height", labelpad=6)
ax.set_title("Capstone Project 7 - the full data pipeline in 3D (18 stages, notebook order, cell 7 to cell 96)",
             fontsize=14, pad=6)
ax.grid(alpha=0.20)
legend_handles = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#2ca02c", markersize=12, label="collect & clean (1-5)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#1f77b4", markersize=12, label="observe & statistics (6-10)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#ff7f0e", markersize=12, label="prepare features (11-13)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#d62728", markersize=12, label="model & evaluate (14-17)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#9467bd", markersize=12, label="export (18)"),
]
ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.92, 1.0), fontsize=9, framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pipeline_3d.png"), bbox_inches="tight")
print("pipeline done")
