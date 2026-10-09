# PRODIGY_DS_03 - Bank Marketing: predicting term deposit subscription
#
# Real-life problem: a bank phones thousands of customers to sell term deposits,
# but most customers say no. This script builds a model that helps decide
# which customers are worth calling.
#
# Steps: 1 Load  2 Check data  3 Explore  4 Encode  5 Split
#        6 Train and compare models  7 Charts  8 Save the model

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save charts as files (no pop-up windows)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

COLOR = "#3B6EA5"                          # one colour for all single-series charts
os.makedirs("outputs", exist_ok=True)      # all charts and tables go in this folder

# ---------------------------------------------------------------
# 1. LOAD THE DATA
# ---------------------------------------------------------------
df = pd.read_csv("bank.csv", sep=";")      # this file uses ; instead of ,
print("Rows, columns:", df.shape)
print(df.head())

# ---------------------------------------------------------------
# 2. CHECK DATA QUALITY
# ---------------------------------------------------------------
print("\nMissing values per column:")
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Class balance: how many customers said yes and no?
counts = df["y"].value_counts()
print("\nCustomers who subscribed (yes) and did not (no):")
print(counts)
baseline_accuracy = counts["no"] / len(df)
print(f"A model that always says 'no' would be right {baseline_accuracy:.1%} of the time.")

# ---------------------------------------------------------------
# 3. EXPLORE THE DATA (3 simple charts)
# ---------------------------------------------------------------
# Chart 1: class balance
plt.figure(figsize=(6, 4))
ax = sns.barplot(x=counts.index, y=counts.values, color=COLOR)
ax.bar_label(ax.containers[0], padding=3)
plt.title("Did the customer subscribe to a term deposit?")
plt.xlabel("Subscribed")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.savefig("outputs/class_balance.png", dpi=150)
plt.close()

# Chart 2: subscription rate for each job
rate_by_job = df.groupby("job")["y"].apply(lambda s: (s == "yes").mean() * 100)
rate_by_job = rate_by_job.sort_values()
plt.figure(figsize=(8, 5))
rate_by_job.plot(kind="barh", color=COLOR)
plt.title("Subscription Rate by Job")
plt.xlabel("Customers who subscribed (%)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("outputs/rate_by_job.png", dpi=150)
plt.close()

# Chart 3: subscription rate by the outcome of the previous campaign
rate_by_prev = df.groupby("poutcome")["y"].apply(lambda s: (s == "yes").mean() * 100)
rate_by_prev = rate_by_prev.sort_values()
plt.figure(figsize=(7, 4))
rate_by_prev.plot(kind="barh", color=COLOR)
plt.title("Subscription Rate by Previous Campaign Outcome")
plt.xlabel("Customers who subscribed (%)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("outputs/rate_by_previous_outcome.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. ENCODE TEXT COLUMNS AS NUMBERS
# ---------------------------------------------------------------
# Models need numbers. We keep one encoder per column so that predict.py
# can convert a new customer's details in exactly the same way.
encoders = {}
for column in df.select_dtypes(exclude="number").columns:
    encoders[column] = LabelEncoder()
    df[column] = encoders[column].fit_transform(df[column])   # no -> 0, yes -> 1

# 'duration' is how long the last call lasted. We only know it AFTER calling
# the customer, so it cannot be used to decide who to call. We leave it out.
features = [c for c in df.columns if c not in ["y", "duration"]]
X = df[features]
y = df["y"]

# ---------------------------------------------------------------
# 5. SPLIT INTO TRAINING AND TEST DATA
# ---------------------------------------------------------------
# stratify=y keeps the same yes/no ratio in both parts.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("\nTraining rows:", len(X_train), " Test rows:", len(X_test))

# ---------------------------------------------------------------
# 6. TRAIN AND COMPARE MODELS
# ---------------------------------------------------------------
results = []

def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"{name:34s} accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    results.append([name, round(acc, 3), round(prec, 3), round(rec, 3), round(f1, 3)])

print("\nResults on the test set (precision/recall/F1 are for the 'yes' class):")

# Model 0: always predict "no"
evaluate("Always predict no", y_test, [0] * len(y_test))

# Model 1: your original Decision Tree (no limits)
tree_original = DecisionTreeClassifier(random_state=42)
tree_original.fit(X_train, y_train)
evaluate("Decision Tree (original)", y_test, tree_original.predict(X_test))
print("   original tree: depth =", tree_original.get_depth(), " leaves =", tree_original.get_n_leaves())

# Model 2: Decision Tree with a depth limit.
# We pick the depth using cross-validation on the TRAINING data only.
# class_weight="balanced" makes the rare 'yes' customers count more.
print("\nChoosing tree depth (5-fold cross-validation, F1 score):")
depths = list(range(2, 11))
depth_scores = []
for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=42)
    score = cross_val_score(tree, X_train, y_train, cv=5, scoring="f1").mean()
    depth_scores.append(score)
    print(f"   depth {depth:2d}: F1 = {score:.3f}")
best_depth = depths[depth_scores.index(max(depth_scores))]
print("Best depth:", best_depth)

plt.figure(figsize=(7, 4))
plt.plot(depths, depth_scores, marker="o", color=COLOR)
plt.title("Choosing the Tree Depth")
plt.xlabel("Maximum depth of the tree")
plt.ylabel("Cross-validated F1 score")
plt.tight_layout()
plt.savefig("outputs/tree_depth_choice.png", dpi=150)
plt.close()

tree_final = DecisionTreeClassifier(max_depth=best_depth, class_weight="balanced", random_state=42)
tree_final.fit(X_train, y_train)
tree_pred = tree_final.predict(X_test)
print()
evaluate("Decision Tree (pruned, balanced)", y_test, tree_pred)

# What this means for the bank, in plain numbers
yes_share = counts["yes"] / len(df)
flagged_share = tree_pred.mean()
print(f"The model flags {flagged_share:.1%} of customers as worth calling.")
print(f"Of those flagged, {precision_score(y_test, tree_pred):.1%} subscribe, "
      f"compared with {yes_share:.1%} when calling customers at random.")
print(f"It finds {recall_score(y_test, tree_pred):.1%} of all customers who would subscribe.")

# Model 3: Random Forest (many trees voting together), for comparison
forest = RandomForestClassifier(n_estimators=100, max_depth=8,
                                class_weight="balanced", random_state=42)
forest.fit(X_train, y_train)
evaluate("Random Forest (balanced)", y_test, forest.predict(X_test))

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
results_df.to_csv("outputs/model_comparison.csv", index=False)

# ---------------------------------------------------------------
# 7. CHARTS FOR THE FINAL MODEL (the pruned Decision Tree)
# ---------------------------------------------------------------
# Confusion matrix
cm = confusion_matrix(y_test, tree_pred)
plt.figure(figsize=(6, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Confusion Matrix (pruned Decision Tree)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# Feature importance: which details did the tree rely on most?
importance = pd.Series(tree_final.feature_importances_, index=features)
importance = importance.sort_values().tail(8)
plt.figure(figsize=(8, 5))
importance.plot(kind="barh", color=COLOR)
plt.title("Most Important Features (pruned Decision Tree)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.close()

# The tree itself (first 3 levels, so it stays readable)
plt.figure(figsize=(22, 10))
plot_tree(tree_final, feature_names=features, class_names=["No", "Yes"],
          filled=True, fontsize=9, max_depth=3)
plt.title("Decision Tree (top 3 levels)")
plt.savefig("outputs/decision_tree.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 8. SAVE THE MODEL FOR predict.py
# ---------------------------------------------------------------
joblib.dump({"model": tree_final, "encoders": encoders, "features": features},
            "model.joblib")
print("\nDone. Charts and tables are in the outputs folder; model saved as model.joblib")
