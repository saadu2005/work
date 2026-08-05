import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report

# ==================================
# Load Dataset
# ==================================

df = pd.read_csv("fish_data.csv")

# ==================================
# DATASET EXPLORATION
# ==================================

print("\n----- First 10 Rows -----")
print(df.head(10))

print("\n----- Last 10 Rows -----")
print(df.tail(10))

print("\n----- Dataset Shape -----")
print(df.shape)

print("\n----- Column Names -----")
print(df.columns)

print("\n----- Data Types -----")
print(df.dtypes)

print("\n----- Statistical Summary -----")
print(df.describe())

print("\n----- Missing Values Before Handling -----")
print(df.isnull().sum())

# ==================================
# HANDLE MISSING VALUES
# ==================================

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\n----- Missing Values After Handling -----")
print(df.isnull().sum())

# ==================================
# DATA PREPROCESSING
# ==================================

# Species (text) ko number mein convert kar rahe hain
# (Encoding the target column into numbers)
le = LabelEncoder()
df["Species_Encoded"] = le.fit_transform(df["Species"])

# ==================================
# FEATURES AND TARGET
# ==================================

X = df[["Weight", "Length1", "Length2", "Length3", "Height", "Width"]]
y = df["Species_Encoded"]

print("\n----- Final Missing Values Check -----")
print(X.isnull().sum())

# ==================================
# TRAIN TEST SPLIT
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==================================
# DECISION TREE CLASSIFIER
# ==================================

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# ==================================
# PREDICTION
# ==================================

y_pred = model.predict(X_test)

# ==================================
# RESULTS
# ==================================

print("\n----- Predictions -----")
print(y_pred)

print("\n----- Accuracy -----")
print(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

# -------------------------
# Classification Report
# -------------------------
print("\n========== Classification Report ==========")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# ----------------------------------
# 1. Feature Importance Chart
# ----------------------------------
# Yeh batayega ke Decision Tree ne node splitting mein kis feature ko kitna weigh kiya
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(X.shape[1]), importances[indices], color="mediumseagreen", align="center", edgecolor="black")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.title("Decision Tree - Feature Importance", fontsize=14, fontweight='bold')
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("01_decision_tree_Feature_Importance.png", dpi=300)
plt.show()

# ----------------------------------
# 2. Confusion Matrix Chart
# ----------------------------------
# Yeh batayega ke Decision Tree classifier ne kis machli ko sahi pehchana aur kahan galti ki
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

# Iske liye hum 'viridis' use kar rahe hain
disp.plot(cmap='viridis', ax=plt.gca(), xticks_rotation=45)

plt.title("Decision Tree - Confusion Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("01_decision_tree_Confusion_Matrix.png", dpi=300)
plt.show()

# ----------------------------------
# 3. Decision Tree Visualization (Extra)
# ----------------------------------
# Yeh actual tree structure draw karega taake rules dekhe ja sakein
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=feature_names, class_names=le.classes_, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree - Structure", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("01_decision_tree_Structure.png", dpi=300)
plt.show()