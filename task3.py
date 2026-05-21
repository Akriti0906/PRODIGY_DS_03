import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("bank.csv", sep=';')

# Display first rows
print(df.head())

# Check missing values
print(df.isnull().sum())

# Encode categorical columns
encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = encoder.fit_transform(df[column])

# Split features and target
X = df.drop('y', axis=1)
y = df['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

# plt.show()

# -----------------------------
# Decision Tree Visualization
# -----------------------------
plt.figure(figsize=(20,10))

plot_tree(
    model,
    filled=True,
    feature_names=X.columns,
    class_names=['No', 'Yes']
)

plt.title("Decision Tree Classifier")

plt.savefig("decision_tree.png")

# plt.show()