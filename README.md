# Titanic Survival Prediction using Naive Bayes
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load train data
train = pd.read_csv("train.csv")

# Drop columns that won't help the model
train = train.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Fill missing values
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])

# Convert categorical data to numbers
le = LabelEncoder()
for col in ["Sex", "Embarked"]:
    train[col] = le.fit_transform(train[col])

# Split features and labels
X = train.drop("Survived", axis=1)
y = train["Survived"]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Evaluation
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=["Not Survived", "Survived"],
            yticklabels=["Not Survived", "Survived"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Load test data and prepare similarly
test = pd.read_csv("test.csv")
passenger_ids = test["PassengerId"]
test = test.drop(columns=["Name", "Ticket", "Cabin"])

# Fill missing values
test["Age"] = test["Age"].fillna(train["Age"].median())
test["Fare"] = test["Fare"].fillna(train["Fare"].median())
test["Embarked"] = test["Embarked"].fillna(train["Embarked"].mode()[0])

# Encode categorical columns
for col in ["Sex", "Embarked"]:
    test[col] = le.fit_transform(test[col])

# Predict on test data
X_test = test.drop(columns=["PassengerId"])
test_preds = model.predict(X_test)

# Save predictions to CSV
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": test_preds
})
submission.to_csv("titanic_naive_bayes_submission.csv", index=False)
print("✅ Submission file saved: titanic_naive_bayes_submission.csv")
