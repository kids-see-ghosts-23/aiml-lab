import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Load Titanic dataset
titanic = sns.load_dataset("titanic")

# Select useful columns
titanic = titanic[[
    "survived", "pclass", "sex", "age",
    "sibsp", "parch", "fare", "embarked"
]]

# Handle missing values
titanic["age"].fillna(titanic["age"].median(), inplace=True)
titanic["embarked"].fillna(titanic["embarked"].mode()[0], inplace=True)

# Encode categorical features
le = LabelEncoder()
titanic["sex"] = le.fit_transform(titanic["sex"])        # male=1, female=0
titanic["embarked"] = le.fit_transform(titanic["embarked"])

# Split features and target
X = titanic.drop("survived", axis=1)
y = titanic["survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=["Died", "Survived"]))
