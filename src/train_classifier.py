import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# 1. Load and inspect the dataset

data = pd.read_csv("red_objects.csv")

print("Dataset shape:", data.shape)
print("\nSamples per class:")
print(data["label"].value_counts())

features = [
    "area",
    "width",
    "height",
    "aspect_ratio",
    "perimeter",
    "circularity",
    "solidity",
    "extent",
]

# 2. Prepare inputs (X) and target labels (y)
X = data[features].apply(pd.to_numeric, errors="coerce")
y = data["label"]

# Remove rows with missing feature values or labels
valid = X.notna().all(axis=1) & y.notna()
X = X.loc[valid]
y = y.loc[valid]

if y.nunique() < 2:
    raise ValueError("The dataset needs at least two object classes.")

if y.value_counts().min() < 2:
    raise ValueError(
        "Each class needs at least two samples for this split."
    )

# 3. Split: 75% training, 25% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# 4. Create and train the classifier
model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42,
)

model.fit(X_train, y_train)

# 5. Evaluate on unseen test samples
predictions = model.predict(X_test)

print("\nTest accuracy:",
      round(accuracy_score(y_test, predictions), 3))

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_,
))
print("Class order:", list(model.classes_))

# 6. Save model and feature order
joblib.dump(
    {"model": model, "features": features},
    "red_object_classifier.joblib",
)

print("\nSaved: red_object_classifier.joblib")
