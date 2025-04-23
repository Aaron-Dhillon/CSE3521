# Author: Aaron Dhillon
# Description: Train and evaluate Random Forest on word2vec sentiment embeddings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load precomputed embeddings and labels
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Initialize Random Forest
rf = RandomForestClassifier(n_estimators=1000, random_state=42)

# Train the model
print("Training Random Forest...")
rf.fit(X_train, y_train)
print("Training complete.")

# Predict on test data
y_pred = rf.predict(X_test)

# Evaluate accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {acc * 100:.2f}%")
