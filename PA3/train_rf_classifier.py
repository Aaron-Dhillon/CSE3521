# Author: Aaron Dhillon and Ben Varkey
# Description: Train and evaluate Random Forest on word2vec sentiment embeddings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load precomputed embeddings and labels
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Define training set sizes to evaluate
train_sizes = [25, 50, 150, 200, 300]

# Set random seed for reproducibility


print("\nRandom Forest Accuracy Study:")
print("-" * 40)
print("Training Size | Accuracy (%)")
print("-" * 40)

# For each training size
for size in train_sizes:
    # Generate random indices for subset selection
    indices = np.random.permutation(len(X_train))[:size]
    
    # Create training subset
    X_train_subset = X_train[indices]
    y_train_subset = y_train[indices]
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=5000, random_state=42)
    rf.fit(X_train_subset, y_train_subset)
    
    # Predict and calculate accuracy
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{size:12d} | {acc * 100:10.2f}")

print("-" * 40)