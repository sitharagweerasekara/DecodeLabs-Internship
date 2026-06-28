import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("==================================================")
    print("  DecodeLabs AI Phase 2: Data Classification      ")
    print("==================================================")
    
    # 1. LOAD AND UNDERSTAND DATASET
    # Loading the classic Iris flower dataset
    iris = load_iris()
    X = iris.data  # Features: Sepal length, Sepal width, Petal length, Petal width
    y = iris.target  # Labels: 0 (Setosa), 1 (Versicolor), 2 (Virginica)
    
    print(f"[*] Dataset Loaded Successfully.")
    print(f"    - Total Samples: {X.shape[0]}")
    print(f"    - Features per Sample: {X.shape[1]} (Sepal/Petal dimensions)")
    print(f"    - Target Classes: {len(iris.target_names)} ({', '.join(iris.target_names)})")

    # 2. STRUCTURAL INTEGRITY: THE SPLIT
    # Shuffling and splitting into 80% Training and 20% Testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )
    print(f"\n[*] Data Split Complete:")
    print(f"    - Training Set: {X_train.shape[0]} samples")
    print(f"    - Testing Set: {X_test.shape[0]} samples")

    # 3. THE GATEKEEPER RULE: FEATURE SCALING
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[*] Feature Scaling Applied (Mean=0, Variance=1).")

    # 4. APPLY CLASSIFICATION ALGORITHM (KNN)
    # Initializing the K-Nearest Neighbors Classifier (using K=3 neighbors)
    k_neighbors = 3
    model = KNeighborsClassifier(n_neighbors=k_neighbors)
    
    # Training the model
    model.fit(X_train_scaled, y_train)
    print(f"\n[*] KNN Classifier (K={k_neighbors}) Trained Successfully.")

    # 5. MODEL EVALUATION (OUTPUT)
    # Making predictions on the unseen test set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n================ EVALUATION METRICS ================")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("====================================================")

if __name__ == "__main__":
    main()
