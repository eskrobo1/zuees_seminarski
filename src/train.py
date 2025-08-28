import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_data


def main():
    # učitaj podatke
    X, y, encoder = load_data("simulation_results")

    print("Shape X:", X.shape)
    print("Shape y:", y.shape)

    # podjela train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # baseline model: Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # evaluacija
    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
