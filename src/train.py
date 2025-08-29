from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data

def main():
    X, y, encoder = load_data("simulation_results")

    # PCA redukcija
    print("Pokrećem PCA...")
    pca = PCA(n_components=20)   # možemo mijenjati broj komponenti
    X_reduced = pca.fit_transform(X)
    print("Shape nakon PCA:", X_reduced.shape)

    # Podjela na train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Izvještaj
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Matrica konfuzije
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

if __name__ == "__main__":
    main()
