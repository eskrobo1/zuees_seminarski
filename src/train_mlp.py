import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_dft import FaultDFTDataset
from model_mlp import MLP_DFT_Classifier


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50):
    model.to(device)
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, Z_batch, y_batch in train_loader:
            X_batch, Z_batch, y_batch = X_batch.to(device), Z_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch, Z_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, Z_batch, y_batch in val_loader:
                X_batch, Z_batch, y_batch = X_batch.to(device), Z_batch.to(device), y_batch.to(device)
                outputs = model(X_batch, Z_batch)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_mlp.pth")
            print(f"🌟 New best model saved (val_acc={best_val_acc:.4f})")

    return model


def evaluate_model(model, test_loader, encoder, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, Z_batch, y_batch in test_loader:
            X_batch, Z_batch, y_batch = X_batch.to(device), Z_batch.to(device), y_batch.to(device)
            outputs = model(X_batch, Z_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    print("=== Classification Report ===")
    print(classification_report(
        all_labels,
        all_preds,
        labels=range(len(encoder.classes_)),   # forsira sve klase
        target_names=encoder.classes_,
        zero_division=0                        # sprječava greške za prazne klase
    ))

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=range(len(encoder.classes_))    # isto ovdje – sve klase uključene
    )
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def main():
    dataset = FaultDFTDataset("simulation_results", timesteps=651, fs=1000.0)
    encoder = dataset.encoder

    # Split
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)
    test_loader = DataLoader(test_set, batch_size=16)

    # Model
    input_dim = dataset.X.shape[1]
    num_classes = len(encoder.classes_)
    model = MLP_DFT_Classifier(input_dim=input_dim, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50)

    model.load_state_dict(torch.load("best_mlp.pth", map_location=device))
    evaluate_model(model, test_loader, encoder, device)


if __name__ == "__main__":
    main()
