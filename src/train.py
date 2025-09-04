import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import FaultDataset
from model import CNN_RNN_Classifier


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=20,
                checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_ckpt = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    best_ckpt = os.path.join(checkpoint_dir, "best_model.pth")

    model.to(device)
    start_epoch = 0
    best_val_acc = 0.0

    # Ako postoji najbolji model -> ucitaj njega
    if os.path.exists(best_ckpt):
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"✅ Loaded best model (epoch {checkpoint['epoch']+1}, val_acc={best_val_acc:.4f})")

    for epoch in range(start_epoch, epochs):
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

        # Validacija
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, Z_batch, y_batch in val_loader:
                X_batch, Z_batch, y_batch = X_batch.to(device), Z_batch.to(device), y_batch.to(device)
                outputs = model(X_batch, Z_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Spremaj zadnji checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc
        }, last_ckpt)

        # Ako je najbolji rezultat -> spremi best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_acc": best_val_acc
            }, best_ckpt)
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
        target_names=encoder.classes_
    ))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def main():
    # Dataset
    dataset = FaultDataset("simulation_results", timesteps=651)
    encoder = dataset.encoder

    # Stratifikacija
    labels = dataset.y.numpy()
    indices = list(range(len(dataset)))

    # prvo Train vs Temp (Val+Test)
    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # onda Val vs Test
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=temp_labels
    )

    # kreiraj podskupove
    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)
    test_set  = Subset(dataset, test_idx)

    # DataLoaderi
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)
    test_loader = DataLoader(test_set, batch_size=16)

    # Model
    input_dim = dataset.X.shape[2]   # broj feature-a po timestepu
    hidden_dim = 128
    num_classes = len(encoder.classes_)

    model = CNN_RNN_Classifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100)

    # Test -> koristi najbolji model ako postoji
    best_ckpt = os.path.join("checkpoints", "best_model.pth")
    if os.path.exists(best_ckpt):
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"✅ Evaluating best model (val_acc={checkpoint['best_val_acc']:.4f})")

    evaluate_model(model, test_loader, encoder, device)


if __name__ == "__main__":
    main()