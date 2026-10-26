"""
Plant Disease Detection - training script (PyTorch + EfficientNetB0 transfer learning)

Kaggle par run karne ke liye:
    !python train.py --data_dir /kaggle/input/plantvillage-dataset/color --out_dir /kaggle/working

Note: Kaggle notebook me Settings -> Internet ON karo (pretrained weights download hote hain)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="ImageFolder style dataset path")
    p.add_argument("--out_dir", default=".", help="model/plots yahan save honge")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Same files, alag transforms: train me augmentation, val/test me nahi
    full_train = datasets.ImageFolder(args.data_dir, transform=train_tf)
    full_eval = datasets.ImageFolder(args.data_dir, transform=eval_tf)
    classes = full_train.classes
    n = len(full_train)
    print(f"Images: {n} | Classes: {len(classes)}")

    idx = np.random.permutation(n)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    train_idx, val_idx, test_idx = np.split(idx, [n_train, n_train + n_val])

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_eval, val_idx)
    test_ds = Subset(full_eval, test_idx)

    kw = dict(batch_size=args.batch_size, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, **kw)
    test_loader = DataLoader(test_ds, **kw)

    # Model: pretrained EfficientNetB0, last layer replace
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def run_epoch(loader, train=False):
        model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += x.size(0)
        return total_loss / total, correct / total

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = 0.0
    model_path = os.path.join(args.out_dir, "model.pth")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        va_loss, va_acc = run_epoch(val_loader)
        scheduler.step()
        for k, v in zip(history, [tr_loss, va_loss, tr_acc, va_acc]):
            history[k].append(v)
        print(f"Epoch {epoch}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), model_path)

    # Best model load karke test karo
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
            labels.extend(y.tolist())
    test_acc = float(np.mean(np.array(preds) == np.array(labels)))
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(classification_report(labels, preds, target_names=classes, digits=3))

    # Save classes + metrics
    with open(os.path.join(args.out_dir, "classes.json"), "w") as f:
        json.dump(classes, f)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": test_acc, "best_val_accuracy": best_val}, f, indent=2)

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "training_curves.png"), dpi=150)
    plt.close()

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
    plt.title(f"Confusion Matrix (test acc {test_acc:.2%})")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved: model.pth, classes.json, metrics.json, training_curves.png, confusion_matrix.png")


if __name__ == "__main__":
    main()
