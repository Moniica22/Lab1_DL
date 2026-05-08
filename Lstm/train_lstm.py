import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def find_data_array(mat):
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.size > 1:
            return v
    


def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    X = np.array(X)  # (samples, seq_len, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


def prepare_train_val(mat_path, seq_len, split_ratio=0.8, feature_col=0):
    """
    Returns: X_train, X_val, y_train, y_val, scaler
    """
    mat = loadmat(mat_path)
    arr = find_data_array(mat)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    elif arr.ndim == 2:
        arr = arr[:, feature_col]
    arr = arr.astype(np.float32)

    n = len(arr)
    split_time = int(n * split_ratio)

    # <AI> fit scaler only on data up to split_time
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(arr[:split_time].reshape(-1, 1))

    arr_scaled = scaler.transform(arr.reshape(-1, 1))
    X_all, y_all = create_sequences(arr_scaled, seq_len)

    # </AI>

    # Determine split index in terms of sequence samples: first sample index i has target time i+seq_len
    split_index = max(0, split_time - seq_len)

    X_train = X_all[:split_index]
    X_val = X_all[split_index:]
    y_train = y_all[:split_index]
    y_val = y_all[split_index:]

    return X_train, X_val, y_train, y_val, scaler


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))


def plot_predictions(true_vals, pred_vals, out_path):
    plt.figure(figsize=(10, 4))
    plt.plot(true_vals, label="Ground Truth")
    plt.plot(pred_vals, label="Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train_loop(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def val_loop(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_all = []
    trues_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * xb.size(0)
            preds_all.append(preds.cpu().numpy())
            trues_all.append(yb.cpu().numpy())
    preds_all = np.vstack(preds_all)
    trues_all = np.vstack(trues_all)
    return total_loss / len(loader.dataset), preds_all, trues_all


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch LSTM one-step predictor from Xtrain.mat")
    parser.add_argument("--mat", default="Xtrain.mat", help="Path to .mat file containing training signal")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", default="lstm_output")
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading and preparing data...")
    X_train, X_val, y_train, y_val, scaler = prepare_train_val(args.mat, args.seq_len)

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(
        input_size=1,
        hidden_size=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_loop(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = val_loop(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

    # load best and predict on validation
    model.load_state_dict(torch.load(best_path, map_location=device))
    _, y_pred_scaled, y_val_scaled = val_loop(model, val_loader, criterion, device)

    # inverse scale
    y_val_inv = scaler.inverse_transform(y_val_scaled).reshape(-1)
    y_pred_inv = scaler.inverse_transform(y_pred_scaled).reshape(-1)

    plot_path = os.path.join(args.out_dir, "pred_vs_true.png")
    plot_predictions(y_val_inv, y_pred_inv, plot_path)

    #np.save(os.path.join(args.out_dir, "y_val_true.npy"), y_val_inv)
    #np.save(os.path.join(args.out_dir, "y_val_pred.npy"), y_pred_inv)

    print(f"Finished. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
