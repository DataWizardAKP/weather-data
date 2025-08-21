import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# ---- Config ----
DATA_PATH = "Weather data.xlsx"   # change if needed
SHEET = "Sheet1"
FEATURE_COLS = [
    "temperature_celsius", "wind_mph", "wind_degree", "pressure_mb",
    "precip_mm", "humidity", "cloud", "uv_index", "gust_mph"
]
TARGET_COL = "feels_like_celsius"
BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 100
VAL_SPLIT = 0.2
PATIENCE = 4
OUT_PICKLE = "feelslike_one.pkl"   # <— single file with model + scaler

# ---- Data ----
df = pd.read_excel(DATA_PATH, sheet_name=SHEET)
X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

X_t = torch.from_numpy(X_scaled)
y_t = torch.from_numpy(y)

ds = TensorDataset(X_t, y_t)

n_val = int(len(ds) * VAL_SPLIT)
n_train = len(ds) - n_val
train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---- Model ----
class MLPRegressor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPRegressor(in_features=X_t.shape[1]).to(device)

crit = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=LR)

# ---- Train (with tiny early stopping) ----
best_val = float("inf")
best_bytes = None
pat = 0

def eval_loss(m, loader):
    m.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = m(xb)
            losses.append(crit(pred, yb).item())
    return float(np.mean(losses)) if losses else float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()

    vloss = eval_loss(model, val_loader)
    # print(f"Epoch {epoch:02d}  val_MSE={vloss:.4f}")
    if vloss < best_val - 1e-6:
        best_val = vloss
        best_bytes = pickle.dumps(model)  # snapshot whole model
        pat = 0
    else:
        pat += 1
        if pat >= PATIENCE:
            break

# restore best
if best_bytes is not None:
    model = pickle.loads(best_bytes)

# ---- Save ONE pickle with model + scaler + metadata ----
bundle = {
    "model": model,                 # full nn.Module object
    "state_dict": model.state_dict(),
    "scaler": scaler,               # fitted StandardScaler
    "feature_cols": FEATURE_COLS,   # to ensure correct ordering
    "target": TARGET_COL,
}


with open(OUT_PICKLE, "wb") as f:
    pickle.dump(bundle, f)

print(f"Saved -> {OUT_PICKLE} (model + scaler in one file)")
