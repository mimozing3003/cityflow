import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path(".").resolve()
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "tgsim_transfer"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Define Physical GRU
class TrajPilotPhysicalGRU(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=128, num_layers=2, future_steps=30, target_dim=2, dropout=0.10):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, future_steps * target_dim)
        )
        self.future_steps = future_steps
        self.target_dim = target_dim

    def forward(self, x):
        proj = self.input_projection(x)
        _, hidden = self.encoder(proj)
        context = hidden[-1]
        out = self.decoder(context)
        return out.view(x.shape[0], self.future_steps, self.target_dim)

# Prepare I-90/I-94 Moving dataset with clean vehicle-disjoint split
print("Preparing TGSIM I-90/I-94 train/test split...")
df_i90 = pd.read_csv("data/raw/TGSIM_I90_I94_MOVING/I90_94_moving_final.csv")
v_ids = df_i90["ID"].unique()
np.random.shuffle(v_ids)
split_idx = int(0.7 * len(v_ids))
train_vids = set(v_ids[:split_idx])
test_vids = set(v_ids[split_idx:])

print(f"Total vehicles: {len(v_ids)} | Train vehicles: {len(train_vids)} | Test vehicles: {len(test_vids)}")

def extract_windows_for_vids(df, target_vids, max_wins=4000):
    sub = df[df["ID"].isin(target_vids)]
    groups = sub.groupby(["ID", "run_index"])
    X_list, Y_list, cv_list = [], [], []
    
    for _, v_df in groups:
        if len(v_df) < 60:
            continue
        v_df = v_df.sort_values("time").reset_index(drop=True)
        xs = v_df["xloc_kf"].values
        ys = v_df["yloc_kf"].values
        times = v_df["time"].values
        
        if np.max(np.diff(times)) > 0.25:
            continue
            
        for start in range(0, len(v_df) - 60 + 1, 15):
            hist_x = xs[start : start + 30]
            hist_y = ys[start : start + 30]
            fut_x = xs[start + 30 : start + 60]
            fut_y = ys[start + 30 : start + 60]
            
            x0, y0 = hist_x[-1], hist_y[-1]
            vx0 = (hist_x[-1] - hist_x[-2]) / 0.1
            vy0 = (hist_y[-1] - hist_y[-2]) / 0.1
            v_mag = np.hypot(vx0, vy0)
            psi0 = np.arctan2(vy0, vx0) if v_mag > 0.1 else 0.0
            
            dx_glob = fut_x - x0
            dy_glob = fut_y - y0
            dx_ego = np.cos(psi0) * dx_glob + np.sin(psi0) * dy_glob
            dy_ego = -np.sin(psi0) * dx_glob + np.cos(psi0) * dy_glob
            y_ego = np.stack([dx_ego, dy_ego], axis=-1)
            
            horizons = np.linspace(0.1, 3.0, 30)
            cv_pred = np.stack([v_mag * horizons, np.zeros_like(horizons)], axis=-1)
            
            feats = np.zeros((30, 8), dtype=np.float32)
            for t in range(1, 30):
                vx_t = (hist_x[t] - hist_x[t-1]) / 0.1
                vy_t = (hist_y[t] - hist_y[t-1]) / 0.1
                spd_t = np.hypot(vx_t, vy_t)
                feats[t, 0] = vx_t
                feats[t, 1] = vy_t
                feats[t, 2] = spd_t
                if t >= 2:
                    feats[t, 3] = (spd_t - feats[t-1, 2]) / 0.1
                if t >= 5:
                    feats[t, 4] = (spd_t - feats[t-5, 2]) / 0.5
                if spd_t > 0.1:
                    psi_t = np.arctan2(vy_t, vx_t)
                    psi_prev = np.arctan2(feats[t-1, 1], max(feats[t-1, 0], 1e-4))
                    dpsi = (psi_t - psi_prev + np.pi) % (2 * np.pi) - np.pi
                    feats[t, 5] = dpsi / 0.1
                    feats[t, 6] = feats[t, 5] / max(spd_t, 1.0)
                if t >= 3:
                    feats[t, 7] = (feats[t, 3] - feats[t-1, 3]) / 0.1
            feats[0] = feats[1]
            
            X_list.append(feats)
            Y_list.append(y_ego)
            cv_list.append(cv_pred)
            if len(X_list) >= max_wins:
                break
        if len(X_list) >= max_wins:
            break
            
    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32), np.array(cv_list, dtype=np.float32)

X_train, Y_train, _ = extract_windows_for_vids(df_i90, train_vids, max_wins=3000)
X_test, Y_test, cv_test = extract_windows_for_vids(df_i90, test_vids, max_wins=1500)

print(f"Extracted train windows: {len(X_train)} | test windows: {len(X_test)}")

# Train-only normalization
train_mean = np.mean(X_train, axis=(0, 1))
train_std = np.std(X_train, axis=(0, 1)) + 1e-5

X_train_norm = (X_train - train_mean) / train_std
X_test_norm = (X_test - train_mean) / train_std

# Add validity masks (all 1.0)
X_train_in = np.concatenate([X_train_norm, np.ones_like(X_train_norm)], axis=-1)
X_test_in = np.concatenate([X_test_norm, np.ones_like(X_test_norm)], axis=-1)

# Training loop
model = TrajPilotPhysicalGRU(input_dim=16, hidden_dim=128, num_layers=2, future_steps=30, target_dim=2)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(torch.from_numpy(X_train_in).float(), torch.from_numpy(Y_train).float())
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

model.train()
for epoch in range(1, 4):
    total_loss = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(bx)
    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch}/3 - Training Loss: {avg_loss:.4f}")

# Save checkpoint
save_path = MODEL_DIR / "best_tgsim_physical_gru.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "model_config": {"input_dim": 16, "hidden_dim": 128, "num_layers": 2, "future_steps": 30, "target_dim": 2},
    "feature_names": ['vx_causal', 'vy_causal', 'speed_derived', 'acceleration_native', 'acceleration_causal_0p5s', 'turn_rate_valid', 'curvature_valid', 'jerk_causal'],
    "train_mean": train_mean,
    "train_std": train_std,
    "dataset": "TGSIM_I90_I94_MOVING"
}, save_path)
print(f"Saved new transferable model to: {save_path}")

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_preds = model(torch.from_numpy(X_test_in).float()).numpy()

gru_err = np.linalg.norm(test_preds - Y_test, axis=-1)
cv_err = np.linalg.norm(cv_test - Y_test, axis=-1)

gru_ade = float(gru_err.mean())
gru_fde = float(gru_err[:, -1].mean())
cv_ade = float(cv_err.mean())
cv_fde = float(cv_err[:, -1].mean())

print("=" * 70)
print("HELD-OUT EVALUATION ON TGSIM I-90/I-94 TEST PARTITION:")
print("=" * 70)
print(f"  Constant Velocity (CV) : ADE = {cv_ade:.4f} m | FDE = {cv_fde:.4f} m")
print(f"  Trained Physical GRU   : ADE = {gru_ade:.4f} m | FDE = {gru_fde:.4f} m")
improvement = ((cv_ade - gru_ade) / cv_ade) * 100
print(f"  Relative ADE Improvement over CV: {improvement:+.2f}%")
print("=" * 70)
