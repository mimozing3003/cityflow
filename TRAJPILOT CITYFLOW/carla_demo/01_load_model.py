"""
TrajPilot CityFlow - Stage 01: Exact Frozen GRU Architecture and Checkpoint Loader
Preserves the verified 227,132-parameter TrajPilot GRU model without modification.
"""

from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "models" / "gru_baseline" / "best_gru_baseline.pt"

EXPECTED_PARAMS = 227132
INPUT_DIM = 32
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.10
FUTURE_STEPS = 30
TARGET_DIM = 2


class TrajPilotGRU(nn.Module):
    """
    Exact frozen TrajPilot GRU trajectory prediction model.
    Input:  (batch, 30, 32) -> 16 normalized causal features + 16 missingness indicators.
    Output: (batch, 30, 2)  -> 30 future (dx, dy) trajectory coordinates at 10 FPS (3.0s horizon).
    Total trainable parameters: 227,132.
    """
    def __init__(
        self,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        future_steps=FUTURE_STEPS,
        target_dim=TARGET_DIM,
        dropout=DROPOUT
    ):
        super().__init__()
        self.future_steps = future_steps
        self.target_dim = target_dim

        # Input projection: 32 -> 128
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 2-layer causal recurrent encoder
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Trajectory decoder: 128 -> 128 -> 60 (30 x 2)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, future_steps * target_dim)
        )

    def forward(self, x):
        # x: (batch, 30, 32)
        x_proj = self.input_projection(x)
        _, h_n = self.encoder(x_proj)
        # h_n[-1] is top-layer final hidden state: (batch, 128)
        y = self.decoder(h_n[-1])
        return y.view(-1, self.future_steps, self.target_dim)


def load_frozen_model(device=None, checkpoint_path=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH

    print("=" * 78)
    print("TRAJPILOT CITYFLOW | 01 - FROZEN GRU MODEL LOADER")
    print("=" * 78)
    print(f"Checkpoint Path    : {checkpoint_path}")
    print(f"Target Device      : {device}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model = TrajPilotGRU(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        future_steps=FUTURE_STEPS,
        target_dim=TARGET_DIM,
        dropout=DROPOUT
    ).to(device)

    # Count parameters before loading
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters   : {param_count:,} (Expected: {EXPECTED_PARAMS:,})")
    if param_count != EXPECTED_PARAMS:
        raise ValueError(f"Parameter count mismatch: got {param_count}, expected {EXPECTED_PARAMS}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"Strict Load        : PASS (All keys matched exactly)")
    print(f"Inference Mode     : EVAL (Dropout disabled)")
    print("=" * 78)
    return model, device


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, d = load_frozen_model(dev)
    # Quick dummy forward test
    dummy = torch.randn(1, 30, 32, device=d)
    with torch.no_grad():
        out = m(dummy)
    print(f"Dummy Forward Pass : Input {tuple(dummy.shape)} -> Output {tuple(out.shape)} | PASS")
