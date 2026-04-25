"""PatchTST Transformer model implementation for Poseidon.

Uses a Patch Time Series Transformer (encoder-only) with patching on
time-series input for 3-class directional prediction (long/short/hold).
Falls back to a stub if torch is not installed.
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.ml.base import BaseModel
from poseidon.ml.registry import register_model

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Label mapping (identical to xgboost_model.py)
# ---------------------------------------------------------------------------
LABEL_MAP = {0: "hold", 1: "long", 2: "short"}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Default feature set (matching FeatureEngine defaults)
DEFAULT_FEATURES = [
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_60",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "bb_upper_20",
    "bb_middle_20",
    "bb_lower_20",
    "atr_14",
    "return_1d",
    "log_return_1d",
    "std_vol_20",
    "volume_sma_20",
    "volume_ratio_20",
    "obv",
]


# ---------------------------------------------------------------------------
# TimeSeriesDataset
# ---------------------------------------------------------------------------
if _HAS_TORCH:

    class TimeSeriesDataset(Dataset):
        """Sliding-window dataset that converts wide feature arrays into
        ``[lookback_window, n_features]`` tensors for the Transformer.

        Per-feature z-score normalization is computed from the *training*
        features and stored as ``_mean`` / ``_std`` so the same
        normalization can be applied at prediction time.
        """

        def __init__(
            self,
            features: np.ndarray,
            targets: np.ndarray,
            lookback_window: int = 60,
        ) -> None:
            super().__init__()
            self._features = features.astype(np.float32)
            self._targets = targets.astype(np.int64)
            self._lookback_window = lookback_window

            # Per-feature standardization (z-score)
            self._mean = self._features.mean(axis=0)  # shape [n_features]
            self._std = self._features.std(axis=0)  # shape [n_features]
            # Avoid division by zero
            self._std[self._std == 0] = 1.0
            self._features = (self._features - self._mean) / self._std

        def __len__(self) -> int:
            return len(self._features) - self._lookback_window

        def __getitem__(self, idx: int):
            x = self._features[idx : idx + self._lookback_window]  # [lookback, n_feat]
            y = self._targets[idx + self._lookback_window]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)


# ---------------------------------------------------------------------------
# PatchTST nn.Module
# ---------------------------------------------------------------------------
if _HAS_TORCH:

    class PatchTST(nn.Module):
        """Encoder-only Transformer with patching for time-series classification.

        Input:  ``[batch, lookback_window, n_features]``
        Output: ``[batch, num_classes]`` logits
        """

        def __init__(
            self,
            n_features: int,
            lookback_window: int = 60,
            patch_length: int = 16,
            stride: int = 8,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
            num_classes: int = 3,
        ) -> None:
            super().__init__()
            self.patch_length = patch_length
            self.stride = stride
            self.n_features = n_features
            self.lookback_window = lookback_window

            # Number of patches
            self.num_patches = (lookback_window - patch_length) // stride + 1
            patch_dim = patch_length * n_features

            # Patch embedding (linear projection)
            self.patch_proj = nn.Linear(patch_dim, d_model)

            # Learnable positional encoding
            self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Classification head (after mean-pooling across patches)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Forward pass.

            Args:
                x: ``[batch, lookback_window, n_features]``

            Returns:
                logits of shape ``[batch, num_classes]``
            """
            batch_size = x.size(0)

            # --- Create patches ---
            # x: [B, L, F] -> patches: [B, num_patches, patch_length * F]
            patches = []
            for i in range(self.num_patches):
                start = i * self.stride
                end = start + self.patch_length
                patch = x[:, start:end, :]  # [B, patch_length, F]
                patches.append(patch.reshape(batch_size, -1))  # [B, patch_length * F]
            patches = torch.stack(patches, dim=1)  # [B, num_patches, patch_dim]

            # --- Project patches to d_model ---
            z = self.patch_proj(patches)  # [B, num_patches, d_model]
            z = z + self.pos_embed  # add positional encoding

            # --- Transformer encoder ---
            z = self.encoder(z)  # [B, num_patches, d_model]

            # --- Mean pooling across patch dimension ---
            z = z.mean(dim=1)  # [B, d_model]

            # --- Classification head ---
            logits = self.head(z)  # [B, num_classes]
            return logits


# ---------------------------------------------------------------------------
# TransformerModel (BaseModel implementation)
# ---------------------------------------------------------------------------
@register_model
class TransformerModel(BaseModel):
    """PatchTST Transformer for directional prediction (long/short/hold)."""

    name = "transformer"
    description = "PatchTST Transformer for directional prediction"
    supports_live = True

    def __init__(self) -> None:
        self._model: PatchTST | None = None  # type: ignore[name-defined]
        self._feature_list: list[str] = list(DEFAULT_FEATURES)
        self._device: str = "cpu"
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._lookback_window: int = 60
        self._model_params: dict = {}

        if _HAS_TORCH and torch.cuda.is_available():
            self._device = "cuda"

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------
    def train(self, features: pd.DataFrame, targets: pd.Series, params: dict) -> dict:
        if not _HAS_TORCH:
            raise RuntimeError("torch is not installed. Install with: pip install torch")

        merged = {**self.get_default_params(), **params}

        # Feature selection
        feature_cols = [c for c in self._feature_list if c in features.columns]
        self._feature_list = feature_cols

        # Drop NaN rows and align targets
        X_df = features[feature_cols].dropna()
        y_s = targets.loc[X_df.index]

        X_np = X_df.values.astype(np.float32)
        y_np = y_s.values.astype(np.int64)

        lookback_window = int(merged["lookback_window"])
        self._lookback_window = lookback_window

        # --- Validation split (last 20%) ---
        n_total = len(X_np) - lookback_window
        n_val = max(1, int(n_total * 0.2))
        n_train = n_total - n_val

        # Training set
        train_ds = TimeSeriesDataset(
            X_np[: n_train + lookback_window],
            y_np[: n_train + lookback_window],
            lookback_window=lookback_window,
        )
        # Validation set
        val_ds = TimeSeriesDataset(
            X_np[n_train:],
            y_np[n_train:],
            lookback_window=lookback_window,
        )

        # Store normalization params from training set
        self._mean = train_ds._mean.copy()
        self._std = train_ds._std.copy()

        batch_size = int(merged["batch_size"])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Build PatchTST model
        n_features = len(feature_cols)
        model_params = {
            "n_features": n_features,
            "lookback_window": lookback_window,
            "patch_length": int(merged["patch_length"]),
            "stride": int(merged["stride"]),
            "d_model": int(merged["d_model"]),
            "nhead": int(merged["nhead"]),
            "num_layers": int(merged["num_layers"]),
            "dim_feedforward": int(merged["dim_feedforward"]),
            "dropout": float(merged["dropout"]),
        }
        self._model_params = model_params
        model = PatchTST(**model_params).to(self._device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(merged["lr"]),
            weight_decay=float(merged["weight_decay"]),
        )

        # Class weight balancing — inverse frequency so minority classes
        # (long/short) get higher loss weight than the dominant class (hold).
        train_targets = y_np[lookback_window : n_train + lookback_window]
        class_counts = np.bincount(train_targets, minlength=3).astype(np.float32)
        class_counts[class_counts == 0] = 1.0  # avoid division by zero
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        logger.info(
            "Class weights: hold=%.3f, long=%.3f, short=%.3f (counts: %s)",
            class_weights[0],
            class_weights[1],
            class_weights[2],
            class_counts.astype(int).tolist(),
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(self._device),
        )

        use_amp = self._device == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        epochs = int(merged["epochs"])
        patience = int(merged["patience"])
        best_val_loss = math.inf
        patience_counter = 0
        epochs_trained = 0
        final_loss = 0.0

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = model(x_batch)
                        loss = criterion(logits, y_batch)
                    scaler.scale(loss).backward()  # type: ignore[union-attr]
                    scaler.step(optimizer)  # type: ignore[union-attr]
                    scaler.update()  # type: ignore[union-attr]
                else:
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            final_loss = avg_train_loss
            epochs_trained = epoch + 1

            # --- Validation loss ---
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(self._device)
                    y_val = y_val.to(self._device)
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            v_logits = model(x_val)
                            v_loss = criterion(v_logits, y_val)
                    else:
                        v_logits = model(x_val)
                        v_loss = criterion(v_logits, y_val)
                    val_loss += v_loss.item()
                    val_batches += 1
            avg_val_loss = val_loss / max(val_batches, 1)
            model.train()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                    epoch + 1,
                    epochs,
                    avg_train_loss,
                    avg_val_loss,
                )

            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        # --- Training accuracy ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                logits = model(x_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        accuracy = correct / max(total, 1)

        self._model = model
        self._model.eval()

        return {
            "accuracy": accuracy,
            "n_samples": len(X_df),
            "n_features": n_features,
            "epochs_trained": epochs_trained,
            "final_loss": final_loss,
        }

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")
        if not _HAS_TORCH:
            raise RuntimeError("torch is not installed")

        feature_cols = [c for c in self._feature_list if c in features.columns]
        X_df = features[feature_cols].dropna()
        X_np = X_df.values.astype(np.float32)

        if len(X_np) <= self._lookback_window:
            # Not enough data for even one window — return all "hold"
            return pd.DataFrame(
                {"prediction": "hold", "confidence": 0.0},
                index=features.index,
            )

        # Normalize using stored training statistics
        mean = self._mean if self._mean is not None else np.zeros(X_np.shape[1])
        std = self._std if self._std is not None else np.ones(X_np.shape[1])
        X_norm = (X_np - mean) / std

        # Build sliding windows
        windows = []
        for i in range(len(X_norm) - self._lookback_window):
            windows.append(X_norm[i : i + self._lookback_window])
        windows_arr = np.stack(windows)  # [n_windows, lookback, n_features]

        # Run inference
        self._model.eval()
        device = self._device
        use_amp = device == "cuda"

        with torch.no_grad():
            x_tensor = torch.tensor(windows_arr, dtype=torch.float32).to(device)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self._model(x_tensor)
            else:
                logits = self._model(x_tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()

        pred_indices = np.argmax(proba, axis=1)
        confidences = np.max(proba, axis=1)
        predictions = [LABEL_MAP.get(int(i), "hold") for i in pred_indices]

        # The predictions correspond to X_df.index[lookback_window:]
        valid_index = X_df.index[self._lookback_window :]
        result = pd.DataFrame(
            {"prediction": predictions, "confidence": confidences},
            index=valid_index,
        )

        # Reindex to original, fill missing (first lookback rows + NaN rows) with "hold"
        result = result.reindex(features.index)
        result["prediction"] = result["prediction"].fillna("hold")
        result["confidence"] = result["confidence"].fillna(0.0)
        return result

    # ------------------------------------------------------------------
    # validate
    # ------------------------------------------------------------------
    def validate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")

        pred_df = self.predict(features)

        # Align targets with predictions
        common_idx = pred_df.index.intersection(targets.index)
        preds_str = pred_df.loc[common_idx, "prediction"]
        y = targets.loc[common_idx]

        # Convert string predictions back to numeric for comparison
        pred_numeric = preds_str.map(REVERSE_LABEL_MAP).fillna(0).astype(int)
        accuracy = float(np.mean(pred_numeric.values == y.values))

        result: dict[str, float] = {
            "accuracy": accuracy,
            "n_samples": float(len(common_idx)),
        }

        # Per-class accuracy
        for label_idx, label_name in LABEL_MAP.items():
            mask = y == label_idx
            if mask.sum() > 0:
                result[f"accuracy_{label_name}"] = float(np.mean(pred_numeric.values[mask.values] == label_idx))

        return result

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("torch is not installed")
        if self._model is None:
            raise RuntimeError("No model to save")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self._model.state_dict(), path / "model.pt")
        (path / "features.json").write_text(json.dumps(self._feature_list))

        metadata = {
            **self._model_params,
            "lookback_window": self._lookback_window,
            "mean": self._mean.tolist() if self._mean is not None else None,
            "std": self._std.tolist() if self._std is not None else None,
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2))

        logger.info("Saved Transformer model to %s", path)

    @classmethod
    def load(cls, path: Path) -> "TransformerModel":
        if not _HAS_TORCH:
            raise RuntimeError("torch is not installed")

        path = Path(path)
        instance = cls()

        # Load feature list
        features_file = path / "features.json"
        if features_file.exists():
            instance._feature_list = json.loads(features_file.read_text())

        # Load metadata
        metadata_file = path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"metadata.json not found in {path}")
        metadata = json.loads(metadata_file.read_text())

        # Reconstruct model params
        model_params = {
            "n_features": metadata.get("n_features", len(instance._feature_list)),
            "lookback_window": metadata.get("lookback_window", 60),
            "patch_length": metadata.get("patch_length", 16),
            "stride": metadata.get("stride", 8),
            "d_model": metadata.get("d_model", 64),
            "nhead": metadata.get("nhead", 4),
            "num_layers": metadata.get("num_layers", 2),
            "dim_feedforward": metadata.get("dim_feedforward", 128),
            "dropout": metadata.get("dropout", 0.1),
        }
        instance._model_params = model_params
        instance._lookback_window = metadata.get("lookback_window", 60)

        # Restore normalization stats
        if metadata.get("mean") is not None:
            instance._mean = np.array(metadata["mean"], dtype=np.float32)
        if metadata.get("std") is not None:
            instance._std = np.array(metadata["std"], dtype=np.float32)

        # Build model and load weights
        model = PatchTST(**model_params)
        device = instance._device
        state_dict = torch.load(path / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        instance._model = model

        logger.info("Loaded Transformer model from %s", path)
        return instance

    # ------------------------------------------------------------------
    # params / features
    # ------------------------------------------------------------------
    def get_default_params(self) -> dict:
        return {
            "lookback_window": 60,
            "patch_length": 16,
            "stride": 8,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "batch_size": 32,
            "epochs": 50,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "patience": 10,
        }

    def get_feature_list(self) -> list[str]:
        return list(self._feature_list)
