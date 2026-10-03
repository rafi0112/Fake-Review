"""
train.py

Stage 2 of the SigDINO-Qwen-GateX pipeline.

Trains ONLY the fusion head (projections + bidirectional cross-attention +
gated multimodal unit + classifier) on the tensors cached by
extract_features.py. The three big pretrained encoders are never loaded
here, so this script trains comfortably on a 6GB laptop GPU -- the
trainable model is a few million parameters, not a few hundred million.

Architecture (matches the "SigDINO-Qwen-GateX" design):
    T_seq      = proj_text(ModernBERT token sequence)              (B, L_t, D)
    V_img_seq  = concat(proj_siglip(SigLIP2 patches),
                         proj_dino(DINOv2 patches))                (B, N_img, D)
    H_text     = mean_pool( CrossAttn(Q=T_seq,     K=V=V_img_seq) )
    H_img      = mean_pool( CrossAttn(Q=V_img_seq, K=V=T_seq) )
    g          = sigmoid( W_g [H_text ; H_img ; s_align] )
    F          = g * H_text + (1-g) * H_img
    logits     = MLP( [F ; s_align] )

Usage:
    python train.py --features features --out runs/exp1
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CachedFeatureDataset(Dataset):
    def __init__(self, path: Path):
        cache = torch.load(path, map_location="cpu")
        self.text_tokens = cache["text_tokens"]  # (N, L_t, H_text)
        self.text_mask = cache["text_mask"]  # (N, L_t) bool
        self.siglip_patches = cache["siglip_patches"]  # (N, G*G, H_siglip)
        self.dino_patches = cache["dino_patches"]  # (N, G*G, H_dino)
        self.s_align = cache["s_align"]  # (N, 1)
        self.labels = cache["labels"]  # (N,)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return {
            "text_tokens": self.text_tokens[idx].float(),
            "text_mask": self.text_mask[idx],
            "siglip_patches": self.siglip_patches[idx].float(),
            "dino_patches": self.dino_patches[idx].float(),
            "s_align": self.s_align[idx].float(),
            "label": self.labels[idx],
        }


class CrossAttnGMU(nn.Module):
    def __init__(self, h_text, h_siglip, h_dino, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.proj_text = nn.Linear(h_text, d_model)
        self.proj_siglip = nn.Linear(h_siglip, d_model)
        self.proj_dino = nn.Linear(h_dino, d_model)

        self.text_to_img_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.img_to_text_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_text = nn.LayerNorm(d_model)
        self.norm_img = nn.LayerNorm(d_model)

        self.gate = nn.Linear(2 * d_model + 1, d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model + 1, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

        self.d_model = d_model

    def forward(self, text_tokens, text_mask, siglip_patches, dino_patches, s_align, ablation="none"):
        # text_tokens: (B, L_t, h_text)   text_mask: (B, L_t) bool, True = real token
        # siglip_patches: (B, N_s, h_siglip)   dino_patches: (B, N_d, h_dino)
        # s_align: (B, 1)
        # ablation: "none" (default) | "text_only" (zero the image stream and the
        #   cross-modal alignment score) | "image_only" (zero the text stream and
        #   the alignment score). Used to check whether fusion is actually adding
        #   value over either modality alone.
        if ablation == "text_only":
            siglip_patches = torch.zeros_like(siglip_patches)
            dino_patches = torch.zeros_like(dino_patches)
            s_align = torch.zeros_like(s_align)
        elif ablation == "image_only":
            text_tokens = torch.zeros_like(text_tokens)
            s_align = torch.zeros_like(s_align)
        elif ablation != "none":
            raise ValueError(f"unknown ablation mode: {ablation}")

        t_seq = self.proj_text(text_tokens)  # (B, L_t, D)
        s_seq = self.proj_siglip(siglip_patches)  # (B, N_s, D)
        d_seq = self.proj_dino(dino_patches)  # (B, N_d, D)
        v_img_seq = torch.cat([s_seq, d_seq], dim=1)  # (B, N_s+N_d, D)

        key_padding_mask = ~text_mask  # True = position to IGNORE, for nn.MultiheadAttention

        h_text_seq, _ = self.text_to_img_attn(query=t_seq, key=v_img_seq, value=v_img_seq)
        h_text_seq = self.norm_text(h_text_seq + t_seq)
        mask_f = text_mask.unsqueeze(-1).float()
        h_text = (h_text_seq * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)  # (B, D)

        h_img_seq, _ = self.img_to_text_attn(
            query=v_img_seq, key=t_seq, value=t_seq, key_padding_mask=key_padding_mask
        )
        h_img_seq = self.norm_img(h_img_seq + v_img_seq)
        h_img = h_img_seq.mean(dim=1)  # (B, D)

        gate_in = torch.cat([h_text, h_img, s_align], dim=1)  # (B, 2D+1)
        g = torch.sigmoid(self.gate(gate_in))  # (B, D)
        fused = g * h_text + (1 - g) * h_img  # (B, D)

        logits = self.classifier(torch.cat([fused, s_align], dim=1))  # (B, 2)
        return logits


@torch.no_grad()
def evaluate(model, loader, device, ablation="none"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n = 0
    for batch in loader:
        text_tokens = batch["text_tokens"].to(device)
        text_mask = batch["text_mask"].to(device)
        siglip_patches = batch["siglip_patches"].to(device)
        dino_patches = batch["dino_patches"].to(device)
        s_align = batch["s_align"].to(device)
        labels = batch["label"].to(device)

        logits = model(text_tokens, text_mask, siglip_patches, dino_patches, s_align, ablation=ablation)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="features")
    ap.add_argument("--out", type=str, default="runs/exp1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5, help="early stopping patience on val F1")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--ablation", type=str, default="none", choices=["none", "text_only", "image_only"],
        help="zero out one modality (+ the cross-modal alignment score) to check "
             "whether fusion adds value over a single modality alone",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    features_dir = Path(args.features)
    train_ds = CachedFeatureDataset(features_dir / "train.pt")
    val_ds = CachedFeatureDataset(features_dir / "val.pt")
    test_ds = CachedFeatureDataset(features_dir / "test.pt")
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    h_text = train_ds.text_tokens.shape[-1]
    h_siglip = train_ds.siglip_patches.shape[-1]
    h_dino = train_ds.dino_patches.shape[-1]
    print(f"encoder dims -> text:{h_text}  siglip:{h_siglip}  dino:{h_dino}")

    model = CrossAttnGMU(
        h_text, h_siglip, h_dino,
        d_model=args.d_model, num_heads=args.num_heads, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_f1 = -1.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            text_tokens = batch["text_tokens"].to(device)
            text_mask = batch["text_mask"].to(device)
            siglip_patches = batch["siglip_patches"].to(device)
            dino_patches = batch["dino_patches"].to(device)
            s_align = batch["s_align"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(text_tokens, text_mask, siglip_patches, dino_patches, s_align, ablation=args.ablation)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            n += labels.size(0)
            pbar.set_postfix(loss=running_loss / n)

        val_metrics = evaluate(model, val_loader, device, ablation=args.ablation)
        scheduler.step(val_metrics["f1"])
        print(
            f"epoch {epoch}: train_loss={running_loss / n:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}"
        )
        history.append({"epoch": epoch, "train_loss": running_loss / n, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            epochs_no_improve = 0
            torch.save(
                {"model_state": model.state_dict(), "args": vars(args),
                 "h_text": h_text, "h_siglip": h_siglip, "h_dino": h_dino},
                out_dir / "best_model.pt",
            )
            print(f"  -> new best (val_f1={best_f1:.4f}), checkpoint saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stopping: no val_f1 improvement in {args.patience} epochs")
                break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # final test evaluation with the best checkpoint
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, device, ablation=args.ablation)
    print("\n=== TEST SET (best val_f1 checkpoint) ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
