"""
extract_features.py

Stage 1 of the SigDINO-Qwen-GateX pipeline.

Runs the three FROZEN pretrained encoders once over the dataset and caches
their outputs to disk, so that stage 2 (train.py) never has to load the big
models again -- it only trains a small fusion head on cached tensors. This
is what makes the whole pipeline fit comfortably on a 6GB laptop GPU: the
three encoders run one at a time, with no gradients, and nothing but the
small trainable fusion head is ever backpropped through.

Text encoder:   answerdotai/ModernBERT-base   (token-level sequence output)
Image encoder A: google/siglip2-base-patch16-224  (patch tokens + aligned
                 text/image embeddings used for the cosine alignment score)
Image encoder B: facebook/dinov2-base          (patch tokens, fine-grained)

Each image's patch-token grid is average-pooled down to a small GxG grid
(default 4x4 = 16 tokens per stream) before caching, to keep the on-disk
cache and the cross-attention sequence length manageable on laptop hardware.
Increase --pool-grid later (e.g. for a final paper run on a bigger machine)
to keep more spatial detail.

Usage:
    python extract_features.py --data data --out features --split train
    python extract_features.py --data data --out features --split val
    python extract_features.py --data data --out features --split test
"""

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer

TEXT_MODEL = "answerdotai/ModernBERT-base"
SIGLIP_MODEL = "google/siglip2-base-patch16-224"
DINO_MODEL = "facebook/dinov2-base"

MAX_TEXT_LEN = 128
MAX_SIGLIP_TEXT_LEN = 64


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: no CUDA device found, falling back to CPU (will be slow).")
    return torch.device("cpu")


def load_split(data_dir: Path, split: str, id_col: str, text_col: str, label_col: str):
    csv_path = data_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found -- run download_dataset.py first")
    df = pd.read_csv(csv_path)
    for col in (id_col, text_col, label_col):
        if col not in df.columns:
            raise KeyError(
                f"column '{col}' not found in {csv_path.name}. "
                f"Available columns: {list(df.columns)}. "
                f"Pass the correct name via --id-col/--text-col/--label-col."
            )
    images_dir = data_dir / "images"
    records = []
    for _, row in df.iterrows():
        img_path = images_dir / f"{row[id_col]}.jpg"
        records.append(
            {
                "id": row[id_col],
                "text": str(row[text_col]),
                "label": int(row[label_col]),
                "image_path": img_path,
            }
        )
    return records


def load_image(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"WARNING: could not load image {path} ({e}); using a blank gray image instead.")
        return Image.new("RGB", (224, 224), color=(128, 128, 128))


def get_vision_tower(model):
    """Return the SigLIP2 vision submodule that yields patch-level tokens.

    This is the one spot in this script I could not verify against a live
    checkpoint (no model download in the environment I wrote this in), so it
    tries the standard CLIP/SigLIP naming first and fails loudly with a fix
    if the checkpoint uses something else.
    """
    for attr in ("vision_model", "vision_tower", "visual"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError(
        "Could not find the SigLIP2 vision submodule on this model (tried "
        "'vision_model', 'vision_tower', 'visual'). Run "
        "`python -c \"from transformers import AutoModel; "
        f"m=AutoModel.from_pretrained('{SIGLIP_MODEL}'); print(list(dict(m.named_children()).keys()))\"` "
        "to see the real submodule names, then edit get_vision_tower() in "
        "this file to match (one line)."
    )


def pool_patch_tokens(tokens: torch.Tensor, grid: int) -> torch.Tensor:
    """tokens: (B, N, H) patch tokens from a square ViT grid -> (B, grid*grid, H)."""
    b, n, h = tokens.shape
    side = int(round(n**0.5))
    assert side * side == n, f"expected a square patch grid, got N={n}"
    x = tokens.reshape(b, side, side, h).permute(0, 3, 1, 2)  # (B,H,side,side)
    x = F.adaptive_avg_pool2d(x, output_size=(grid, grid))  # (B,H,grid,grid)
    x = x.permute(0, 2, 3, 1).reshape(b, grid * grid, h)  # (B,grid*grid,H)
    return x


@torch.no_grad()
def extract_text_stream(records, device, batch_size):
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()

    all_tokens, all_masks = [], []
    for i in tqdm(range(0, len(records), batch_size), desc="ModernBERT text"):
        batch = records[i : i + batch_size]
        texts = [r["text"] for r in batch]
        enc = tok(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(**enc)
        all_tokens.append(out.last_hidden_state.detach().to("cpu", dtype=torch.float16))
        all_masks.append(enc["attention_mask"].detach().to("cpu", dtype=torch.bool))

    del model, tok
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return torch.cat(all_tokens, dim=0), torch.cat(all_masks, dim=0)


@torch.no_grad()
def extract_siglip_stream(records, device, batch_size, pool_grid):
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model = AutoModel.from_pretrained(SIGLIP_MODEL).to(device).eval()

    all_patches, all_img_emb, all_txt_emb = [], [], []
    for i in tqdm(range(0, len(records), batch_size), desc="SigLIP2 image+text"):
        batch = records[i : i + batch_size]
        images = [load_image(r["image_path"]) for r in batch]
        texts = [r["text"] for r in batch]

        img_inputs = processor(images=images, return_tensors="pt").to(device)
        txt_inputs = processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_SIGLIP_TEXT_LEN,
            return_tensors="pt",
        ).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            vision_out = get_vision_tower(model)(pixel_values=img_inputs["pixel_values"])
            patch_tokens = vision_out.last_hidden_state  # (B, N, H) patch tokens, no CLS
            # NOTE: in this transformers version, get_image_features/get_text_features
            # return a BaseModelOutputWithPooling (not a bare tensor) -- use .pooler_output.
            img_emb = model.get_image_features(pixel_values=img_inputs["pixel_values"]).pooler_output
            txt_emb = model.get_text_features(
                input_ids=txt_inputs["input_ids"],
                attention_mask=txt_inputs.get("attention_mask"),
            ).pooler_output

        pooled = pool_patch_tokens(patch_tokens.float(), pool_grid)
        all_patches.append(pooled.detach().to("cpu", dtype=torch.float16))
        all_img_emb.append(F.normalize(img_emb.float(), dim=-1).detach().to("cpu", dtype=torch.float16))
        all_txt_emb.append(F.normalize(txt_emb.float(), dim=-1).detach().to("cpu", dtype=torch.float16))

    del model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return torch.cat(all_patches, dim=0), torch.cat(all_img_emb, dim=0), torch.cat(all_txt_emb, dim=0)


@torch.no_grad()
def extract_dino_stream(records, device, batch_size, pool_grid):
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
    model = AutoModel.from_pretrained(DINO_MODEL).to(device).eval()

    all_patches = []
    for i in tqdm(range(0, len(records), batch_size), desc="DINOv2 image"):
        batch = records[i : i + batch_size]
        images = [load_image(r["image_path"]) for r in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(**inputs)
        patch_tokens = out.last_hidden_state[:, 1:, :]  # drop CLS token at position 0

        pooled = pool_patch_tokens(patch_tokens.float(), pool_grid)
        all_patches.append(pooled.detach().to("cpu", dtype=torch.float16))

    del model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return torch.cat(all_patches, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data", help="dir with {split}.csv and images/")
    ap.add_argument("--out", type=str, default="features", help="output dir for cached tensors")
    ap.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    ap.add_argument("--id-col", type=str, default="id")
    ap.add_argument("--text-col", type=str, default="text")
    ap.add_argument("--label-col", type=str, default="label")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--pool-grid", type=int, default=4, help="GxG pooled visual tokens per image stream")
    ap.add_argument("--max-samples", type=int, default=None, help="debug: cap number of samples")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    records = load_split(data_dir, args.split, args.id_col, args.text_col, args.label_col)
    if args.max_samples:
        records = records[: args.max_samples]
    print(f"{args.split}: {len(records)} samples")

    # run one encoder at a time -- each block frees its VRAM before the next loads
    text_tokens, text_mask = extract_text_stream(records, device, args.batch_size)
    siglip_patches, siglip_img_emb, siglip_txt_emb = extract_siglip_stream(
        records, device, args.batch_size, args.pool_grid
    )
    dino_patches = extract_dino_stream(records, device, args.batch_size, args.pool_grid)

    labels = torch.tensor([r["label"] for r in records], dtype=torch.long)
    s_align = (siglip_img_emb.float() * siglip_txt_emb.float()).sum(dim=-1, keepdim=True).half()

    cache = {
        "text_tokens": text_tokens,  # (N, L_t, H_text)
        "text_mask": text_mask,  # (N, L_t) bool
        "siglip_patches": siglip_patches,  # (N, G*G, H_siglip)
        "dino_patches": dino_patches,  # (N, G*G, H_dino)
        "s_align": s_align,  # (N, 1) cosine alignment score
        "labels": labels,  # (N,)
    }

    out_path = out_dir / f"{args.split}.pt"
    torch.save(cache, out_path)
    print(f"Saved {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")
    for k, v in cache.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main()
