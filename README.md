# SigDINO-Qwen-GateX -- training on a 6GB laptop GPU

This is a two-stage pipeline, split specifically so it fits an RTX 3050
Laptop GPU (6GB VRAM):

- **Stage 1 (`extract_features.py`)** runs three frozen pretrained encoders
  (ModernBERT for text, SigLIP2 + DINOv2 for images) once over the dataset,
  with no gradients, one model at a time, and caches their outputs to disk.
- **Stage 2 (`train.py`)** loads only those cached tensors and trains a
  small fusion head (cross-attention + gated multimodal unit + classifier,
  a few million parameters). The big encoders are never loaded in stage 2.

This is why it fits in 6GB: at no point are three large models and a
backward pass all in VRAM together. The heaviest single thing you ever load
is one encoder (largest is SigLIP2-base at ~800MB in fp32), used purely for
inference.

## 0. One honest note before you run this

I wrote and shape-checked this code carefully, but **I could not execute it
against the real models in my own environment** (no GPU and no access to
Hugging Face from where I generated this). Before trusting a multi-hour run,
do the smoke test in step 3 first -- it runs the exact same code path on 50
samples in a couple of minutes, so any real bug surfaces immediately instead
of after hours of extraction.

## 1. Prerequisites

Check your GPU is visible and which CUDA version your driver supports:

```bash
nvidia-smi
```

Look at the "CUDA Version" in the top-right of the output (e.g. `12.4`) --
that's the max your driver supports, not what you must install; CUDA 12.1
wheels work on any driver that reports 12.1 or higher.

You need Python 3.10 or 3.11.

## 2. Environment setup

```bash
# create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd.exe):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# install the CUDA build of PyTorch FIRST, and separately from everything
# else -- the plain `pip install torch` can resolve to a CPU-only build
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

# verify it sees your GPU before installing anything else
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You should see something like `2.x.x+cu121 True NVIDIA GeForce RTX 3050 Laptop GPU`.
If `cuda.is_available()` is `False`, stop here and fix that first (usually an
outdated NVIDIA driver) -- nothing below will use your GPU until this prints `True`.

Then install the rest:

```bash
python -m pip install -r requirements.txt
```

## 3. Get the dataset

```bash
python download_dataset.py --out data
```

This pulls the AiGen-FoodReview dataset (Gambetti & Han, ICWSM 2024; MIT
license) from Zenodo: ~1.3GB of images plus three CSVs (train/val/test,
20,144 review-image pairs total, 60/20/20 split). It can take a while on a
slow connection -- the `images.zip` download bar will tell you where it's at.

**Before extracting features on the full dataset**, open `data/train.csv`
once and confirm the column names. The scripts default to `id`, `text`,
`label` -- if yours differ, pass `--id-col / --text-col / --label-col` to
every command below.

## 4. Smoke test (do this first -- takes ~2 minutes)

```bash
python extract_features.py --data data --out features_smoke --split train --max-samples 50 --batch-size 8
python extract_features.py --data data --out features_smoke --split val   --max-samples 50 --batch-size 8
python extract_features.py --data data --out features_smoke --split test  --max-samples 50 --batch-size 8
python train.py --features features_smoke --out runs/smoke --epochs 2 --batch-size 8
```

This downloads the three pretrained models (a few GB total, one-time) and
runs the entire pipeline end to end on 50 samples. If this finishes without
errors, the real run will too -- any shape mismatch, missing column, or
corrupted image will show up here in minutes, not hours. Watch the printed
tensor shapes from `extract_features.py`; they should look like:

```
text_tokens: (50, 128, 768) torch.float16
siglip_patches: (50, 16, 768) torch.float16
dino_patches: (50, 16, 768) torch.float16
s_align: (50, 1) torch.float16
labels: (50,) torch.int64
```

(Exact hidden sizes can differ slightly by checkpoint version -- that's
fine, the training script reads them dynamically.)

## 5. Full feature extraction

```bash
python extract_features.py --data data --out features --split train
python extract_features.py --data data --out features --split val
python extract_features.py --data data --out features --split test
```

Run these one after another (not in parallel -- they'd fight over the same
6GB of VRAM). Expect roughly 30-60 minutes total on an RTX 3050 laptop for
all three splits combined, mostly dependent on disk/image-loading speed.
The cached `.pt` files will land in `features/` and together should be in
the 2-4GB range on disk (cached in fp16, with patch tokens pooled down to a
4x4 grid per image stream -- see `--pool-grid` below if you want more detail
later).

If this step is killed by the OS for using too much RAM (not VRAM) on a
machine with 8GB system RAM, re-run with a smaller `--batch-size` (e.g. 16)
-- it won't speed up a slow disk, but it keeps peak memory lower.

## 6. Train the fusion head

```bash
python train.py --features features --out runs/exp1
```

This is the fast part. With everything cached, training is small-model,
tabular-style training -- expect a few minutes per epoch on an RTX 3050,
not hours. Default is up to 20 epochs with early stopping (patience 5) on
validation F1. Outputs land in `runs/exp1/`:

- `best_model.pt` -- checkpoint with the best validation F1
- `history.json` -- per-epoch train/val metrics
- `test_metrics.json` -- final accuracy / precision / recall / F1 on the
  held-out test split, evaluated with the best checkpoint

## VRAM budget, if you hit OOM

Stage 1 (`extract_features.py`) loads one encoder at a time and clears CUDA
cache between them, so OOM there is unlikely on 6GB -- if it happens, lower
`--batch-size` (try 16, then 8).

Stage 2 (`train.py`) trains only the small fusion head on cached features,
so it has a lot of headroom on 6GB -- the default `--batch-size 64` should
be very safe, and you can likely raise it (128, 256) for faster epochs.
If you do hit OOM here, lower `--batch-size` first before anything else.

## What's a simplification here, worth noting in your writeup

- **Text encoder**: this code uses **ModernBERT-base** (not Qwen3-Embedding)
  because ModernBERT naturally outputs a full token sequence, which is what
  the bidirectional cross-attention needs. Qwen3-Embedding is an
  embedding-specialized model that mainly exposes a single pooled vector;
  using it would collapse the text side of cross-attention to a single
  query token. If you want to try Qwen3-Embedding as a stronger pooled-only
  text representation (e.g. as an additional ablation), it's a separate,
  simpler branch to add rather than a drop-in swap for the current
  cross-attention design -- worth flagging as future work rather than
  silently claiming both at once.
- **Patch-grid pooling**: image patch tokens from SigLIP2 and DINOv2 are
  average-pooled to a small `G x G` grid (default 4x4 = 16 tokens per
  stream) before being cached, to keep cross-attention and disk usage
  laptop-sized. This trades some spatial resolution for practicality. Once
  the pipeline works end to end, you can re-run `extract_features.py` with
  `--pool-grid 7` (or higher) to keep more detail for your final reported
  numbers, VRAM/disk permitting.
- **Encoders stay frozen** -- no LoRA fine-tuning of the encoders is
  implemented in this version (the architecture doc proposed it as a later
  refinement). Freezing is also the right first baseline to report before
  adding that complexity.

## Next steps once this runs cleanly

1. Confirm the smoke test end to end.
2. Run full extraction + training, record `test_metrics.json`.
3. Compare against a text-only baseline (zero out the image stream) and an
   image-only baseline, the way all four papers in your literature review
   do -- this is what lets you claim the multimodal fusion itself helps,
   not just that the backbones are newer.
4. Only then consider LoRA-adapting the encoders or raising `--pool-grid`.
