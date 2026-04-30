# NanoVit Attention Visualizer

Tiny Flask app to visualize attention maps from your NanoVit model on Imagenette images.

## Folder layout

```
attn_app/
├── app.py              # Flask server + attention extraction
├── nanovit.py          # YOUR file — copy it in here
├── templates/
│   └── index.html
└── (optional) imagenette2-320/val/...
└── (optional) checkpoints/nanovit_xxs.pth
```

## 1. Install deps (Windows 11, PowerShell)

From inside the `attn_app` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision flask einops timm pillow numpy
```

If PowerShell complains about activation scripts, run once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 2. Get Imagenette

Download `imagenette2-320.tgz` from https://github.com/fastai/imagenette and extract it
anywhere. You want the `val/` subfolder, which contains 10 class folders like
`n01440764`, `n02102040`, etc.

## 3. Edit `app.py`

At the top of `app.py`, set:

```python
IMAGENETTE_DIR = r"C:\path\to\imagenette2-320\val"
CKPT_PATH      = r"C:\path\to\your\checkpoint.pth"
MODEL_FN       = NanoVit_XXS    # or NanoVit_XS / NanoVit_S
```

Use raw strings (`r"..."`) for Windows paths so backslashes don't need escaping.

## 4. Copy `nanovit.py` into this folder

The app imports it directly: `from nanovit import NanoVit_XXS, ...`

## 5. Run

```powershell
python app.py
```

Open **http://localhost:5000** in your browser.

## Using the UI

- **Image** — pick any image from the 10 Imagenette classes
- **Layer** — `stage S · layer L · GxG` where G is the patch grid size:
  - stage 0 → 16×16 patches (early, fine spatial detail)
  - stage 1 → 8×8 patches
  - stage 2 → 8×8 patches (late, semantic)
- **Head** — average over heads, or pick a specific one (0–3)
- **Opacity slider** — adjust overlay strength

The grid shown is the per-patch importance (mean attention received), normalized to [0,1]
and rendered with a blue→green→yellow→red colormap.

## How it works

The app monkey-patches each `mvit_scale_dot_product.forward` to stash the post-softmax
attention tensor of shape `[B, P, H, N, N]` into a dict on every forward pass. To turn
this into a 2D map on the patch grid, the app:

1. averages over heads (or picks one)
2. averages over query positions → "attention received per patch"
3. averages over the `P` patch-position groups (MobileViT's interleaved patch layout)
4. reshapes the resulting `N` values to `G × G`

That gives one importance value per patch on the input grid, which is then upscaled to
the image and color-mapped.
