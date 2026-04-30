"""
NanoVit Attention Visualizer - simple Flask app.

Run:
    python app.py

Then open http://localhost:5000 in your browser.

Configure CHECKPOINTS below to point at the .pth file for each NanoVit variant
you want available in the model dropdown.
"""

import os
import io
import sys
import uuid
import base64
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from flask import Flask, render_template, request, jsonify

from nanovit import NanoVit_XXS, NanoVit_XS, NanoVit_S, load_backbone_from_checkpoint_into_fcn

# ============================================================
# CONFIG --- EDIT PATHS BELOW
# ============================================================

DATA_ROOT = Path("./data")
IMAGENETTE_DIR = DATA_ROOT / "imagenette2-320" / "val"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

# Where uploaded images get saved (kept across runs).
UPLOADS_DIR = DATA_ROOT / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Model registry: name shown in UI -> (constructor, checkpoint path)
# Set checkpoint to None (or a non-existent path) to use random weights.
# All NanoVit factories build a 1000-class head by default.
CHECKPOINTS = {
    "NanoVit_XXS": (NanoVit_XXS, r"weights/nanovit_xxs.pth"),
    "NanoVit_XS":  (NanoVit_XS,  r"weights/nanovit_xs.pth"),
    "NanoVit_S":   (NanoVit_S,   r"weights/nanovit_s.pth"),
}

DEFAULT_MODEL = "NanoVit_XXS"

IMG_SIZE = 256

IMAGENETTE_CLASSES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

# ============================================================
# IMAGENETTE AUTO-DOWNLOAD
# ============================================================

def ensure_imagenette():
    if IMAGENETTE_DIR.exists() and any(IMAGENETTE_DIR.iterdir()):
        print(f"[init] imagenette already present at {IMAGENETTE_DIR}")
        return

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    archive = DATA_ROOT / "imagenette2-320.tgz"

    if not archive.exists():
        print(f"[init] downloading imagenette2-320 (~325 MB) from {IMAGENETTE_URL}")
        def _report(block_num, block_size, total_size):
            done = block_num * block_size
            if total_size > 0:
                pct = min(100.0, 100.0 * done / total_size)
                bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
                sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  ({done/1e6:.1f}/{total_size/1e6:.1f} MB)")
                sys.stdout.flush()
        try:
            urllib.request.urlretrieve(IMAGENETTE_URL, archive, reporthook=_report)
            sys.stdout.write("\n")
        except Exception as e:
            if archive.exists():
                archive.unlink()
            raise RuntimeError(f"failed to download imagenette: {e}") from e

    print(f"[init] extracting {archive} ...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(DATA_ROOT)
    print(f"[init] extracted to {DATA_ROOT / 'imagenette2-320'}")


ensure_imagenette()


# ============================================================
# IMAGENET-1K LABELS
# ============================================================

IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
IMAGENET_LABELS_PATH = DATA_ROOT / "imagenet_classes.txt"

def ensure_imagenet_labels():
    if IMAGENET_LABELS_PATH.exists():
        return
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[init] downloading ImageNet-1k labels from {IMAGENET_LABELS_URL}")
    try:
        urllib.request.urlretrieve(IMAGENET_LABELS_URL, IMAGENET_LABELS_PATH)
    except Exception as e:
        print(f"[init] WARNING: could not download ImageNet labels: {e}")
        print(f"[init] class names will fall back to indices.")

ensure_imagenet_labels()

if IMAGENET_LABELS_PATH.exists():
    with open(IMAGENET_LABELS_PATH, "r", encoding="utf-8") as f:
        IMAGENET_LABELS = [line.strip() for line in f if line.strip()]
    print(f"[init] loaded {len(IMAGENET_LABELS)} ImageNet labels")
else:
    IMAGENET_LABELS = [str(i) for i in range(1000)]

# Imagenette WNID -> ImageNet-1k class index (alphabetical-by-WNID order,
# which is the standard PyTorch / Keras label ordering)
IMAGENETTE_WNID_TO_IMAGENET_IDX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


# ============================================================
# MODEL + ATTENTION HOOKS
# ============================================================
#
# Architecture recap for NanoVit (MobileViTv4):
#   - 3 mvit stages: model.mvit[0], model.mvit[1], model.mvit[2]
#   - Each contains a `transformer` with multiple layers
#   - Each layer's attention is mvit_scale_dot_product(...)
#   - attn shape after rearrange: [B, P, H, N, N]
#       P = patch-positions (ph*pw)
#       H = heads (4)
#       N = number of spatial patch groups
#
# We monkey-patch each attention module's forward to capture `attn`.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[init] device = {device}")

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = T.Compose([
    T.Resize(IMG_SIZE + 32),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    normalize,
])

# Global state, reassigned when the user switches models
MODEL = None
LAYER_INFO = []
CURRENT_MODEL_NAME = None
attention_store = {}  # (stage_idx, layer_idx) -> attn tensor on cpu


def make_attn_forward(stage_idx, layer_idx):
    def wrapper(self, x):
        from einops import rearrange
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attention_store[(stage_idx, layer_idx)] = attn.detach().cpu()
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out), q, k, v
    return wrapper


def _patch_attention_modules(model):
    feat_sizes = [IMG_SIZE // 8, IMG_SIZE // 16, IMG_SIZE // 32]  # 32, 16, 8 for IMG_SIZE=256
    patch_sizes = [2, 2, 1]
    info = []
    for s_idx, mvit_block in enumerate(model.mvit):
        ph = patch_sizes[s_idx]
        feat = feat_sizes[s_idx]
        grid = feat // ph
        transformer = mvit_block.transformer
        for l_idx, (prenorm_attn, prenorm_ff) in enumerate(transformer.layers):
            attn_module = prenorm_attn.fn
            attn_module.forward = make_attn_forward(s_idx, l_idx).__get__(
                attn_module, type(attn_module))
            info.append({
                "stage": s_idx,
                "layer": l_idx,
                "grid_h": grid,
                "grid_w": grid,
                "patch_size": ph,
                "feat_size": feat,
                "num_patch_pos": ph * ph,
            })
    return info


def load_model(name: str):
    """Build the named model, load its checkpoint, patch attention hooks.
    Replaces the global MODEL, LAYER_INFO, CURRENT_MODEL_NAME.
    """
    global MODEL, LAYER_INFO, CURRENT_MODEL_NAME
    if name not in CHECKPOINTS:
        raise ValueError(f"unknown model: {name}")
    ctor, ckpt_path = CHECKPOINTS[name]
    print(f"[model] building {name} ...")
    model = ctor(img_size=IMG_SIZE)

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            report = load_backbone_from_checkpoint_into_fcn(
                ckpt_path, model,
                dst_backbone_attr=None,
                ignore_if_contains=(),
                trust_checkpoint=True,
                verbose=False,
            )
            print(f"[model] {name}: loaded {report['num_loaded']} tensors from {ckpt_path}")
        except Exception as e:
            print(f"[model] WARNING: could not load checkpoint for {name}: {e}")
            print(f"[model] {name}: using random weights")
    else:
        print(f"[model] {name}: no checkpoint at {ckpt_path}, using random weights")

    model.eval().to(device)
    info = _patch_attention_modules(model)
    n_out = model.head.out_features
    print(f"[model] {name}: patched {len(info)} attention layers, head outputs {n_out} classes")

    # Free old model from VRAM if any
    if MODEL is not None and device.type == "cuda":
        del MODEL
        torch.cuda.empty_cache()

    MODEL = model
    LAYER_INFO = info
    CURRENT_MODEL_NAME = name


load_model(DEFAULT_MODEL)


# ============================================================
# IMAGE LISTING (Imagenette + uploads)
# ============================================================

def list_imagenette_images():
    out = []
    base = Path(IMAGENETTE_DIR)
    if not base.exists():
        return out
    for class_dir in sorted(base.iterdir()):
        if not class_dir.is_dir():
            continue
        cname = IMAGENETTE_CLASSES.get(class_dir.name, class_dir.name)
        files = sorted([p for p in class_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")])[:30]
        for f in files:
            out.append({
                "source": "imagenette",
                "label": f"{cname} — {f.name}",
                "ref": f"imagenette:{class_dir.name}/{f.name}",
            })
    return out


def list_uploaded_images():
    out = []
    if not UPLOADS_DIR.exists():
        return out
    for f in sorted(UPLOADS_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            out.append({
                "source": "upload",
                "label": f"[upload] {f.name}",
                "ref": f"upload:{f.name}",
            })
    return out


def all_images():
    # uploads first so a freshly-uploaded image is easy to find
    return list_uploaded_images() + list_imagenette_images()


def resolve_image(ref: str) -> Path:
    """`ref` is like 'imagenette:<class>/<file>' or 'upload:<file>'."""
    if ":" not in ref:
        raise ValueError(f"bad image ref: {ref}")
    src, rel = ref.split(":", 1)
    if src == "imagenette":
        path = Path(IMAGENETTE_DIR) / rel
    elif src == "upload":
        # prevent path traversal
        name = Path(rel).name
        path = UPLOADS_DIR / name
    else:
        raise ValueError(f"unknown image source: {src}")
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    return path


# ============================================================
# CORE: run model + extract attention
# ============================================================

def compute_attention_map(image_ref: str, stage: int, layer: int,
                          head: str = "mean"):
    """Returns the attention tensor [P, N, N] (after head reduction) for the
    chosen layer. The frontend uses both N (token grid) and P (patch position)
    to compose a feat_size x feat_size visualization, where each (p, n) cell
    maps to a unique spatial location of the input feature map."""
    img_path = resolve_image(image_ref)
    pil = Image.open(img_path).convert("RGB")

    display_pil = T.Compose([
        T.Resize(IMG_SIZE + 32),
        T.CenterCrop(IMG_SIZE),
    ])(pil)

    x = preprocess(pil).unsqueeze(0).to(device)

    attention_store.clear()
    with torch.no_grad():
        logits = MODEL(x)
    pred_idx = int(logits.argmax(dim=-1).item())

    key = (stage, layer)
    if key not in attention_store:
        raise ValueError(f"no attention captured for stage={stage} layer={layer}")
    attn = attention_store[key]  # [B, P, H, N, N]
    B, P, H, N, _ = attn.shape

    # head selection -> [B, P, N, N]
    if head == "mean":
        a = attn.mean(dim=2)
    else:
        h = int(head)
        a = attn[:, :, h, :, :]

    matrices = a[0].numpy()  # [P, N, N]

    info = next(li for li in LAYER_INFO if li["stage"] == stage and li["layer"] == layer)

    buf = io.BytesIO()
    display_pil.save(buf, format="PNG")
    orig_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "image_b64": orig_b64,
        "matrices": matrices.tolist(),  # [P][N][N]
        "grid_h": info["grid_h"],       # token grid (e.g. 16 at stage 0)
        "grid_w": info["grid_w"],
        "patch_size": info["patch_size"],
        "feat_size": info["feat_size"], # full feature-map size (e.g. 32 at stage 0)
        "num_patch_pos": info["num_patch_pos"],
        "pred_idx": pred_idx,
        "model": CURRENT_MODEL_NAME,
    }


# ============================================================
# FLASK ROUTES
# ============================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload


@app.route("/")
def index():
    n = MODEL.head.out_features
    classes = [{"idx": i, "name": class_name(i)} for i in range(n)]
    imagenette_indices = sorted(IMAGENETTE_WNID_TO_IMAGENET_IDX.values())
    return render_template("index.html",
                           images=all_images(),
                           layers=LAYER_INFO,
                           img_size=IMG_SIZE,
                           display_size=512,
                           models=list(CHECKPOINTS.keys()),
                           current_model=CURRENT_MODEL_NAME,
                           classes=classes,
                           num_classes=n,
                           imagenette_indices=imagenette_indices)


@app.route("/api/state")
def api_state():
    """Return current model name, layer info, and image list (used after uploads
    or model switches to refresh the UI without a full page reload)."""
    return jsonify({
        "ok": True,
        "current_model": CURRENT_MODEL_NAME,
        "models": list(CHECKPOINTS.keys()),
        "layers": LAYER_INFO,
        "images": all_images(),
    })


@app.route("/api/switch_model", methods=["POST"])
def api_switch_model():
    data = request.get_json()
    name = data.get("name")
    try:
        load_model(name)
        return jsonify({"ok": True, "current_model": CURRENT_MODEL_NAME,
                        "layers": LAYER_INFO})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "no file part"})
    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "error": "no filename"})
    ext = Path(f.filename).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        return jsonify({"ok": False, "error": f"unsupported extension: {ext}"})

    # safe unique name: <stem>_<uuid8><ext>
    stem = Path(f.filename).stem.replace(" ", "_")[:40]
    safe_name = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"
    save_path = UPLOADS_DIR / safe_name
    f.save(save_path)

    # validate it's a real image
    try:
        Image.open(save_path).convert("RGB")
    except Exception as e:
        save_path.unlink(missing_ok=True)
        return jsonify({"ok": False, "error": f"not a valid image: {e}"})

    return jsonify({
        "ok": True,
        "ref": f"upload:{safe_name}",
        "label": f"[upload] {safe_name}",
        "images": all_images(),
    })


@app.route("/api/attention", methods=["POST"])
def api_attention():
    data = request.get_json()
    image_ref = data["image"]
    stage = int(data["stage"])
    layer = int(data["layer"])
    head = data.get("head", "mean")
    try:
        result = compute_attention_map(image_ref, stage, layer, head)
        return jsonify({"ok": True, **result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)})


# ============================================================
# GRAD-CAM
# ============================================================

def _pil_to_b64(pil):
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def compute_gradcam(image_ref: str, target_class: int = -1):
    """Grad-CAM on the last conv block (model.out). Returns image + heatmap."""
    img_path = resolve_image(image_ref)
    pil = Image.open(img_path).convert("RGB")
    display_pil = T.Compose([T.Resize(IMG_SIZE + 32), T.CenterCrop(IMG_SIZE)])(pil)
    x = preprocess(pil).unsqueeze(0).to(device)

    target_layer = MODEL.out  # last spatial block before global pool, output 8x8

    activations = {}
    gradients = {}

    def fwd_hook(_, __, output):
        activations["v"] = output  # keep grad-tracking version
    def bwd_hook(_, grad_in, grad_out):
        gradients["v"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        # Need gradients but we don't update params, so set requires_grad on input
        x.requires_grad_(True)
        MODEL.zero_grad(set_to_none=True)
        logits = MODEL(x)
        if target_class < 0:
            target_class = int(logits.argmax(dim=-1).item())
        score = logits[0, target_class]
        score.backward()

        A = activations["v"]   # [1, C, h, w]
        dA = gradients["v"]    # [1, C, h, w]
        weights = dA.mean(dim=(2, 3), keepdim=True)   # GAP over spatial
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))[0, 0]  # [h, w]
        cam_min, cam_max = float(cam.min().item()), float(cam.max().item())
        if cam_max - cam_min > 1e-12:
            cam_norm = ((cam - cam_min) / (cam_max - cam_min)).detach().cpu().numpy()
        else:
            cam_norm = np.zeros_like(cam.detach().cpu().numpy())

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            top = probs.argsort()[::-1][:10]
            top_list = [{"idx": int(i), "name": class_name(int(i)),
                         "prob": float(probs[i])} for i in top]
    finally:
        h1.remove(); h2.remove()
        x.requires_grad_(False)
        MODEL.zero_grad(set_to_none=True)

    return {
        "image_b64": _pil_to_b64(display_pil),
        "cam": cam_norm.tolist(),
        "h": int(cam_norm.shape[0]),
        "w": int(cam_norm.shape[1]),
        "target_class": int(target_class),
        "target_class_name": class_name(int(target_class)),
        "top": top_list,
    }


@app.route("/api/gradcam", methods=["POST"])
def api_gradcam():
    data = request.get_json()
    image_ref = data["image"]
    target_class = int(data.get("target_class", -1))
    try:
        result = compute_gradcam(image_ref, target_class)
        return jsonify({"ok": True, **result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)})


# ============================================================
# CONV FILTER VISUALIZATION (stem only; deeper convs aren't RGB-shaped)
# ============================================================

def get_stem_filters():
    """Return the 32 RGB stem filters as base64 PNGs (upscaled & per-filter normalized)."""
    w = MODEL.stem[0].conv.weight.detach().cpu().numpy()  # [C_out, 3, kh, kw]
    out = []
    for i, filt in enumerate(w):
        # filt: [3, kh, kw], normalize per-filter to [0,1]
        f = filt.transpose(1, 2, 0)  # HWC
        fmin, fmax = f.min(), f.max()
        if fmax - fmin > 1e-12:
            f = (f - fmin) / (fmax - fmin)
        else:
            f = np.zeros_like(f)
        f8 = (f * 255).astype(np.uint8)
        # upscale 32x for visibility (3x3 -> 96x96)
        pil = Image.fromarray(f8, mode="RGB").resize((96, 96), Image.NEAREST)
        out.append({"idx": i, "image_b64": _pil_to_b64(pil),
                    "min": float(fmin), "max": float(fmax)})
    return out


@app.route("/api/conv_filters", methods=["POST"])
def api_conv_filters():
    try:
        return jsonify({
            "ok": True,
            "filters": get_stem_filters(),
            "info": (f"Showing the {MODEL.stem[0].conv.out_channels} filters of "
                     f"the first conv (stem[0].conv): kernel "
                     f"{tuple(MODEL.stem[0].conv.weight.shape[2:])}, "
                     f"3 input channels (RGB). Deeper conv filters have many input "
                     f"channels and aren't directly visualizable as RGB images — "
                     f"use the activation-maximization page for those."),
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)})


# ============================================================
# ACTIVATION MAXIMIZATION (per Imagenette class)
# ============================================================
#
# NOTE on activation maximization: this model is trained on ImageNet-1k
# (1000 classes). The factory functions in nanovit.py hardcode nclasses=1000.

def class_name(idx: int) -> str:
    """Return the human label for a class index (1000-class ImageNet by default)."""
    if 0 <= idx < len(IMAGENET_LABELS):
        return IMAGENET_LABELS[idx]
    return str(idx)


# Imagenette images are a subset of ImageNet-1k. We keep the WNID->ImageNet-idx
# mapping defined above (IMAGENETTE_WNID_TO_IMAGENET_IDX) so we can highlight
# the "true" class on the Grad-CAM page when an Imagenette image is chosen.


def _jitter(t, max_shift):
    """Random circular shift on H, W."""
    if max_shift <= 0:
        return t
    sh = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    sw = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    return torch.roll(t, shifts=(sh, sw), dims=(2, 3))


def activation_maximization(target_class: int, steps: int = 256,
                            lr: float = 0.05, tv_weight: float = 1e-4,
                            l2_weight: float = 1e-3, jitter_px: int = 8,
                            seed: int = 0):
    """Optimize an image to maximize logit[target_class].
    Uses pixel-space optimization with jitter + TV/L2 regularization. Simple
    and reliable — Olah-style fourier parameterization isn't necessary at this
    image size, and adds complexity."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = (torch.randn(1, 3, IMG_SIZE, IMG_SIZE, generator=g) * 0.1).to(device)
    img.requires_grad_(True)
    optimizer = torch.optim.Adam([img], lr=lr)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    MODEL.zero_grad(set_to_none=True)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        # jittered, normalized input
        j = _jitter(img, jitter_px)
        x = (j - mean) / std
        logits = MODEL(x)
        # maximize target logit
        target_score = logits[0, target_class]
        # regularizers (penalties)
        tv = (
            (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean() +
            (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
        )
        l2 = img.pow(2).mean()
        loss = -target_score + tv_weight * tv + l2_weight * l2
        loss.backward()
        optimizer.step()
        # soft clamp to a reasonable visual range
        with torch.no_grad():
            img.clamp_(-2.0, 2.0)

    MODEL.zero_grad(set_to_none=True)

    # Render: undo nothing (we optimized in unnormalized pixel-ish space),
    # just min-max scale per channel so it looks decent.
    with torch.no_grad():
        v = img[0].detach().cpu().numpy()  # [3, H, W]
        v = v - v.min()
        if v.max() > 1e-12:
            v = v / v.max()
        v8 = (v.transpose(1, 2, 0) * 255).astype(np.uint8)
        pil = Image.fromarray(v8, mode="RGB")

        # also report the final logit / prob
        x_eval = (img - mean) / std
        logits = MODEL(x_eval)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    return {
        "image_b64": _pil_to_b64(pil),
        "target_class": int(target_class),
        "target_class_name": class_name(int(target_class)),
        "final_logit": float(logits[0, target_class].item()),
        "final_prob": float(probs[target_class]),
    }


@app.route("/api/activation_max", methods=["POST"])
def api_activation_max():
    data = request.get_json()
    target_class = int(data["target_class"])
    steps = int(data.get("steps", 256))
    lr = float(data.get("lr", 0.05))
    tv_weight = float(data.get("tv_weight", 1e-4))
    seed = int(data.get("seed", 0))
    n = MODEL.head.out_features
    if target_class < 0 or target_class >= n:
        return jsonify({"ok": False, "error": f"target_class must be in [0, {n-1}]"})
    try:
        result = activation_maximization(
            target_class=target_class, steps=steps, lr=lr,
            tv_weight=tv_weight, seed=seed,
        )
        return jsonify({"ok": True, **result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/classes")
def api_classes():
    n = MODEL.head.out_features
    return jsonify({
        "classes": [{"idx": i, "name": class_name(i)} for i in range(n)],
        "num_classes": n,
        "imagenette_indices": sorted(IMAGENETTE_WNID_TO_IMAGENET_IDX.values()),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)