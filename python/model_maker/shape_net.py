"""TripoSG's shape model (weights MIT, VAST-AI-Research) run entirely with model-maker's own network code: no code of
TripoSG's repository is fetched, imported or run (its code folder holds a licence that excludes the European Union, the
United Kingdom and South Korea; its transformer file is under the Tencent Hunyuan Community License). The block and the
denoiser below are our own, built from diffusers' HunyuanDiT block (Apache-2.0,
`diffusers/models/transformers/hunyuan_transformer_2d.py`), the weights' own tensor names and shapes, and the arguments
the upstream VAE passed to the block, recorded by running it as a black box. What the weights
alone did not settle (how the time step enters, the sinusoid's convention, the skip's order) was chosen by measurement:
a shape encoded by the VAE, noised to a known level, and each choice's prediction of the way back scored; `CHOICES`
holds the result. The decoder from the latent to the field, the attention processor and the query points' embedding
are our own too (`shape_vae`, from diffusers' `AttnProcessor2_0`). The surface is found by our own narrow-band search
(`surface`): a coarse grid, then finer cells only near the surface, then scikit-image's marching cubes (BSD).

The weights are read from `TRIPOSG_WEIGHTS`. Needs an NVIDIA card (about 8 GB).
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

WEIGHTS = os.environ.get("TRIPOSG_WEIGHTS", os.path.join("models", "triposg"))
# The shape's room: TripoSG's shapes lie within this cube.
BOUND = 1.005
# The latent: this many tokens of this many channels.
TOKENS = 2048

def _norm(kind, dim, eps):
    from diffusers.models.normalization import FP32LayerNorm

    return FP32LayerNorm(dim, eps) if kind == "fp32_layer_norm" else nn.LayerNorm(dim, eps)


class Block(nn.Module):
    """A pre-norm transformer block: self-attention, cross-attention and a feed-forward, each optional, each added
    back to its input; a block in the denoiser's second half first joins the skip from its mirror in the first."""

    def __init__(self, dim, num_attention_heads, use_self_attention=True, use_cross_attention=False,
                 self_attention_norm_type="fp32_layer_norm", cross_attention_dim=None,
                 cross_attention_norm_type=None, activation_fn="gelu", norm_type="fp32_layer_norm", norm_eps=1e-5,
                 qk_norm=False, qkv_bias=False, skip=False, skip_first=False, cross_first=False, qk_kind="rms_norm",
                 split="per_head", **_unused):
        super().__init__()
        from diffusers.models.attention import FeedForward
        from diffusers.models.attention_processor import Attention

        if split == "per_head":
            from .shape_vae import JoinedHeadsAttnProcessor as processor
        else:
            from diffusers.models.attention_processor import AttnProcessor2_0 as processor
        head = dim // num_attention_heads
        self.skip_first = skip_first
        self.cross_first = cross_first
        self.attn1 = self.attn2 = None
        if use_self_attention:
            self.norm1 = _norm(self_attention_norm_type or norm_type, dim, norm_eps)
            self.attn1 = Attention(query_dim=dim, heads=num_attention_heads, dim_head=head, bias=qkv_bias,
                                   qk_norm="rms_norm" if qk_norm else None, eps=1e-6, processor=processor())
        if use_cross_attention:
            self.norm2 = _norm(norm_type, dim, norm_eps)
            self.attn2 = Attention(query_dim=dim, cross_attention_dim=cross_attention_dim, heads=num_attention_heads,
                                   dim_head=head, bias=qkv_bias, cross_attention_norm=cross_attention_norm_type,
                                   qk_norm="rms_norm" if qk_norm else None, eps=1e-6, processor=processor())
        if qk_norm and qk_kind == "layer_norm_without_bias":
            for attention in (self.attn1, self.attn2):
                if attention is not None:
                    attention.norm_q = nn.LayerNorm(head, eps=1e-6, bias=False)
                    attention.norm_k = nn.LayerNorm(head, eps=1e-6, bias=False)
        self.norm3 = _norm(norm_type, dim, norm_eps)
        self.ff = FeedForward(dim, activation_fn=activation_fn)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        if skip:
            self.skip_norm = _norm(norm_type, dim, norm_eps)

    def forward(self, hidden_states, encoder_hidden_states=None, skip=None, **_unused):
        if self.skip_linear is not None:
            joined = torch.cat([skip, hidden_states] if self.skip_first else [hidden_states, skip], dim=-1)
            hidden_states = self.skip_norm(self.skip_linear(joined))
        if self.attn2 is not None and self.cross_first:
            hidden_states = hidden_states + self.attn2(self.norm2(hidden_states),
                                                       encoder_hidden_states=encoder_hidden_states)
        if self.attn1 is not None:
            hidden_states = hidden_states + self.attn1(self.norm1(hidden_states))
        if self.attn2 is not None and not self.cross_first:
            hidden_states = hidden_states + self.attn2(self.norm2(hidden_states),
                                                       encoder_hidden_states=encoder_hidden_states)
        return hidden_states + self.ff(self.norm3(hidden_states))


class Denoiser(nn.Module):
    """The shape latent's velocity from its noisy self, the time and the picture's DINOv2 features."""

    def __init__(self, in_channels=64, width=2048, num_layers=21, num_attention_heads=16, cross_attention_dim=1024,
                 time_as_token=True, flip_sin_to_cos=True, freq_shift=0.0, skip_first=False, time_last=False,
                 skip_fifo=False, cross_first=False, qk_kind="rms_norm", activation_fn="gelu", time_scale=1.0,
                 split="per_head", time_act="silu"):
        super().__init__()
        from diffusers.models.embeddings import TimestepEmbedding, Timesteps

        self.time_as_token = time_as_token
        self.time_last = time_last
        self.skip_fifo = skip_fifo
        self.time_scale = time_scale
        self.timesteps = Timesteps(width, flip_sin_to_cos, freq_shift)
        self.time_proj = TimestepEmbedding(width, width * 4, act_fn=time_act, out_dim=width)
        self.proj_in = nn.Linear(in_channels, width)
        half = num_layers // 2
        self.blocks = nn.ModuleList(
            Block(width, num_attention_heads, use_self_attention=True, use_cross_attention=True,
                  cross_attention_dim=cross_attention_dim, norm_type="fp32_layer_norm", norm_eps=1e-6, qk_norm=True,
                  qkv_bias=False, skip=layer > half, skip_first=skip_first, cross_first=cross_first, qk_kind=qk_kind,
                  activation_fn=activation_fn, split=split)
            for layer in range(num_layers))
        self.half = half
        self.norm_out = nn.LayerNorm(width)
        self.proj_out = nn.Linear(width, in_channels)

    def forward(self, latents, timestep, features):
        time = self.time_proj(self.timesteps(timestep * self.time_scale).to(latents.dtype))
        hidden = self.proj_in(latents)
        if not self.time_as_token:
            hidden = hidden + time[:, None]
        elif self.time_last:
            hidden = torch.cat([hidden, time[:, None]], dim=1)
        else:
            hidden = torch.cat([time[:, None], hidden], dim=1)
        skips = []
        for layer, block in enumerate(self.blocks):
            skip = None
            if layer > self.half:
                skip = skips.pop(0) if self.skip_fifo else skips.pop()
            hidden = block(hidden, encoder_hidden_states=features, skip=skip)
            if layer < self.half:
                skips.append(hidden)
        hidden = self.norm_out(hidden)
        if self.time_as_token:
            hidden = hidden[:, :-1] if self.time_last else hidden[:, 1:]
        return self.proj_out(hidden)


def load_vae(device, dtype):
    from . import shape_vae

    return shape_vae.load_decoder(os.path.join(WEIGHTS, "vae"), device, dtype)


def load_denoiser(device, dtype, **choices):
    from safetensors.torch import load_file

    model = Denoiser(**choices)
    state = load_file(os.path.join(WEIGHTS, "transformer", "diffusion_pytorch_model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"the denoiser does not match the weights: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model.to(device, dtype).eval()


def load_encoder(device, dtype):
    from transformers import BitImageProcessor, Dinov2Model

    processor = BitImageProcessor.from_pretrained(os.path.join(WEIGHTS, "feature_extractor_dinov2"))
    encoder = Dinov2Model.from_pretrained(os.path.join(WEIGHTS, "image_encoder_dinov2")).to(device, dtype).eval()
    return processor, encoder


def prepared(picture_rgb, mask):
    """The object on white, cropped to its outline and padded to a square with a tenth to spare (as TripoSG's pipeline
    prepares a picture whose outline is known)."""
    from PIL import Image

    alpha = mask.astype(np.float32)[..., None]
    image = picture_rgb.astype(np.float32) / 255 * alpha + (1 - alpha)
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    crop = image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = crop.shape[:2]
    side = int(max(h, w) * 1.2)
    square = np.ones((side, side, 3), np.float32)
    square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    return Image.fromarray((square * 255).round().astype(np.uint8))


@torch.no_grad()
def picture_features(processor, encoder, image, device, dtype):
    pixels = processor(image, return_tensors="pt").pixel_values.to(device, dtype)
    return encoder(pixels).last_hidden_state


def start_noise(seed):
    """The shape's starting noise, drawn on the CPU (float32, torch's CPU generator), the same on every machine. Drawn on
    the card, it depended on the card: torch's CUDA generator lays its draws out by the card's number of
    multiprocessors, so two cards (12 GB and 16 GB) made other shapes from one seed."""
    return torch.randn((1, TOKENS, 64), generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)


@torch.no_grad()
def sampled_latents(denoiser, features, seed, steps, guidance, device, dtype):
    """Rectified flow from noise (sigma 1) to the shape (sigma 0) in even steps, the picture's guidance against no
    picture (zero features)."""
    latents = start_noise(seed).to(device, dtype)
    both = torch.cat([torch.zeros_like(features), features])
    sigmas = [1.0 - k / steps for k in range(steps)] + [0.0]
    for sigma, following in zip(sigmas[:-1], sigmas[1:]):
        timestep = torch.full((2,), sigma * 1000.0, device=device)
        unguided, guided = denoiser(torch.cat([latents, latents]), timestep, both).chunk(2)
        velocity = unguided + guidance * (guided - unguided)
        latents = latents + (sigma - following) * velocity
    return latents


@torch.no_grad()
def surface(vae, latents, coarse=256, levels=1, band_cells=2, chunk=200_000):
    """The shape's surface: its field (a logit, positive inside) queried on a coarse grid, then on finer grids only
    within `band_cells` of where the coarse field changes sign (a narrow band, the standard way to reach a fine grid
    without querying all of it), then marching cubes on the finest grid. The band follows the sign changes, not the
    field's size: the field is a logit, not a distance, so a size threshold left stale coarse values near the
    surface and tore it. Returns corners and triangles in TripoSG's frame."""
    from scipy import ndimage
    from skimage import measure

    device = latents.device
    # The latent's tokens worked out once; each chunk of points then asks them (the same field as asking the whole
    # decoder each time, without its 16 self-attention blocks per chunk).
    tokens = vae.tokens(latents)

    def field_at(points):
        values = []
        for start in range(0, len(points), chunk):
            part = torch.as_tensor(points[start:start + chunk], device=device, dtype=latents.dtype)[None]
            values.append(vae.field(tokens, part)[0].float().cpu().numpy())
        return np.concatenate(values)

    size = coarse + 1
    axis = np.linspace(-BOUND, BOUND, size, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    field = field_at(grid).reshape(size, size, size)
    for _ in range(levels):
        finer = (size - 1) * 2 + 1
        # The coarse field carried to the finer grid, then queried again only near the surface.
        sign = field > 0
        crossing = ndimage.binary_dilation(sign, iterations=1) & ndimage.binary_dilation(~sign, iterations=1)
        crossing = ndimage.binary_dilation(crossing, iterations=band_cells)
        field = ndimage.zoom(field, finer / size, order=1)
        ask = ndimage.zoom(crossing.astype(np.uint8), finer / size, order=0).astype(bool)
        axis = np.linspace(-BOUND, BOUND, finer, dtype=np.float32)
        index = np.argwhere(ask)
        field[ask] = field_at(axis[index])
        size = finer
    corners, triangles, _, _ = measure.marching_cubes(field, 0.0)
    corners = corners * (2 * BOUND / (size - 1)) - BOUND
    triangles, specks = without_specks(corners, triangles)
    # Wound outward whichever way marching cubes went: a closed surface wound outward encloses a positive volume.
    a, b, c = (corners[triangles[:, k]] for k in range(3))
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        triangles = triangles[:, ::-1]
    return corners.astype(np.float32), np.ascontiguousarray(triangles), specks


# A loose piece of the surface with fewer than this share of its triangles is a speck of the field, not a part
# (a fleck floating beside a part); whole parts (a handle, a knob) are far larger.
LEAST_PIECE_SHARE = 0.005


def without_specks(corners, triangles):
    """The surface's triangles without its loose specks (the usual clean-up after marching cubes), and the specks
    left out: [(triangles, middle, size)] in the shape's frame."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    n = len(corners)
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]]])
    graph = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, piece_of = connected_components(graph, directed=False)
    piece = piece_of[triangles[:, 0]]
    sizes = np.bincount(piece)
    kept = sizes[piece] >= LEAST_PIECE_SHARE * len(triangles)
    specks = []
    for p in np.unique(piece[~kept]):
        used = corners[np.unique(triangles[piece == p])]
        low, high = used.min(0), used.max(0)
        specks.append((int(sizes[p]), ((low + high) / 2).tolist(), float((high - low).max())))
    return triangles[kept], specks


# The models, loaded from disk once and kept in the computer's memory between objects; each goes to the card only for
# its own step (loading them for every object was a quarter of the card's time).
_KEPT = {}


def kept_models(dtype, choices=None):
    key = (dtype, tuple(sorted((choices or CHOICES).items())))
    if key not in _KEPT:
        _KEPT.clear()
        _KEPT[key] = {"encoder": load_encoder("cpu", dtype), "denoiser": load_denoiser("cpu", dtype, **(choices or CHOICES)),
                      "vae": load_vae("cpu", dtype)}
    return _KEPT[key]


class on_the_card:
    """A model on the card for a step, back in the computer's memory after it."""

    def __init__(self, model, device):
        self.model, self.device = model, device

    def __enter__(self):
        self.model.to(self.device)
        return self.model

    def __exit__(self, *_):
        self.model.to("cpu")
        torch.cuda.empty_cache()


# The denoiser's choices as measured (see the module's note).
CHOICES = dict(time_as_token=True, flip_sin_to_cos=False, freq_shift=0.0, skip_first=True)


def make(picture_rgb, mask, seed=42, steps=50, guidance=7.0, device="cuda", choices=None):
    """The object's surface from its picture (H x W x 3, uint8) and its cut-out (bool): (corners, triangles, the
    specks left out), in TripoSG's frame (+Y up, facing +Z, within the unit cube)."""
    from PIL import Image

    dtype = torch.float16
    started = time.time()
    loaded = kept_models(dtype, choices)
    processor, encoder = loaded["encoder"]
    image = prepared(np.asarray(Image.fromarray(picture_rgb).convert("RGB")), mask)
    with on_the_card(encoder, device):
        features = picture_features(processor, encoder, image, device, dtype)
    with on_the_card(loaded["denoiser"], device):
        latents = sampled_latents(loaded["denoiser"], features, seed, steps, guidance, device, dtype)
    with on_the_card(loaded["vae"], device):
        corners, triangles, specks = surface(loaded["vae"], latents)
    print(f"shape: {len(corners)} corners, {len(triangles)} triangles in {time.time() - started:.0f} s", flush=True)
    return corners, triangles, specks
