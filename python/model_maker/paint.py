"""The object's colours painted to fit its own shape: MV-Adapter's image-and-geometry model (Apache-2.0) paints six
views of the object from its picture, conditioned on the shape's own position and normal pictures from six cameras,
and each view goes back onto the shape through the same camera (views painted from the picture alone land beside
the modelled parts, not on them).

The position and normal pictures are drawn by model-maker's own rasteriser (`render.rasterised`): MV-Adapter's
renderer needs nvdiffrast, whose licence does not allow this use; it is never imported, nor its other mesh tools
(pymeshlab, CV-CUDA): stand-in modules take their names. MV-Adapter's code is read from `MV_ADAPTER_SRC`, a checkout
checked to be at the commit the registry names with nothing changed or added before it is imported. Needs an NVIDIA
card: about 14 GB with the text encoders moved off it; a card under 15 GB gets the UNet's blocks one at a time
(`BlockOffload`), 10.1 GB at most on a 12 GB card, the same pictures.
"""

import os
import sys
import types
from typing import Union

import numpy as np
from PIL import Image
from scipy import sparse

from . import render

SIDE = 768
# MV-Adapter's image-and-geometry cameras: front, the sides, back, from above and below.
ELEVATIONS = [0, 0, 0, 0, 89.99, -89.99]
AZIMUTHS = [0, 90, 180, 270, 180, 180]
# Each view's weight when its colours go back onto the shape: the four round it fully, above and below less.
VIEW_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 0.3, 0.3]
# A corner takes a view's colour from this facing on, by this power of it (the best view nearly alone).
LEAST_FACING = 0.2
FACING_POWER = 12
# The colour cloud for the bake: each painted pixel's surface point, normal and colour, kept where its view sees it
# this squarely or more (every surface faces one of the six views at 0.58 or more).
CLOUD_LEAST_FACING = 0.55
# A corner this far behind the drawn surface in a view (in clip depth) is hidden from it.
HIDDEN_BEHIND = 0.004
NEGATIVE = "watermark, ugly, deformed, noisy, blurry, low contrast, shadow"
# The view painter's steps: at 20 the second trial's jug, sword and barrel came out with the same coverage and
# likeness as at 50 (95.8 %, 99.9 %, 98.6 %; delta E 8.9, 9.2, 21.9 against 9.0, 9.0, 22.0), in 230 s, not 465-550 s.
STEPS = 20
# A card with less memory than this gets the UNet's blocks one at a time (BlockOffload): 12 GB ran out of memory with
# the whole UNet on it. MODEL_MAKER_VIEW_OFFLOAD=1 or 0 overrides.
WHOLE_UNET_LEAST_GIB = 15.0
GUIDANCE = 3.0
# Colours spread to the corners no view saw, from their neighbours, this many rounds.
SPREAD_ROUNDS = 60
# The registry's entry for MV-Adapter, whose `source` names the commit its code must be at.
MV_ADAPTER_ID = "mv-adapter-ig2mv-sdxl"


def _mv_adapter_on_path():
    """MV-Adapter's code on the path, its mesh tools' package and helpers replaced by stand-ins so their imports
    (nvdiffrast, pymeshlab, CV-CUDA) never run."""
    import torch

    from . import registry

    source = os.environ.get("MV_ADAPTER_SRC", "mv-adapter")
    # Its code is checked before any of it is imported: the registry's commit, nothing changed or added.
    registry.verify_checkout(MV_ADAPTER_ID, source)
    if source not in sys.path:
        sys.path.insert(0, source)
    mesh_package = types.ModuleType("mvadapter.utils.mesh_utils")
    mesh_package.__path__ = [os.path.join(source, "mvadapter/utils/mesh_utils")]
    sys.modules.setdefault("mvadapter.utils.mesh_utils", mesh_package)
    helpers = types.ModuleType("mvadapter.utils.mesh_utils.utils")
    helpers.LIST_TYPE = Union[list, np.ndarray, torch.Tensor]
    sys.modules.setdefault("mvadapter.utils.mesh_utils.utils", helpers)
    saving = types.ModuleType("mvadapter.utils.saving")
    saving.make_image_grid = saving.tensor_to_image = saving.image_to_tensor = None
    sys.modules.setdefault("mvadapter.utils.saving", saving)


def standard_with(points):
    """TripoSG's frame to MV-Adapter's (Z up, the object facing -Y), centred and held within half a unit; and how to
    undo it: (points, centre, scale), MV-Adapter's frame = (turned - centre) * scale."""
    p = np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=1)
    centre = (p.max(0) + p.min(0)) / 2
    scale = 0.5 / np.abs(p - centre).max()
    return (p - centre) * scale, centre, scale


def from_standard(p, centre, scale):
    """MV-Adapter's frame back to TripoSG's (positions; a direction with centre 0 and scale 1)."""
    turned = p / scale + centre
    return np.stack([turned[:, 0], turned[:, 2], -turned[:, 1]], axis=1)


def cameras():
    _mv_adapter_on_path()
    from mvadapter.utils.mesh_utils.camera import get_orthogonal_camera

    return get_orthogonal_camera(elevation_deg=ELEVATIONS, distance=[1.8] * 6, left=-0.55, right=0.55, bottom=-0.55,
                                 top=0.55, azimuth_deg=[x - 90 for x in AZIMUTHS], device="cpu")


def projected(points, mvp):
    """Pixel (col, row; pixel centres at whole numbers) and clip depth of each point (smaller nearer)."""
    homo = np.concatenate([points, np.ones((len(points), 1))], 1) @ mvp.T
    ndc = homo[:, :3] / homo[:, 3:4]
    return (ndc[:, 0] + 1) / 2 * SIDE - 0.5, (ndc[:, 1] + 1) / 2 * SIDE - 0.5, ndc[:, 2]


def surface_maps(points, normals, triangles, mvp):
    """Each pixel's nearest surface in a view: its position, its normal and its clip depth (inf where none)."""
    cols, rows, depth = projected(points, mvp)
    zbuf, owner, bary = render.rasterised(np.stack([cols + 0.5, rows + 0.5], 1), -depth, triangles, SIDE, SIDE)
    hit = owner >= 0
    position = np.zeros((SIDE, SIDE, 3))
    normal = np.zeros((SIDE, SIDE, 3))
    tri = triangles[owner[hit]]
    position[hit] = np.einsum("pk,pkd->pd", bary[hit], points[tri])
    n = np.einsum("pk,pkd->pd", bary[hit], normals[tri])
    normal[hit] = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    return position, normal, np.where(hit, -zbuf, np.inf)


def reference(picture_rgb, mask):
    """MV-Adapter's own preparation of the reference: the object cropped, its longer side 0.9 of the square, on grey."""
    rgb = picture_rgb.astype(np.float32) / 255
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    crop = rgb[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    alpha = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1, None].astype(np.float32)
    crop = crop * alpha + 0.5 * (1 - alpha)
    h, w = crop.shape[:2]
    scale = SIDE * 0.9 / max(h, w)
    small = Image.fromarray((crop * 255).astype(np.uint8)).resize((max(1, int(w * scale)), max(1, int(h * scale))))
    square = Image.new("RGB", (SIDE, SIDE), (128, 128, 128))
    square.paste(small, ((SIDE - small.width) // 2, (SIDE - small.height) // 2))
    return square


class BlockOffload:
    """A model's blocks on the card only while each runs, the rest of the time in pinned memory of the computer:
    block-level offloading, as later diffusers releases do it (their group offloading); torch.nn's hooks bring each
    block's weights to the card before it runs and drop them after. The same weights and the same sums, so the same
    pictures; it costs the copies (about 7 GB a step for the view painter's UNet with its adapter)."""

    def __init__(self, blocks, device):
        self.device = device
        self.kept = {}
        for block in blocks:
            tensors = list(block.parameters()) + list(block.buffers())
            for tensor in tensors:
                tensor.data = tensor.data.to("cpu").pin_memory()
                self.kept[tensor] = tensor.data
            block.register_forward_pre_hook(self._on(tensors))
            block.register_forward_hook(self._off(tensors))

    def _on(self, tensors):
        def hook(module, args):
            for tensor in tensors:
                tensor.data = self.kept[tensor].to(self.device, non_blocking=True)
        return hook

    def _off(self, tensors):
        def hook(module, args, output):
            for tensor in tensors:
                tensor.data = self.kept[tensor]
        return hook


BLOCK_LISTS = ("down_blocks", "mid_block", "up_blocks")


def unet_blocks(unet):
    """The UNet's big parts, each run once a step: its down, middle and up blocks (the rest is small)."""
    return [*unet.down_blocks, unet.mid_block, *unet.up_blocks]


def offload_wanted():
    import torch

    named = os.environ.get("MODEL_MAKER_VIEW_OFFLOAD")
    if named in ("0", "1"):
        return named == "1"
    return torch.cuda.get_device_properties(0).total_memory / 2 ** 30 < WHOLE_UNET_LEAST_GIB


class ViewPainter:
    """The view painter, built from disk once and kept in the computer's memory; its big parts go to the card for
    each painting."""

    def __init__(self, sdxl_dir, vae_dir, adapter_dir):
        import torch
        from diffusers import AutoencoderKL

        _mv_adapter_on_path()
        from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
        from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
        from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

        vae = AutoencoderKL.from_pretrained(vae_dir)
        pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(sdxl_dir, vae=vae, variant="fp16", torch_dtype=torch.float16)
        pipe.scheduler = ShiftSNRScheduler.from_scheduler(pipe.scheduler, shift_mode="interpolated", shift_scale=8.0)
        pipe.init_custom_adapter(num_views=6, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0)
        pipe.load_custom_adapter(adapter_dir, weight_name="mvadapter_ig2mv_sdxl.safetensors")
        pipe.to(device="cpu", dtype=torch.float16)
        pipe.cond_encoder.to(device="cpu", dtype=torch.float16)
        # The feed-forward layers in chunks (the same sums in smaller pieces) and the views decoded one at a time:
        # it fits a 16 GB card another program holds a little of.
        for block in pipe.unet.modules():
            if hasattr(block, "set_chunk_feed_forward"):
                block.set_chunk_feed_forward(chunk_size=1, dim=1)
        pipe.vae.enable_slicing()
        self.pipe = pipe
        self.offload = None
        if offload_wanted():
            self.offload = BlockOffload(unet_blocks(pipe.unet), "cuda")

    def views(self, control, text, reference_image, seed, steps=STEPS):
        import torch

        pipe = self.pipe
        # With the blocks offloaded only the UNet's small parts move; each block comes onto the card as it runs.
        self._unet_to("cuda")
        pipe.vae.to("cuda")
        pipe.cond_encoder.to("cuda")
        # The words' embeddings first, then both text encoders off the card.
        pipe.text_encoder.to("cuda")
        pipe.text_encoder_2.to("cuda")
        with torch.no_grad():
            embeds, negative_embeds, pooled, negative_pooled = pipe.encode_prompt(
                prompt=text, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=True,
                negative_prompt=NEGATIVE)
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()
        # It judges its device by its parts; with the encoders away it would think the CPU.
        type(pipe)._execution_device = property(lambda self: torch.device("cuda"))
        control = torch.from_numpy(control.clip(0, 1)).permute(0, 3, 1, 2).to("cuda", torch.float16)
        made = pipe(prompt_embeds=embeds, negative_prompt_embeds=negative_embeds, pooled_prompt_embeds=pooled,
                    negative_pooled_prompt_embeds=negative_pooled, height=SIDE, width=SIDE,
                    num_inference_steps=steps, guidance_scale=GUIDANCE, num_images_per_prompt=6,
                    control_image=control, control_conditioning_scale=1.0, reference_image=reference_image,
                    reference_conditioning_scale=1.0, cross_attention_kwargs={"scale": 1.0},
                    # Its noise drawn on the CPU, the same on every card (diffusers draws it there and moves it
                    # over); drawn on the card it depends on the card.
                    generator=torch.Generator(device="cpu").manual_seed(seed)).images
        self._unet_to("cpu")
        pipe.vae.to("cpu")
        pipe.cond_encoder.to("cpu")
        torch.cuda.empty_cache()
        return made

    def _unet_to(self, device):
        if self.offload is None:
            self.pipe.unet.to(device)
            return
        for name, child in self.pipe.unet.named_children():
            if name not in BLOCK_LISTS:
                child.to(device)


def sampled_at(rgb, cols, rows):
    """Bilinear samples of a picture at (col, row)."""
    h, w = rgb.shape[:2]
    c0 = np.clip(np.floor(cols).astype(int), 0, w - 2)
    r0 = np.clip(np.floor(rows).astype(int), 0, h - 2)
    fc, fr = np.clip(cols - c0, 0, 1)[:, None], np.clip(rows - r0, 0, 1)[:, None]
    return ((rgb[r0, c0] * (1 - fc) + rgb[r0, c0 + 1] * fc) * (1 - fr)
            + (rgb[r0 + 1, c0] * (1 - fc) + rgb[r0 + 1, c0 + 1] * fc) * fr)


def spread(colours, known, triangles, rounds=SPREAD_ROUNDS):
    """Colours of the corners no view saw, spread from their neighbours."""
    n = len(colours)
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    adjacency = sparse.coo_matrix((np.ones(len(edges) * 2), (np.r_[edges[:, 0], edges[:, 1]],
                                                             np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    colours, weight = colours.copy(), known.astype(float)
    for _ in range(rounds):
        summed = adjacency @ (colours * weight[:, None])
        counted = adjacency @ weight
        fresh = (weight == 0) & (counted > 0)
        if not fresh.any():
            break
        colours[fresh] = summed[fresh] / counted[fresh, None]
        weight[fresh] = 1.0
    return colours


def painted(painter, corners, triangles, picture_rgb, mask, text, seed=0, views_out=None, report=print, steps=STEPS):
    """The shape's colours from six views painted to fit it: (each corner's colour 0-1, which corners a view saw, the
    colour cloud (points, normals, colours) in the shape's frame)."""
    from .mesh_tools import corner_normals

    points, centre, scale = standard_with(corners.astype(np.float64))
    normals = corner_normals(points, triangles)
    cams = cameras()
    mvps = [cams.mvp_mtx[v].numpy().astype(np.float64) for v in range(6)]
    maps = [surface_maps(points, normals, triangles, mvp) for mvp in mvps]
    control = np.stack([np.concatenate([np.where(np.isfinite(z)[..., None], pos + 0.5, 0.5),
                                        np.where(np.isfinite(z)[..., None], nrm / 2 + 0.5, 0.5)], -1)
                        for pos, nrm, z in maps])
    views = painter.views(control, text, reference(picture_rgb, mask), seed, steps)
    if views_out:
        os.makedirs(views_out, exist_ok=True)
        for k, image in enumerate(views):
            image.save(os.path.join(views_out, f"view-{k}.png"))
    total = np.zeros((len(points), 3))
    weights = np.zeros(len(points))
    cloud = []
    for v, image in enumerate(views):
        cols, rows, depth = projected(points, mvps[v])
        position, normal, zbuf = maps[v]
        c = np.clip(np.round(cols).astype(int), 0, SIDE - 1)
        r = np.clip(np.round(rows).astype(int), 0, SIDE - 1)
        seen = depth <= zbuf[r, c] + HIDDEN_BEHIND
        towards = -cams.c2w[v, :3, 2].numpy()  # the camera looks along its -z
        facing = normals @ -towards
        takes = seen & (facing > LEAST_FACING)
        w = VIEW_WEIGHTS[v] * np.where(takes, facing, 0.0) ** FACING_POWER
        rgb = np.asarray(image.convert("RGB")).astype(np.float64) / 255
        total += sampled_at(rgb, cols, rows) * w[:, None]
        weights += w
        keep = np.isfinite(zbuf) & ((normal @ -towards) >= CLOUD_LEAST_FACING)
        cloud.append((from_standard(position[keep], centre, scale), from_standard(normal[keep], np.zeros(3), 1.0),
                      rgb[keep]))
        report(f"view {v}: colours {int(takes.sum())} corners")
    known = weights > 0
    colours = np.where(known[:, None], total / np.maximum(weights, 1e-12)[:, None], 0.5)
    cloud = tuple(np.concatenate([part[k] for part in cloud]).astype(np.float32) for k in range(3))
    return spread(colours, known, triangles), known, cloud
