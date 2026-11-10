"""The object cut from its picture's backdrop by BiRefNet (MIT), its own model code read from its checked folder.

The raw mask is kept for the picture checks (a second piece is a fault they must see); the shape model is given the
largest piece with its holes as the picture shows them: only a pinhole of the cut-out is filled. With every hole
closed off inside the object filled, the shape model saw the backdrop's grey (darkened by the object's shadow) in a
jug's handle, as if it were a surface, and filled the loop with a web (docs/objects.md, "The jug's handle"). The holes
are kept and the object is put on white, as the shape model's input was prepared in its training.
"""

import os
import sys
import types

import numpy as np
from PIL import Image
from scipy import ndimage

SIDE = 1024
THRESHOLD = 0.5
# A hole of the cut-out smaller than this (pixels of the 1024-pixel picture) is a pinhole of the cut, filled; a larger
# one is a hole in the object (a jug's handle, a chair's back between its spindles) and is kept.
LEAST_HOLE_PIXELS = 200
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Cutter:
    """BiRefNet, loaded once from `folder` (checked against the registry by the caller)."""

    def __init__(self, folder, device=None):
        import torch

        # Its own code from its folder, as the package its relative imports expect.
        package = types.ModuleType("birefnet_folder")
        package.__path__ = [folder]
        sys.modules["birefnet_folder"] = package
        from birefnet_folder.birefnet import BiRefNet

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = BiRefNet.from_pretrained(folder).to(self.device, self.dtype).eval()

    def mask(self, image):
        """Where the object is (bool, the picture's size)."""
        import torch

        rgb = image.convert("RGB")
        small = rgb.resize((SIDE, SIDE), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = ((x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None])[None]
        with torch.no_grad():
            found = self.model(x.to(self.device, self.dtype))[-1].sigmoid()[0, 0].float().cpu().numpy()
        found = Image.fromarray((found * 255.0).astype(np.uint8)).resize(rgb.size, Image.Resampling.BILINEAR)
        return np.asarray(found) > THRESHOLD * 255


def for_shape(mask):
    """The mask the shape model is given: its largest piece, its pinholes filled, its holes kept."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    piece = labels == sizes.argmax()
    holes, count = ndimage.label(ndimage.binary_fill_holes(piece) & ~piece)
    if count == 0:
        return piece
    hole_sizes = np.bincount(holes.ravel())
    pinholes = np.flatnonzero(hole_sizes < LEAST_HOLE_PIXELS)
    return piece | (np.isin(holes, pinholes[pinholes > 0]))
