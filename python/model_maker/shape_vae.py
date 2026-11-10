"""The shape model's decoder: TripoSG's latent (2048 tokens of 64 channels; weights MIT, VAST-AI-Research) turned into
the field the surface is found in (a logit at any point, positive inside). All of it is model-maker's own code; no code
of TripoSG's repository is fetched, imported or run (its code folder holds a licence that excludes the European Union,
the United Kingdom and South Korea).

What each piece follows (diffusers, Apache-2.0, Copyright The HuggingFace Team; NOTICE):

* `JoinedHeadsAttnProcessor` follows diffusers' `AttnProcessor2_0` (`diffusers/models/attention_processor.py`):
  the same projections, norms and `scaled_dot_product_attention`. Only how the heads take their channels differs: the
  projections are joined (query, key and value in self-attention; key and value in cross-attention) and each head takes
  its own run of the joined channels. That order is fixed by the weights (they were trained with it); it was found by
  comparing outputs on random inputs, and it is the one the denoiser's and this decoder's weights need.
* `point_features` is the frequency (positional) encoding the TripoSG paper describes for its query points: the point,
  then the sines and the cosines of each coordinate times 1, 2, 4, ... 128 (`vae/config.json`: 8 frequencies, no
  factor of pi), coordinate by coordinate. Its 51 channels are the width of the weights' `decoder.proj_query`.
* `ShapeDecoder` is built from the weights' tensor names and shapes and `vae/config.json`: `post_quant` (64 to 1024),
  16 self-attention blocks over the latent's tokens, one cross-attention block in which the query points look at those
  tokens, a layer norm and `proj_out` (1024 to 1), whose sign is turned so the field is positive inside. Its blocks are `shape_net.Block` (written from diffusers' HunyuanDiT
  block, `diffusers/models/transformers/hunyuan_transformer_2d.py`) on diffusers' `Attention`, `FeedForward` and
  `FP32LayerNorm`.

How what the weights do not say was settled: the upstream code was run as a black box, the arguments it passed to
model-maker's own block recorded, and its outputs compared with these on the same inputs (the heads' order, the
field's sign). This code gives the same field to the last bit, in 32 and 16 bits (docs/objects.md, "The shape model's
own decoder").

The weights' encoder (`encoder.*`, `quant.*`) turns a mesh into a latent; model-maker only decodes, so it is not built.
Every other tensor of the file is loaded by name and none may be missing or left over (`load_decoder`).
"""

import json
import os

import torch
import torch.nn.functional as F
from torch import nn

# The weights' tensors that belong to the encoder alone (mesh to latent): never built or loaded here.
ENCODER_ONLY = ("encoder.", "quant.")


class JoinedHeadsAttnProcessor:
    """diffusers' `AttnProcessor2_0` with the heads taking their channels from the joined projections (see the module's
    note). Takes what model-maker's blocks pass: the hidden states and, for cross-attention, the encoder's."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **unused):
        if attention_mask is not None or hidden_states.ndim != 3 or attn.spatial_norm is not None \
                or attn.group_norm is not None:
            raise ValueError("JoinedHeadsAttnProcessor takes (batch, tokens, channels) without a mask or extra norms")
        residual = hidden_states
        batch = hidden_states.shape[0]
        heads = attn.heads
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            key, value = attn.to_k(hidden_states), attn.to_v(hidden_states)
            head = query.shape[-1] // heads
            joined = torch.cat([query, key, value], dim=-1).view(batch, -1, heads, 3 * head)
            query, key, value = joined.split(head, dim=-1)
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            key, value = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
            head = query.shape[-1] // heads
            query = query.view(batch, -1, heads, head)
            key, value = torch.cat([key, value], dim=-1).view(batch, -1, heads, 2 * head).split(head, dim=-1)
        query, key, value = (part.transpose(1, 2) for part in (query, key, value))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch, -1, heads * head).to(query.dtype)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def point_features(points, frequencies=8):
    """Each point (..., 3) as 3 + 3 * 2 * `frequencies` channels: itself, then sin and cos of each coordinate times
    2^0 .. 2^(frequencies - 1), the frequencies of one coordinate together."""
    scales = 2.0 ** torch.arange(frequencies, device=points.device, dtype=points.dtype)
    angles = (points[..., None] * scales).flatten(-2)
    return torch.cat([points, angles.sin(), angles.cos()], dim=-1)


class ShapeDecoder(nn.Module):
    """The latent's field at query points: `field(tokens(latents), points)`; the latent's tokens are worked out once
    and asked about any number of points."""

    def __init__(self, latent_channels=64, width=1024, heads=8, layers=16, frequencies=8, point_channels=3):
        super().__init__()
        from .shape_net import Block

        self.frequencies = frequencies
        self.post_quant = nn.Linear(latent_channels, width)
        blocks = [Block(width, heads, use_self_attention=True, use_cross_attention=False, norm_eps=1e-5)
                  for _ in range(layers)]
        blocks.append(Block(width, heads, use_self_attention=False, use_cross_attention=True,
                            cross_attention_dim=width, cross_attention_norm_type="layer_norm", norm_eps=1e-5))
        self.decoder = nn.ModuleDict({
            "blocks": nn.ModuleList(blocks),
            "proj_query": nn.Linear(point_channels * (1 + 2 * frequencies), width),
            "norm_out": nn.LayerNorm(width),
            "proj_out": nn.Linear(width, 1),
        })

    def tokens(self, latents):
        hidden = self.post_quant(latents)
        for block in self.decoder["blocks"][:-1]:
            hidden = block(hidden)
        return hidden

    def field(self, tokens, points):
        """The field (batch, points) at `points` (batch, points, 3) in the shape's frame, positive inside."""
        queries = self.decoder["proj_query"](point_features(points.to(tokens.dtype), self.frequencies))
        queries = self.decoder["blocks"][-1](queries, encoder_hidden_states=tokens)
        # `proj_out` gives the logit of being outside (measured: the field the surface was found in until now is its
        # negative); the field is positive inside.
        return -self.decoder["proj_out"](self.decoder["norm_out"](queries))[..., 0]

    def forward(self, latents, points):
        return self.field(self.tokens(latents), points)


def decoder_settings(config):
    """`ShapeDecoder`'s settings from the weights' `vae/config.json`; refuses what this decoder does not build."""
    if config.get("embedding_type", "frequency") != "frequency" or config.get("embed_include_pi", False):
        raise ValueError(f"the VAE's point embedding is not the one built here: {config}")
    return dict(latent_channels=config["latent_channels"], width=config["width_decoder"],
                heads=config["num_attention_heads"], layers=config["num_layers_decoder"],
                frequencies=config["embed_frequency"], point_channels=config["in_channels"])


def load_decoder(folder, device="cpu", dtype=torch.float32):
    """The decoder from the weights' `vae` folder (its `config.json` and `diffusion_pytorch_model.safetensors`):
    every tensor but the encoder's loaded by name, none missing, none left over."""
    from safetensors.torch import load_file

    with open(os.path.join(folder, "config.json")) as file:
        model = ShapeDecoder(**decoder_settings(json.load(file)))
    state = load_file(os.path.join(folder, "diffusion_pytorch_model.safetensors"))
    loaded = {name: tensor for name, tensor in state.items() if not name.startswith(ENCODER_ONLY)}
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    if missing or unexpected:
        raise ValueError(f"the shape decoder does not match the weights: missing {missing[:5]}, "
                         f"left over {unexpected[:5]}")
    return model.to(device, dtype).eval()
