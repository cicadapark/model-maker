"""The shape model's own decoder, attention processor and point embedding (shape_vae), on small made-up weights: what
each computes, and that the weights load strictly. Their equivalence with the code they replaced was measured on the
real weights once (docs/objects.md, "The shape model's own decoder")."""

import importlib.util
import json
import math
import os
import shutil
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0
    from safetensors.torch import save_file

    from model_maker import shape_vae


def attention(dim, heads, cross_dim=None, seed=0):
    torch.manual_seed(seed)
    kwargs = dict(query_dim=dim, heads=heads, dim_head=dim // heads, bias=False)
    if cross_dim:
        kwargs.update(cross_attention_dim=cross_dim, cross_attention_norm="layer_norm")
    layer = Attention(**kwargs, processor=shape_vae.JoinedHeadsAttnProcessor()).eval()
    for parameter in layer.parameters():
        torch.nn.init.normal_(parameter, mean=1.0 if parameter.ndim == 1 else 0.0, std=0.2)
    return layer


def by_hand(layer, hidden, encoder=None):
    """Attention written out head by head: head i takes the i-th run of the joined projections' channels."""
    heads, width = layer.heads, layer.to_q.out_features
    head = width // heads
    query = layer.to_q(hidden)
    if encoder is None:
        joined = torch.cat([query, layer.to_k(hidden), layer.to_v(hidden)], -1)
        runs = [joined[..., i * 3 * head:(i + 1) * 3 * head].split(head, -1) for i in range(heads)]
    else:
        encoder = layer.norm_cross(encoder)
        joined = torch.cat([layer.to_k(encoder), layer.to_v(encoder)], -1)
        runs = [(query[..., i * head:(i + 1) * head], *joined[..., i * 2 * head:(i + 1) * 2 * head].split(head, -1))
                for i in range(heads)]
    outputs = [torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(head), -1) @ v for q, k, v in runs]
    return layer.to_out[0](torch.cat(outputs, -1))


@unittest.skipUnless(HAS_TORCH, "needs torch (installed by setup.sh)")
class TheAttentionProcessor(unittest.TestCase):
    def test_each_head_takes_its_own_run_of_the_joined_channels(self):
        hidden, encoder = torch.randn(2, 9, 32), torch.randn(2, 5, 12)
        with torch.no_grad():
            for cross_dim in (None, 12):
                layer = attention(32, 4, cross_dim)
                found = layer(hidden, encoder_hidden_states=encoder if cross_dim else None)
                torch.testing.assert_close(found, by_hand(layer, hidden, encoder if cross_dim else None),
                                           rtol=1e-4, atol=1e-5)

    def test_with_one_head_it_is_diffusers_own_processor(self):
        hidden, encoder = torch.randn(1, 7, 16), torch.randn(1, 4, 8)
        with torch.no_grad():
            for cross_dim in (None, 8):
                layer = attention(16, 1, cross_dim, seed=1)
                ours = layer(hidden, encoder_hidden_states=encoder if cross_dim else None)
                layer.set_processor(AttnProcessor2_0())
                theirs = layer(hidden, encoder_hidden_states=encoder if cross_dim else None)
                torch.testing.assert_close(ours, theirs)

    def test_a_mask_is_refused_not_ignored(self):
        layer = attention(16, 2)
        with self.assertRaises(ValueError):
            layer(torch.randn(1, 3, 16), attention_mask=torch.ones(1, 3))


@unittest.skipUnless(HAS_TORCH, "needs torch (installed by setup.sh)")
class ThePointFeatures(unittest.TestCase):
    def test_the_point_then_sines_then_cosines_of_each_coordinate_at_doubling_frequencies(self):
        point = torch.tensor([[0.3, -0.2, 0.7]])
        features = shape_vae.point_features(point, frequencies=8)
        self.assertEqual(tuple(features.shape), (1, 51))
        angles = [c * 2 ** k for c in (0.3, -0.2, 0.7) for k in range(8)]
        expected = [0.3, -0.2, 0.7] + [math.sin(a) for a in angles] + [math.cos(a) for a in angles]
        torch.testing.assert_close(features[0], torch.tensor(expected), rtol=1e-5, atol=1e-6)


SMALL = {"embedding_type": "frequency", "embed_include_pi": False, "embed_frequency": 2, "in_channels": 3,
         "latent_channels": 4, "num_attention_heads": 2, "num_layers_decoder": 2, "width_decoder": 8}


@unittest.skipUnless(HAS_TORCH, "needs torch (installed by setup.sh)")
class TheDecoder(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        torch.manual_seed(2)
        self.model = shape_vae.ShapeDecoder(**shape_vae.decoder_settings(SMALL))
        self.state = {name: torch.randn_like(tensor) for name, tensor in self.model.state_dict().items()}
        # The encoder's tensors are in the weights' file too, and are not built.
        self.state.update({"encoder.proj_in.weight": torch.zeros(8, 15), "quant.weight": torch.zeros(8, 8)})

    def tearDown(self):
        shutil.rmtree(self.folder)

    def write(self, state, config=SMALL):
        with open(os.path.join(self.folder, "config.json"), "w") as file:
            json.dump(config, file)
        save_file({name: tensor.contiguous() for name, tensor in state.items()},
                  os.path.join(self.folder, "diffusion_pytorch_model.safetensors"))

    def test_every_tensor_but_the_encoders_is_loaded_by_name(self):
        self.write(self.state)
        loaded = shape_vae.load_decoder(self.folder)
        for name, tensor in loaded.state_dict().items():
            self.assertTrue(torch.equal(tensor, self.state[name]), name)
        latents, points = torch.randn(1, 6, 4), torch.rand(1, 10, 3) * 2 - 1
        with torch.no_grad():
            field = loaded(latents, points)
            self.assertEqual(tuple(field.shape), (1, 10))
            torch.testing.assert_close(loaded.field(loaded.tokens(latents), points[:, :4]), field[:, :4])

    def test_a_missing_or_left_over_tensor_is_refused(self):
        missing = {k: v for k, v in self.state.items() if k != "decoder.proj_query.weight"}
        self.write(missing)
        with self.assertRaisesRegex(ValueError, "missing"):
            shape_vae.load_decoder(self.folder)
        self.write({**self.state, "decoder.extra.weight": torch.zeros(2)})
        with self.assertRaisesRegex(ValueError, "left over"):
            shape_vae.load_decoder(self.folder)

    def test_an_embedding_it_does_not_build_is_refused(self):
        self.write(self.state, {**SMALL, "embed_include_pi": True})
        with self.assertRaisesRegex(ValueError, "embedding"):
            shape_vae.load_decoder(self.folder)


if __name__ == "__main__":
    unittest.main()
