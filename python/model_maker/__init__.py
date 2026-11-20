"""model-maker: make a checked 3-D object from a text prompt (docs/objects.md)."""

import os

# Every model is read from its checked folder, never fetched: set when the package is first imported, before any of
# the hub's libraries is (they read it then), so it holds for every way in: the command, setup's smoke test, the
# trial tools and a program that imports the pipeline. Set, not defaulted: a value of 0 left in the environment would
# let a model's configuration fetch code from the hub (docs/objects.md, "Offline").
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
