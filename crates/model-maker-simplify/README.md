# model-maker-simplify

Brings the shape model's dense surface (often a million triangles or more) down to a triangle budget with
meshoptimizer's edge collapses by quadric error, so the shape is kept. The vertices kept are the dense
surface's own, so their colours can follow.

```
model-maker-simplify dense.mesh kept.indices --most 60000
```

`dense.mesh` is `MMSH`: a version (1), the vertex and triangle counts, the positions (x, y, z) and the
triangles, all little-endian 32-bit (`u32`, `f32`). `kept.indices` is `MMIX`: a count and the kept
triangles' corners as indices into the dense surface's vertices. The Python side writes and reads both
(`python/model_maker/simplify.py`).

Loose pieces are kept, not pruned: the object checks list them.

Where the edge collapses stop above the budget, meshoptimizer's sloppy simplifier (a grid clustering, which ignores
the topology and can step a thin part's outline) takes over. The command says which ran: `59986 triangles kept of
1203440 by edge collapse`, or `by grid clustering (the edge collapses stopped above the budget)`. In our tests and
trials the edge collapses always reached the budget, many small pieces included (meshoptimizer removes whole pieces).
