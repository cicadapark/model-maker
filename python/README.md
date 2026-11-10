# python

The Python side of the pipeline (`model_maker`), run by the `model-maker` command at the
repository's root:

| Module | What it does |
|---|---|
| `words` | how a prompt is read: what kind of 3-D object, which checks apply, its usual size, the picture's words |
| `painter`, `cutout`, `picture_checks` | the picture tries (FLUX.1 [schnell]), the cut-out (BiRefNet), the picture checks |
| `shape_net`, `shape_vae`, `simplify` | the shape (TripoSG's weights with our own denoiser and decoder), brought to its triangle budget |
| `paint` | six views painted to fit the shape (MV-Adapter), back onto it |
| `object_checks`, `weapon_checks`, `inspection` | the measured checks |
| `carving` | a web the shape model put in a hole the picture shows (a jug's handle) cut away by the picture's silhouette |
| `render`, `bake`, `glb`, `report` | our rasteriser, the texture bake, the `.glb` with its marks, the report page |
| `pipeline` | the stages and each object's folder |
| `registry` | the models' licences and file sums, checked before loading |

`tests/` holds the tests of everything that runs without a card:
`cd python && python -m unittest discover -s tests -t .`
