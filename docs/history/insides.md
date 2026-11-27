# Insides, bake-only

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## Insides, bake-only (11 November 2026)

0.2.0's sealed run's flower pots, bucket and cauldron, and the trials' candle dishes, jugs, wheelbarrow and well, show
grey or dark patches on their inner walls. The diagnosis (`docs/trials/2026-11-11-030-diagnosis.md`) found two
sources: the colour the view from above gave the corners of a wall it sees only at a slant, past the rim, and the
touch-up's fill there. This item was time-boxed to two days and bake-only: what the stored paint allows without
painting again. The answer is: very little, and nothing was changed.

### What texturing from photographs does with places seen badly

* **MV-Adapter's own projection** (its `uv.py` and `pipeline_texture.py`, at the commit the registry names): a texel is
  valid for a view only above a facing of 0.2 and away from depth edges (depth gradient under 0.1, grown by 5 pixels);
  the valid views are blended by their facing cubed; the texels no view validly sees are filled from the valid ones
  round them in the texture (`inpaint_cvc`, its UV padding), not from a slanted view.
* **mvs-texturing** (Waechter, Moehrle and Goesele 2014): one view per face, chosen by how large and sharp the face is
  in it (its gradient magnitude integral), so a face seen at a slant loses to a square view; the faces no view
  textures go to its hole filling (on by default), not to a slanted view.
* **Paint3D** (Zeng et al. 2024): the views a 2-D model paints carry its lighting, shadows and reflections; the
  lighting is taken out afterwards by a model trained on textures without it (UVHD). It is a new model whose weights'
  licence is unchecked, and it is deferred.

### Four ways measured

The bake uses the colour cloud (a view's pixels where it sees the surface at a facing of 0.55 or more); wherever the
cloud does not reach, it shows the corners' own colours, which take a view's colour from a facing of 0.2. So a corner
the cloud does not reach was seen only at a slant, or not at all. On the stored paint of the 51 distinct vessels and
open objects of the trials (both cards' runs, the 0.1.0 unseen run's well, wheelbarrow, basket, cooking pot and
bookshelf; `tools/trial/insides.py`), those corners were coloured four ways:

| Way | Those corners take | Coverage |
|---|---|---|
| as made | the slanted view's colour | as made |
| 1 | the colour spread from the corners the cloud reaches round them (`paint.spread`), as both methods above fill what no view sees well | unchanged on all 51 |
| 2 | the same, never spread across a crease (normals more than 45° apart: a wall's colour not from its floor) | unchanged |
| 3 | way 2 spread from the six views' paint only, the touch-up's fill left out where they did not paint | falls by up to 8.7 points (the bucket 93.1 % to 84.4 %); unchanged where nothing was touched up |

**Labelled by eye, blind** (102 pictures, each object as made and by way 1 from 55° above, shuffled under codes and
labelled before the key was opened): 17 objects smeared inside as made, 12 by way 1. The six that came clean are all
jugs: the pale streaks inside their necks, which only the view from above sees and only at a slant, become even clay.
One changed the other way (the 12 GB card's cauldron, "faint streaks" to "mild patches"), but its two pictures are the
same to the eye side by side: label noise. The 11 still smeared are the ones this item was for: both flower pots, the
bucket, the 16 GB card's cauldron, four candles' dishes, the wheelbarrow's tray, the well and an open chest's inner
wall.

**Why the smears stay.** Drawn where the cloud reaches, the flower pot's inner wall is almost all unreached, but its
reached islands are the touch-up's kept fill, and grey; the view from above painted even the part of the wall it sees
squarely grey, and the well's inner wall pale. The shadow is in the paint, not in the bake's choice of it (Paint3D's
point). Way 3, which leaves out the touch-up's fill, takes some of the bucket's dark smears away, but the walls stay
grey from the six views' own paint, and the coverage falls under its 90 % limit on objects right by eye: the cauldrons
from 95.4 % and 95.5 % to 87.4 % and 87.5 %, a new refusal each.

**Way 1 refuses seven objects more.** Judged on all 217 stored jobs (`tools/trial/rejudge.py`, the finish's checks with
way 1's colours), no verdict changes from fail to pass and seven change from pass to fail, all "fill matches the paint
round it": the well, the bookshelf, both lamps, the 12 GB card's flower pot, cauldron and birdbath. A place left
unpainted whose ring holds two colours showed one colour as made (the slanted view's) and now shows both, spread ring by
ring into streaks; the birdbath's strip, pale and even as made, is pale with dark streaks. The check reads that as a
smear, and by eye it is one. So way 1 trades six jugs' necks for seven new refusals, and is not kept.

**The bookshelf's back** is plain and even in its stored paint, as made and every way; it did not reproduce.

**What would work** is outside a bake from stored paint: paint without lighting (Paint3D's UV model, licence to check),
or a touch-up that paints an inner wall from its rim's own colour and material and is judged against the rim, not
against the shadowed paint round the place. Both need the card.
