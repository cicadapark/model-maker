# The touch-up's fill

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## The touch-up's fill: counted painted, not judged right (10 November 2026)

The touch-up's first run on the card raised the candle's coverage from 85.7 % to 98.4 % and the chair's from 90.7 % to
95.4 %, and both passed every check. By eye the candle's dish was wrong: dark holes in its inner wall, pale grey,
mirror-like patches along its far rim, dark streaks under the candle. A fresh candle made the same way came out the
same, pixel for pixel (the seeds are fixed). The coverage counts whether a texel was painted, not what it was painted
with, so nothing caught it.

### Why the fills went wrong

Each touch-up view's input was rebuilt from the saved colours and its masks drawn over it (the views chosen depend only
on the six views' paint, so the rebuild is exact: the same pixel counts as the run recorded). Four causes, in order of
harm:

* **Painted from noise.** The inpainter ran at strength 0.99: the masked places started from noise and the picture
  under them was ignored. The colour spread to them from their painted neighbours was a fair brass; it was thrown
  away. What SDXL painted instead was what a dish "could" have: holes, and the grey backdrop carried in over the rim
  next to the mask (the far inner wall, seen from 50° above, borders the backdrop). This is the "generate" region of
  TEXTure and Text2Tex, which they use only where nothing is known; a place with a colour round it is their "refine"
  or "update" region, denoised from a partly noised start (Text2Tex's code: strength 1 for new places, 0.3 for
  updates). The SDXL inpainting model card warns that strength 1 lowers the quality too.
* **The whole object's words for a part of it.** The prompt was the object's ("a candle in a brass holder, high
  quality, even light"); nothing steered away from holes, reflections or the backdrop.
* **Context from the spread colours only.** The picture the inpainter saw was drawn in the colours spread to the mesh's
  corners, not in the painted colours: the paint round each gap was shown blurred, and a later view did not see an
  earlier view's fill (against what the module said).
* **Pixels counted painted that the inpainter never repainted.** SDXL's pipeline shrinks the mask to its latents (one
  cell for 8 by 8 pixels, nearest sampling: a cell is masked when its first pixel is). The mask was the unpainted
  pixels grown by 3, so a thin gap could fall between cells; those pixels came back as they went in, the spread colour,
  and were added to the colour cloud as paint. 3-5 % of the candle's filled pixels and 17-23 % of the chair's (along
  its legs' and spindles' edges) were so: the coverage rose by colour nobody painted.

### The fill check

The established way to judge an inpainted region without a reference is against the region round it (no-reference
inpainting measures locate the boundary and the inpainted region from the mask and compare their statistics; a fill's
colours should be found in its surroundings).
Here, each filled point of the colour cloud is compared with the painted points (the six views') nearest it on the
same surface: found by position and facing together, those facing within about 75° of it kept, the least CIE76 delta E
taken. A colour more than 20 from every one of them is foreign: it is found nowhere round the fill. "fill matches the
paint round it" passes when no touch-up view's fill is more than 10 % foreign and no joined foreign place in a view
(drawn back into its pixels) is larger than a square 3 % of the object's size across. The report draws the worst view's
fill green where its colour is found round it and red where it is foreign (`fill.png`).

| | candle (first run) | chair (first run) |
|---|---|---|
| foreign, each view | 27.9, 71.8, 18.0, 65.4 % | 2.4, 4.8 % |
| largest foreign place, each view | 15.4, 20.9, 7.0, 12.9 % | 1.4, 1.5 % |
| verdict | fails | passes |

The fresh candle made the same way reads the same. All 21 regression cases hold.

Without touch-up there is no fill and no fill check (the coverage judges it). The red falls on exactly the holes, the
grey patches and the streaks under the candle; the chair's few foreign pixels lie in threads along its legs' edges.
The measure was first run against the 128 nearest of all 1.02 million painted points (a minute for the candle); a
quarter of them, the 32 nearest, agree on 99 % of the filled points in 3 s.

### The fill itself

* **From the painted neighbours first.** The masked places start from the colour spread to them and are repainted at
  strength 0.6 (`Settings(touch_up_strength=...)`; 1 paints from noise as before).
* **The paint round it as context.** Each view is drawn in the colour cloud's colours as the bake mixes them (the six
  views' paint and the earlier views' kept fills), the spread colours only where the cloud does not reach.
* **The mask grown by a latent cell** (8 pixels, square), so every unpainted pixel's cell is repainted, and only pixels
  in a repainted cell are put back.
* **Words for a part.** "the same surface and material continued, even light" is added to the object's words, and
  "hole, opening, gap, dark spot, stain, reflection, mirror, glare, highlight, background" to what it is steered from.
* **Judged before it is kept.** Each view's fill is measured as the check measures it (stricter: delta E 15); when more
  than 2 % of it is foreign it is painted again from another seed and the better try kept; its pixels still foreign,
  and 2 round each, are left out. They keep the spread colour and count as unpainted, so the coverage stays true.

The first run's own fills, replayed through the judgement on the CPU (only the pixels its latent mask repainted, the
foreign ones left out; no new painting): for the candle 43.7, 79.2, 33.3 and 71.0 % of each view's fill was foreign at
delta E 15 and left out (with 2 pixels round), 49 569 of its 149 148 filled points kept; it reads 91.0 % covered
(passes) and passes the fill check. The holes, the grey patches and the streaks are gone from the turned model; some
paler brass flecks remain. The chair keeps 10 904 of its 18 307 (10.0 and 16.2 % foreign, and the threads its latent
mask never repainted gone) and reads 93.9 %. What the new strength, context and words paint is for the card to show.

The regression set keeps both: `candle-fill-foreign` (the first run's candle: it must fail the fill check and still
pass the coverage, which is the failure) and `chair-fill-alike` (it must pass both).

What did not work, or was not done: a likeness to the source picture (the dish's inside is barely in the picture);
naming each part's own material (the words name the object's; which part a gap lies on is not known); a dedicated SDXL
inpainting model, or diffusers' crop-to-the-mask option (its paste-back blends the whole crop); LaMa (Places, a
non-commercial dataset); painting the touch-up views at SDXL's own 1024 (the views and their choice are at 768).

### What the touch-up costs

On the 16 GB card the paint stage took 192-198 s an object without touch-up (the fix-check) and, with it, 228-231
s for the book and the chest (no view added: about 35 s spent choosing the views, 24 candidate views and the 32
directions of the seen-share, on the CPU), 241 s for the chair (2 views) and 269-280 s for the candle (4 views): about
35 s, and 6 to 12 s a view. The rest of the slower runs is elsewhere: the shape stage (the candle 87 s to 162 s, the
chest 167 s to 255 s) and the finish (37-55 s to 83-95 s, for every object, filled or not) both run the outline check,
whose view search now also tries three camera distances (the outline through a lens); that search is now six times
faster, with the same views and values (see [The outline's search made fast](outline-search.md)). The fill's judgement adds about
3 s a view and to the finish on the CPU; a retried view adds one more inpainting, and strength 0.6 runs 18 of the 30
steps.

## The fill judged on the model (11 November 2026)

Version 0.1.0's unseen run failed six of its ten objects on "fill matches the paint round it". Two of them, the rocking
horse and the boots, failed on nothing else and look right: the horse's fill is plausible wood, a little warmer than
the six views' paint; the inpainter painted the boot's lining and tongue another colour than its outside, and rightly.

### Why good models failed

* **The try was judged, not the model.** The check judged each touch-up view's fill as the inpainter painted it. But
  the touch-up leaves out what is foreign: those pixels never reach the model, which shows there the colour spread from
  round the place. The check failed a model for paint it does not carry.
* **"Found nowhere round it" is not "wrong".** The measure assumes an unseen place should look like its surroundings.
  That holds for a candle's dish; it does not for insides and undersides: a boot's lining, a shelf's boards, a tray, a
  leg's inner face.

### The fix: judge only what the model shows

* **The fill kept** is judged as kept, by the same measure as before (at most 10 % of a view foreign, no foreign place
  over 3 %). What the touch-up keeps is within delta E 15 of the paint round it, so this passes unless the touch-up
  kept everything (a record made before it judged its fill: the touch-up's first candle still fails, 18 to 72 %).
* **The places left unpainted** are judged on the finished model. From each touch-up view, the pixels it sees squarely
  that the finished colour cloud does not reach are joined into places (`touch_up.left_out_places`). There the model
  shows the colour spread from round the place. A place over 3 % of the object's size across is a smear when the
  paint round it is not one colour (its ring: the painted pixels within 4 of it, on the surface within three ring widths
  of it) and the colour the model shows in it is not one colour either. "One colour": at least 90 % of the colours
  within delta E 20 (the fill check's own "foreign") of their median. A place ringed by one colour passes, and so does a
  place that shows one colour though its ring holds two (a chair's post spread brown from brown wood and a grey
  highlight). Leaving a place out does not mend it: a smear still fails.
* **The fill as painted** is shown in the check's words and no longer counted. It is now the trigger for another seed:
  after its 2 tries, a view whose kept try was, as painted, more than 10 % foreign is painted from more seeds, up to 4
  (`MOST_FILL_TRIES`), the least foreign kept.
* The report's `fill.png` rings each place over 3 %: magenta a smear, blue not.

### The limit, measured

The trials hold 33 touched-up jobs (166 finished, copies counted); their places over 3 % were measured and each judged
by eye from crops of the finished model beside the inpainter's try (never on the release's unseen ten). Smeared by
eye: the fifth trial's candle's foot (8.1 % across; ring 61 % one colour, inside 69 %) and a second place on it (3.3 %;
63 %, 81 %), the 12 GB card's candle's foot (19.2 %; 79 %, 86 %), the fill trial's candle at strength 0.99 (17.1 %;
73 %, 76 %), and the fifth trial's chair's front stretcher (4.0 %; 82 %, 72 %, grey highlights and brown wood
spread into dark spikes). Their rings read 61 to 82 % one colour, their insides 69 to 86 %. Places whose ring holds two
colours but which show one by eye (the fill trial's chair's back post, 66 % and 100 %; a candle's foot spread a flat
brass, 70 % and 97 %; the remake's shield's handle, 81 % and 97 %) read 97 to 100 % inside. The limit, 90 % for both,
lies between. The ring alone did not separate them (the chair's post, no smear by eye, reads 66 %; the smears up to
82 %); weighting lightness by half (as CIE94 does for textiles) read every smear's ring at 87 % or more; keeping only
the ring pixels facing the place's way changed almost nothing.

| | ring one colour | inside one colour | verdict |
|---|---|---|---|
| fifth trial's candle, its foot | 61 % | 69 % | smear, fails |
| 12 GB card's candle, its foot | 79 % | 86 % | smear, fails |
| fifth trial's chair, its stretcher | 82 % | 72 % | smear, fails |
| clay jug, inside its mouth | 100 % | 97 % | passes |
| shield's handle, its inner face | 100 % | 76 % | passes |
| rocking horse, a rocker's inner face (unseen run) | 55 % | 100 % | passes |
| boots, the lining (unseen run) | 73 % | 100 % | passes |

### The trials finished again

All 166 trial jobs (125 on the 16 GB card, 41 on the 12 GB card) were finished again on the CPU by the code before and
after. 132 passed every check before, 135 after. Every candle copy still fails "fill matches the paint round it": the
fifth trial's (and its copies in the fill trial and the open-loops run) on its foot's smear, 8.1 % across; the 12 GB
card's (two copies) on its foot's, 19.2 %; the fill trial's at strength 0.99 on three smears; the touch-up's first
candle on its kept fill (it kept everything). The verdicts that changed:

| Job (copies) | Before | After | Is the new verdict right? |
|---|---|---|---|
| the clay jug of the open-loops run (2) | fails the fill (one place 4.7 % across as painted) | passes | Yes. The place, inside the jug's mouth, shows the clay spread from round it, ringed by one colour |
| the 16 GB card's remake of the round shield | fails the fill (3.3 % across as painted) | passes | Yes. The handle's inner face shows its iron |
| the fifth trial's chair (also in the fill trial at strength 0.6) | passes | fails the fill: a smear 4.0 % across | Yes, a small fault: the front stretcher's top, seen from above, shows grey patches and dark brown spikes where the wood should run on. The as-painted check passed it, as the inpainter's own try was brown wood there; what it left out is what the model shows |
| the unseen run's rocking horse | fails the fill | passes | Yes. Its places, on the rockers' and legs' inner faces, show flat wood |
| the unseen run's boots | fails the fill | passes | Yes. The lining shows the leather's brown, in two shades where the kept fill meets the spread |
| the unseen run's well, bookshelf and wheelbarrow | fail the fill and more | fail the rest | The well's inside shows dark stone (inside 90 % one colour: at the limit); the bookshelf's boards and the wheelbarrow's tray show wood. They still fail the symmetry, the coverage, the outline and the holes |

The release's unseen ten were only measured: no limit was set or moved on them. The basket still fails the fill (six
smears in its weave) besides its outline, holes and coverage.

### On the card

The candle and the chair were made again on the 16 GB card with this code. The candle's two views from 35° above were
each painted from 4 seeds (their best tries 14 and 17 % foreign as painted, so still over 10 %), the two from 50° from 2;
it fails "fill matches the paint round it" on two smears at its foot (10.8 % across: ring 85 %, inside 83 %; 4.9 %:
84 %, 74 %). Its paint stage took 353 s, about 75 s more than with 2 tries. The chair's views were each painted twice
(3.4 and 5.0 % foreign as painted); it passes every check, its paint stage 252 s.

### Known limits

A smear joined to a large place of one colour reads as one colour: a made-up candle whose whole dish was left out with
its foot (one place 45 % across) read 100 % for both. The trials' places were each on one or two surfaces; a fill left
out whole leaves the coverage to fail it. The kept fill passes by construction for a judged touch-up, as before:
what the inpainter painted and the touch-up kept is within delta E 15 of the paint round it.

The tests: a cup whose inside the inpainter paints as a lining (left out, the model showing the cup's colour) passes; a
candle whose foot's corners took wax and brass by turns, a glass cup painted round it and left out, fails as a smear;
the as-painted share is shown, not counted, and the coverage of a model whose fill was all left out is the coverage
without the touch-up. The regression set adds `candle-glass-cup-smear` (must fail), `jug-mouth-left-out` and
`shield-handle-inner-face` (must pass).
