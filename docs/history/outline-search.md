# The outline's search made fast

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## The outline's search made fast (10 November 2026)

The outline through a lens made the pipeline slow: the view search drew the model from 4 camera distances, 13 turns and
3 tilts and refined round the best, 180 silhouettes where it had drawn 63, for every shape try and again in the finish.
A whole model went from 6-7 to 9-11 minutes.

Measured first (a profile of the chest's judgement, 60 000 triangles): of 42 s, 41.3 s went to drawing the 181
silhouettes (0.23 s each), 0.4 s to the edges' distance transforms, the hulls and the thickest pieces, and under 0.5 s to
framing the outlines and their overlaps. The drawing was our rasteriser's full work: for every pixel of every triangle
its depth, its nearest triangle (a sort) and its place in it, none of which an outline needs.

So a silhouette is now drawn by its own small routine (`render.covered`): each triangle is cut by each pixel row's middle
line into a span of columns, with the same test for a pixel's middle inside a triangle (its slack included), and the
spans are summed along their rows. Every triangle's rows at once, nothing per pixel, no depth. It draws the same pixels:
on 420 views of five surfaces of the regression set and the trials (50 000 to 60 000 triangles, each search's turns,
tilts and distances) not one pixel of 17.7 million differs, and a test holds it to that on known shapes (thin parts,
large triangles, through a lens). It is 5 to 14 times faster (a sword, thin, gains least). Besides: the search keeps the
two outlines of the best view, so the edges' distances draw none again, and a view met twice in the refinement (a tilt
held at 0) is drawn once. The distance transforms were already made once a judgement and cost under 0.3 s; they were
left as they were.

On every shape try and every finish of the regression set, the fix-check and the third trial's three objects (51
judgements), the view found (turn, tilt and camera distance) and every value (the edges' mean, their 95th percentile, the
thickest piece in a gap) are the same to the last digit, and so is every verdict: the search compares the same pixels
in the same order (`tools/trial/outline_bench.py` records each judgement's view, values, verdict and seconds, to run
before and after any change to the outline check). Some of them (mean, 95th percentile, thickest piece in a gap; the
view):

| Case | Judged | Outline, before and now | Verdict |
|---|---|---|---|
| the third trial's chest | shape try 42 | 1.74 %, 5.73 %, 7.3 %; -14° round, 0° above, lens 6 | fails, as before |
| the third trial's chest | shape try 43 | 0.67 %, 1.82 %, 2.6 %; +23° round, 3° above, lens 6 | passes |
| the third trial's chest | finish | 0.68 %, 1.82 %, 2.6 %; +23° round, 3° above, lens 6 | passes |
| the third trial's candle | shape, finish | 0.51 %, 1.58 %, 2.6 %; -38° round, 7° above, lens 6 | passes |
| the third trial's book | shape, finish | 0.58 %, 1.65 %, 3.6 %; 0° round, 0° above, lens 3 | passes |
| the fix-check's jug | shape, finish | 0.95 %, 3.12 %, 15.1 %; +2° round, 0° above, lens 6 | fails (a web inside its handle) |
| table-web | finish | 1.10 %, 4.33 %, 11.5 %; +28° round, 0° above, lens 6 | fails, as it must |
| chair-good | finish | 0.44 %, 1.07 %, 2.6 %; -1° round, 7° above, straight on | passes, as it must |
| sword-good | finish | 0.09 %, 0.26 %, 1.0 %; +30° round, 0° above, straight on | passes, as it must |

The seconds (one judgement at a time on a desktop's CPU; the card's laptop takes about 1.8 times as long, from its
finish growing 45 s for 117 more silhouettes):

| Object | A judgement before | Now |
|---|---|---|
| the chest (each of its two shape tries and the finish) | 39.4-39.6 s | 5.2-5.3 s |
| the candle (its shape try and the finish) | 38.9-39.3 s | 5.8-6.2 s |
| the fix-check's chair | 29.0-29.1 s | 4.9-5.0 s |

All 51 judgements, eight at once: 2049 s before, 377 s now. On the laptop a judgement goes from about 70 s to about
10 s; before the lens it was about 24 s (63 silhouettes the old way). So each shape try and the finish now spend less on
the outline than before the lens: the chest's shape stage (two tries) should come to about 135 s (167 s before the
lens, 255 s with it), the candle's (one try) to about 100 s (87 s, then 162 s), and the finish to about 25-35 s (37-55
s, then 83-95 s). The candle's shape stage grew 75 s with one try, more than its one judgement's 45 s; what else grew
there is not the outline. These laptop figures are worked out, not measured.

What did not pay: the coarse grid drawn at half the size, its best six drawn again at full size, then refined. It saved
another 40 % of the search (2 to 3 s a judgement), but found another view in 10 of the 51 judgements (the candle's six,
the barrel, a jug, a longbow): a coarse grid at half the size ranks close views differently, and the refinement then
climbs another hill, a little higher (the candle 95.80 % against 95.79 %). The values would have moved for 2 or 3 s; the
search is kept whole and exact.
