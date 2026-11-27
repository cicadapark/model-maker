# Defects found by measurement

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## Defects found by measurement

The pipeline's checks, records, robustness and licences were gone through as a whole, and 23 defects found, each
measured on made-up shapes with a known answer or found by reading, and 15 plausible ones. Every defect is fixed below,
each check fix with a test built as it was measured; every plausible one is confirmed and fixed, or shown not to
apply. No existing limit was loosened. Where a fix adds a measure, its limit was chosen from measurements: every model
the trials rightly passed still passes it, and the made-up faults fail it. The made-up shapes are in the regression set
(`made_up` cases, built by code from `python/tests/shapes.py`).

### The checks

1. **The outline's view search.** The coarse grid ranked the views by their overlap, and only the best was refined, at
   its camera distance. A thin part a few pixels off its place overlaps nothing, so the overlap is a needle: three
   degrees from a bench's true view it falls from 0.94 to 0.5, four degrees of tilt to 0.15, and a grid 15 degrees
   apart never finds it. A bench drawn from its own surface (10 degrees round, 12 above, straight on) was matched
   through a lens 3 sizes away (overlap 0.74; edges 2.6 % and 9.4 % apart) and failed. The views are now compared by the
   check's own measure, the edges' mean distance both ways (chamfer matching: Barrow et al. 1977, Borgefors 1988), which
   grows smoothly as an edge moves; the best view at each camera distance is refined, each step size repeated while it
   gains (a pattern search), and the best refined view kept. Tried on 16 views of the made-up bench and three-legged table drawn at random (turn -60 to 60,
   tilt 0 to 35, every lens), how many each search passes (bench, table): by the overlap, each distance's best refined,
   14 and 15; each distance's best three refined, 15 and 14; five tilts in the coarse grid, 14 and 14; by the overlap of
   outlines grown 4 or 8 pixels with the pattern search, 14 and 15 of the bench; by the edges' mean distance, each
   distance's best refined, 15 and 15; with the pattern search, **16 and 15** (kept; the miss: the three-legged table seen
   level at -55 degrees matched at -45, its 95th percentile 6.8 %, a near-copy view of a three-fold shape). The made-up
   chair passed all 16 with every search by the overlap. The search draws about 290 outlines where it drew 180 (see "What it costs" below). The carving
   finds its camera with the same function, its hole's inner part left out of the count (simplification 1).
2. **Left-right symmetry.** The mean diluted a local fault. Now the 98th percentile of the mirrored surface's distances,
   measured to the surface's triangles (no sampling floor), is held to **5 %** of the object's size besides the mean's
   2 %. Measured (98th percentile; the mean beside it):

   | Shape | 98th | Mean | Verdict |
   |---|---|---|---|
   | made-up chair, a leg missing | 23.5 % | 1.26 % | fails (passed before) |
   | made-up chair, an armrest on one side | 17.1 % | 1.18 % | fails (passed before) |
   | made-up chair, a leg 8 cm out | 9.6 % | 0.95 % | fails (passed before) |
   | made-up chair, a leg 4 cm out | 5.6 % | 0.64 % | fails |
   | made-up chair, as built | 1.25 % | 0.46 % | passes |
   | the 50 mirror-symmetric made jobs of the trials that pass every check (of 77 mirror-symmetric, both cards) | at most 3.49 % (a book; tables 2.68 %, shields 1.96 %, stools 1.89 %, the rest under 1 %) | at most 1.14 % | pass |
   | the 12 GB card's book built open in two lobes, the fix-check's book in two lobes (both fail other checks) | 4.83 %, 3.61 % | 1.47 %, 1.18 % | pass this check |

   The 95th percentile, tried first, does not see a missing leg: a chair's leg is under 5 % of its
   surface, and at the 95th the chair with a leg missing read 1.25 %, as a whole one. At the 99th it reads 33.6 %, but
   the good books read up to 5.0 %; the 98th leaves the widest gap between the worst good model (3.49 %) and the least
   fault asked for (9.6 %), and the limit lies in it.
3. **Symmetric across the shaft.** Mirrored points were measured to a cloud of samples, whose own spacing (about 1 mm on
   a 3 m spear) over a 3 cm width failed a perfectly round spear (3.57 %). They are now measured to the surface's
   triangles (`mesh_tools.SurfaceDistance`: the triangles under each point's 8 nearest samples, point to triangle), the
   limit unchanged at 3 %. Made-up shapes: perfectly round spears 3 m by 3 cm 0.51 %, 3 m by 4 cm 0.44 %, 2 m by 3 cm
   0.44 %, a quarterstaff 1.9 m by 3.5 cm 0.27 % (pass); a spear with a lug 2 cm out on one side 5.05 %, 5 cm out 7.32 %
   (fail). The trials' swords and spears 0.10 to 0.12 %, double-bitted axes 1.61 to 1.81 % (all pass).
4. **Levelling a leaning object.** A table turned whole by 3 degrees was trimmed 21.7 mm and passed. The levelling now
   reads the body's up from its flat top (the area-weighted normal of the faces within 10 degrees of its own up or of
   the up its feet settle it to; the established cue of a man-made thing's upright, Fu et al. 2008): when the top is
   square to the floor the feet settle it on and at least 1 degree off its own up, the whole object leans, no foot is
   short, and nothing is trimmed: it stays a failure. A thing with no flat top (at least 3 % of its surface) gives no
   evidence and is levelled as before. Made-up: the three-legged table turned 2, 3 and 4 degrees about either axis is
   left as made and fails; with a leg 2.5 cm short under a level top it reads 0.0 degrees as made, 3.4 settled, and is
   trimmed.
5. **The fill check passed by construction.** The touch-up keeps only fill within delta E 15 of the paint round it and
   the finish judged what was kept at 20. The check now judges each view's fill as it was painted, before what was
   foreign was left out: the touch-up records it (`as_painted`: the share foreign at 20 and the largest foreign place),
   and for a record made before that, the finish judges each view's saved try again, its pixels drawn again from the
   colour cloud as it stood before that view (the same drawing the touch-up made). A record from before the touch-up
   judged its fill (it kept all it painted) is judged on what it kept, as before. (Changed on 11 November: the
   as-painted fill is shown, not counted, and what was left out is judged on the model; see [The fill judged on the
   model](touch-up-fill.md).)
6. **A speck failed an exact shape.** The outline and the carving judged the raw cut-out; they now judge the cut-out the
   shape model was given (`cutout.for_shape`: its largest piece, its pinholes filled). The exact chair with a speck of 1 %
   of its cut-out above it passes (95th percentile under 1 %); before, 5.7 %.
7. **Wider than 6:1.** Outlines wider than the canvas were cut to it, and a 10:1 plank pictured from the front "matched"
   a view of it nearly end-on (overlap 1.000). The canvas is now as wide as the picture's outline needs, with 25 % to
   spare (`canvas_aspect`), the model's outline is drawn on a square that keeps the plank's height in pixels, and a model
   whose outline from the view found is wider still fails. The plank matches its own picture from the front; a plank 30 %
   shorter fails. No object of the trials is that wide, so none moved by this.
8. **Shape tries judged without the levelling.** The shape stage, the finish and the regression set now judge a shape by
   one function (`inspection.judged`: repaired, carved, laid, levelled, checked; simplification 2).
9. **The grip on the blade.** A blade's grip is now the stretch thin enough to hold that meets the guard (the widest
   slice) on its shorter side, at most 30 cm long, kept at the guard's end; and a "grip" over half the length of a sword or knife
   fails ("it is the blade"). The made-up rapier, its blade 2 cm by 1 cm, finds its 12 cm grip and its tip at the point;
   its blade alone fails "grip found".

### The words

10. **A creature's word before an -ed or -ing word.** "-ed" and "-ing" end a participle only when what is left holds a
    vowel (the Porter stemmer's condition, its step 1b: "spiked", "sleeping"; not "red", "sled", "shed", "king"), and a
    participle followed by a thing names the thing ("a wolf headed staff", "a king sized bed", "a giant spiked mace").
    "a hand guard" (and arm, shin, knee, neck, mouth, finger, fire and cross guards) is an object. A prompt check in
    front of the pipeline reads them the same way when it passes the shared cases (`python/tests/prompt_cases.json`),
    and eleven such prompts are in them with five creatures that must stay characters or animals ("a giant", "a
    frost giant", "a dragon sleeping on a chest", "a knight kneeling"; the animal "a horse grazing").
11. **"Never opened".** A negation up to three words before "open", "opened" or "ajar" (or before a word that makes a thing
    not symmetric) denies it: "a book, never opened" and "a chest that cannot be opened" are shut and symmetric. The words
    that stand a thing up are whole words: "a ribbon edged plate" lies.

### Records and reproducibility

12. **One folder, one order.** A job's folder is its slug and a hash of its prompt as read and its options
    (`pipeline.job_folder`); a folder that holds another order's job is refused (`JobMismatch`), and the same prompt twice
    in a batch is one job.
13. **The ledger.** A refinish keeps the settings the job was made with (kept in `job.json`); a prompt refused by `make` or
    the warm process has its own "not made" line; a failed model load writes "aborted" for every job waiting on it; in a
    batch a job whose stage raises is set aside and the rest go on, and the batch still writes its `run` line and
    `times.csv` (docs/metrics.md).
14. **The unseen headline** counts each job's first finish, and shows every unseen batch with its versions, first and
    last.
15. **The random state and the noise.** Each job's stage starts from the one random state, in a batch as in `make`. The
    shape model's, the view painter's and the inpainter's noise is drawn on the CPU (`shape_net.start_noise`; diffusers
    draws on the CPU when given a CPU generator), so a seed starts from the same noise on the 12 GB and the 16 GB card
    (on the card it depended on the card: torch lays out its CUDA draws by the card's multiprocessors). The cards' own
    arithmetic still differs a little. **Measured** on eight prompts: both cards chose the same picture seed for 7 of
    8, and those pictures differ by 0.4–0.8 of 255
    on average per pixel (0.2–2.1 % of pixels by more than 10), which cannot be seen. The shape model can amplify such
    a difference: from those near-identical pictures the 12 GB card's axe failed "haft straight" (the haft check's
    misreading, fixed since) and the 16 GB card's lantern failed "one main piece" (a real 16 cm loose part at its top), while the other
    card's passed. So the same seed gives the same picture to the eye on both cards, and close, not identical, models;
    every model is judged on its own.

### Robustness

16. **A kill mid-write.** Each stage writes its record first and its marker file last, each through a file beside it moved
    into place (`pipeline.written`): a killed stage leaves no file that looks finished.
17. **The touch-up's numbers.** Each view's points are numbered by its place in the record; a cloud made before is put in
    the record's order.
18. **Without SigLIP 2** a thing with a lid, a cover or doors is "not made: needs SigLIP 2", and the rest of a batch is
    made.
19. **Odd prompts.** An empty prompt (nothing left once read) is not made; a slug is at most 60 characters.
20. **Offline.** `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are set to 1 by the `model-maker` command before Python starts
    (and by the package's `__init__`, before any of the hub's libraries is imported, so for every way in: the command,
    setup's smoke test, the trial tools, a program that imports the pipeline), whatever the environment held: a 0 left
    there is overwritten, so no configuration can fetch code from the hub (0.3.1).

### Licences

21. **What the wheels bundle.** The licence check lists every shared library a package bundles and compares it with an
    allow-list; 93 in the locked environment, 90 allowed by their licences, and **libquadmath** (LGPL-2.1-or-later, in
    numpy's and scipy's wheels) allowed as the project's one recorded exception (CONTRIBUTING.md, rule 1; NOTICE;
    docs/setup.md, "What the wheels bundle").
22. **TripoSG's code.** None of TripoSG's repository is fetched or run: its folder of code holds the text of the
    Tencent Hunyuan FlashVDM Community License, and the repository's `NOTICE` names code derived from HunyuanDiT and
    FlashVDM. Its weights run with our own code ([The shape model's own decoder](shape-decoder.md)).

### The trials finished again

Every made job the trials kept (145: 104 on the 16 GB card, 41 on the 12 GB card, copies of a job counted each time
they were kept) was finished again on the CPU, by the code before these fixes and by the code after them, and their
verdicts compared. 124 passed every check before, 119 after; 7 verdicts changed, all for the fill and the outline:

| Job (copies) | Before | After | Is the new verdict right? |
|---|---|---|---|
| the fifth trial's candle (also in the fill trial at strength 0.6 and the open-loops run) | passes | fails "fill matches the paint round it": its second touch-up view 16 % foreign as painted | Yes. As painted (the touch-up's own tries, kept beside the job), the inpainter put a glass cup round the candle's foot and grey-green reflections in the dish; the touch-up left out what was foreign and kept the rest, which the old check then passed by construction. What it kept was painted by the same fill |
| the fill trial's candle at strength 0.99 | passes | fails: 64 % foreign as painted | Yes: the fill from noise, the case the fill check was made for |
| the 12 GB card's candle (fourth trial, and a later run) | passes | fails: 27 % foreign as painted | Yes, as the fifth trial's |
| the 12 GB card's three-legged stool (first trial) | fails "outline matches the picture" | passes | Yes. The old search settled on a view straight on at 58 degrees, where the back leg sticks out of the picture's outline; the new search finds the stool through a lens 3 sizes away (0.37 % mean, 0.78 % at the 95th percentile), every leg in place. The shape is unchanged |

The clay jugs whose touch-up painted a shadow behind the handle (7.2 % foreign as painted, its place left out) pass, as
the share is under the limit and a place the touch-up left out is not on the model. No verdict changed for the
symmetry, the levelling, the shaft, the grip, the carving or the new holes check: the faults they now catch are not
among the trials' kept models.

What it costs: a finish (one thread each, on a desktop's CPU) took a median of 28.2 s before (8 at a time) and 32.4 s
after (4 at a time beside other work, so not a clean comparison): the view search draws about 290 outlines where it
drew 180, each compared by its edges, and a picture with a hole to judge adds the holes check's own search.

### The plausible ones

| Item | Verdict |
|---|---|
| Carving a real thin panel | Confirmed and fixed for an opaque panel: a hole whose colour in the picture lies more than 10 (delta E) from the backdrop's is a part the cut-out missed, never carved and never judged open (the made-up mirror, its glass the cut-out missed, reads 63, every real hole of the trials at most 2.2). A clear pane shows the backdrop through it and cannot be told from a hole by colour; the lanterns' small panes are too small to be judged (see [Holes the picture shows](holes.md)) |
| Length of a sword or spear | Confirmed (a box's largest side was scaled to the length: a spear at 45 degrees came out 1.41 times too long) and fixed: a thing whose size is its length is set to it along its principal axis. Test: a spear made at 45 degrees is 2.0 m |
| Wrong paint passes | Confirmed for a black view (the coverage counts a texel painted whatever its colour): a new check, "painted views not black", fails a view more than 25 % black; on the trials' 654 views the most was 1.7 %. Since 0.3.0 only the view's backdrop (a band round its edge) is counted: a broken decode paints the backdrop black too, while a black thing is black only where it is (a smartphone and a laptop, 37-40 % of a view, were refused as broken; of 470 stored objects none had a black backdrop). A view of near-uniform colour is not judged: the end views of a spear are near-uniform by right |
| Parent-death signal | Does not apply to the pipeline: Linux sends the signal when the thread that started the process ends, so a program that starts `make` (and names itself in `MODEL_MAKER_WORKER_PID`, parent.py) must start it from a thread that lives as long as the program |
| State verdicts near the limit | Fixed: SigLIP 2 judges in 32-bit floats on the card too, as the regression does on the CPU (1.6 GB on the card, a few pictures a job) |
| Memory with every model kept | Confirmed: a process that kept every model loaded between jobs measured 44.1 GB at its peak; a run that loads each stage's models alone (`make`, `batch`) needs the memory in `docs/setup.md` |
| Capping a cut foot ring | Confirmed and fixed: a loop inside another (a ring's inner edge) is bridged to it and the two are capped as one ring (ear clipping of a polygon with holes, Eberly 2002). Test: a cut foot ring has no cap across its hollow |
| Square hafts | Confirmed and fixed: a slice's width is the mean of its widths in 8 directions (a round part's diameter; a square's the same however turned). Test: a 4 cm square haft turned 0, 30 and 45 degrees finds its grip, its width within 3 mm |
| Shadow check wording | Confirmed: the condition that the edge runs through a shadow was measured and never used. The check is kept as it is (the stricter one) and its words now say what it does |
| Load-time gaps | Fixed: every model folder's `__pycache__` is removed before its check, and model-maker writes no bytecode; MV-Adapter's code is checked to be at its commit, unchanged, before it is imported |
| Regression data | Fixed: `regression.py sums` writes every data case's (and labelled picture's) sum into `tools/trial/regression-sums.json`, which `run` checks; `run --only` with a name no case has stops with an error |
| Wasted card time | Fixed: a job none of whose pictures passed its checks ends there with its job line, not after a shape and a paint |
| Version string | Fixed: a tarball made by `git archive` (a release download) carries its commit in `python/model_maker/version.txt` (`export-subst`) |
| Likeness canvas | Fixed: the likeness frames the picture and the model as the outline does (simplification 3) |
| Setup | Fixed: a source folder without `.git` is made again; CI's licence job installs exactly the lock |

### The simplifications

1. One view search (`object_checks.view_search`) for the outline check and the carving.
2. One judging function (`inspection.judged`) for the shape stage, the finish and the regression set, and one way the
   paint follows the judged shape (`Judged.paint_followed`).
3. One framing (`object_checks.placed`) for the outline, the carving and the likeness.
4. One Fibonacci lattice (`mesh_tools.fibonacci_directions`) and one corner joining (`mesh_tools.joined`).
5. Dead code removed: `mesh_tools.bounds`, `cutout.cut_out`, `Cutter.to`, `render.vertex_colour_lookup`, the shadow
   check's unused `touching`, `object_checks.outline_overlap`; the search keeps only the best outline, not every one.

Not done: one plane cut for the carving and the levelling. Both clip triangles by a field, but the carving records
which face of the web each cut edge was on and the levelling snaps its cut onto the plane; one function would take both
as options, and the two are kept apart.
