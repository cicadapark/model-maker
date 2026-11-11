# 3-D objects

Release 1 makes 3-D objects, not characters: "an old looking chair", "a rusty iron lantern",
"an old iron sword", "a double-bitted battle axe". Characters come back in release 2.

```
./model-maker make "an old looking chair"
./model-maker make "an old iron sword" --size 0.9
```

Each object gets a folder with its `.glb`, its sidecar (`<name>.provenance.json`: the prompt, every
model used with its version and licence, the seeds, every check) and its report
(`report/report.html`).

This document gives every check with its limit, and what each limit and fix was measured on. The measurements come
from the release's trials: prompt sets made end to end on a 16 GB card (an RTX 3080 laptop card) and a 12 GB card (an
RTX 3060), named by their order ("the first trial", "the third trial", "the fifth trial", "the 12 GB card's fourth
trial"), and the "fix-check", the ten regression prompts made again after the third trial's fixes. The notes of the
first two are in `docs/trials/`; the later trials' notes, their made objects and the regression set's data are not
published (they are large, and hold machine-specific records): what they showed is given here.

## The steps

| Step | What runs | On |
|---|---|---|
| 1. Picture | FLUX.1 [schnell] paints four tries (another four if none passes); each is cut out by BiRefNet and checked, and SigLIP 2 judges that it shows the thing asked for (and, for a thing with a lid, cover or doors, shut or open as asked); the best that passes is kept | card |
| 2. Shape | TripoSG's weights, run with model-maker's own code, make the shape from the cut-out picture; it is brought to 60,000 triangles (`model-maker-simplify`), set to its size, standing on the ground, and checked; when a check fails, a second shape seed is tried and the better kept | card |
| 3. Paint | MV-Adapter paints six views of the object to fit its shape, from its picture; each view goes back onto the shape; what none of them saw squarely is painted from up to four views chosen to see it (SDXL inpainting) | card |
| 4. Finish | the shape cleaned up, a web in a hole the picture shows (a jug's handle) carved away, a thing that lies (a book) laid on its broad face, levelled when a leg is a little short, and checked again, the colours baked into one texture, the paint's coverage checked, the `.glb` written with its marks, the sidecar and the report | CPU |

Before a shape is checked (in step 2 and again in step 4) it is cleaned up the standard way, each
step counted in the report: corners at the same place joined; triangles with no area, folds of no
thickness and doubled triangles dropped; corners split where two sheets touch; holes filled when
small (at most 24 edges and 5 % of the object's size round); hollows sealed inside it dropped. On
the first trial every hole was a seam of unjoined corners and every torn edge a doubled triangle
left by the cut to the triangle budget; the one hole left after joining was 8 edges and 2.5 % of
the barrel's size round.

Settings worth knowing: `--lead` (the words about light and backdrop the picture model is given
first), `--paint-steps` (the view painter's steps, 20 by default: on the second trial 20 painted as
well as 50 by coverage and likeness in under half the time, about 4 minutes, not 8 to 9),
`--shape-tries`, `--tries`, `--picture-rounds`.

`./model-maker batch PROMPTS.txt --out DIR` runs many prompts stage by stage, so each model is
loaded once, and writes `DIR/times.csv`, the time of each stage for each object. A stage already
done is kept, so a run can be started again. `./model-maker finish DIR` runs step 4 again on the
CPU.

Nothing is fixed by hand: a picture that fails is painted again, a shape that fails is made again,
and what still fails says so in its sidecar and report.

## How a prompt is read

`./model-maker words "a wooden longbow"` shows it. A word list decides, and the report says which
word decided each choice:

* an object, or an object held in the hand, and which kind (a straight blade, a curved blade, a head on a
  haft, a spear, a bow, a shield);
* whether it stands on its own base (a chair, a barrel, a jug do; a broom does not);
* whether it lies on its broad face (a book, a coin, a plate, a tray): the finish lays it on the face it most often
  comes to rest on, unless the words stand it up ("standing", "upright", "on its spine"). A chair, a table, a chest
  stand as made, and one that tips fails;
* whether it is mirror-symmetric (a chest, a sword, a spear, a double-bitted axe are; a war hammer
  with a spike, a broom, anything "broken" or "open" are not; "never opened" or "cannot be opened" opens nothing);
* its usual size: the largest side of an object, the length of one held in the hand. `--size` sets it;
* for a thing with a lid, a cover or doors (a chest, trunk, box, crate, book, cabinet, wardrobe),
  whether it is shut or open: shut unless the words say "open", "opened" or "ajar" (and not "never opened", "not
  open", "cannot be opened": a negation up to three words before the word denies it). The picture model
  is told so after the prompt ("its lid shut"; a book "lying shut, only its cover showing").

* whether it asks for more than one thing: a pair ("a pair of boots"), a count ("two chairs", "3 swords"), a group
  ("a set of knives"), a known thing in the plural ("leather boots") or two known things joined ("a sword and a
  shield"). Release 1 makes one object at a time, so such a prompt is answered before anything is painted, with the
  one thing to ask for instead: `model-maker makes one object at a time, and these words ask for more than one. Try
  one: "a leather boot".` A thing that is one object though named as a pair or a set is made ("a pair of scissors",
  "a pair of tongs", "glasses", "trousers", "a set of drawers"), and so is a count that describes ("a two-handed
  sword", "a three legged stool"), a plural after "of" or "with" ("a bag of coins", "a chest of drawers") and a
  pile, heap, stack or bundle (one object). See "Several things, and qualities the words name";
* the qualities its words give it that the picture must show ("a curved dagger", "a round box"): see the same
  section. A curved blade ("a curved dagger") is checked as a sabre is (its curve without a kink; no straight line,
  no mirror across it).

`--stands yes/no` and `--symmetric yes/no` override the words.

## The checks

Every check has a measure and a limit, written in the report beside what it means.

Pictures:

| Check | Passes when |
|---|---|
| one piece | the cut-out's largest piece holds 97 % of it |
| plain backdrop | at most 2 % of the backdrop is something else, more colourful than it (its own grey in shadow, on a floor or in a vignette is judged in the cut-out) |
| no shadow or floor in the cut-out | at most 0.5 % of the cut-out is backdrop taken in with the object: backdrop-coloured pixels in its lowest tenth, joined to a stretch of its lower edge with no colour step across it (the edge need not run through a shadow on the backdrop: the stricter reading) |

Shown, not counted: the shadow on the backdrop (how much darker it is near the object than far from
it) and a floor line on it. The picture model puts a floor shadow under nearly everything; the second
trial (212 pictures, 156 with such a shadow) found none inside a cut-out by eye, and the models made
from shadowed pictures had no slab and no dark foot. What would harm the model is a shadow inside
the cut-out, and that is what is checked. The measure cannot tell an object as pale and grey as the
backdrop from a shadow (2 of the 212, both a white table top): those fail safe, and another try is
painted.
| not cut off | the object keeps 1 % of the picture from every edge |
| shows the thing asked for | SigLIP 2 finds the thing asked for (the prompt and its kind's description) at least three times as likely as the likeliest other kind release 1 makes or a distractor (a person, text, an empty backdrop, several things), from the whole picture and its crop round the cut-out (see "The picture against the prompt" and "War hammers"); and each quality the words name ("curved", "round") at least three times as likely as its opposite (see "Several things, and qualities the words name"); without SigLIP 2 the tries go unjudged and the run says so |
| shows what the words ask | a thing with a lid, cover or doors: SigLIP 2 finds it shut (or open, as asked) at least three times as likely as the other way, judged by descriptions of what each state looks like ("the edges of its pages in one even block" against "split in two at the middle"), from the whole picture and from its crop round the cut-out |

Objects:

| Check | Passes when |
|---|---|
| outline matches the picture | judged against the cut-out the shape model was given (its largest piece, its pinholes filled), from the view that matches best (its turn, its tilt and the camera's distance: straight on, or through a lens 6, 4 or 3 times the object's size away, as the picture is a photograph; the views compared by the edges' mean distance, each camera distance refined), on a canvas as wide as the picture needs (a model wider still from that view fails), the outlines' edges lie at most 1.9 % of the height apart on average and 4 % at the 95th percentile (a compact shape offset all round by 1.9 % overlaps its picture 85 %), and no piece one outline has in a gap of the other (a web between legs) is thicker than 8 %; the overlap itself is shown, as it is unfair to thin parts: a chair leg a pixel off counts wholly against it |
| holes the picture shows are open | each hole of the cut-out with an inner part (1.9 % of the height in from its edge) of at least 50 canvas pixels, showing the backdrop's colour (within 10, delta E), is covered by the model's outline at most a third, seen from the camera found with those inner parts left out; and no hole of the model's own outline is covered by the picture more than a third. Counted only when a hole was judged; holes too small say "not judged" (see "Holes the picture shows") |
| no holes | no edge is used by one triangle only |
| one main piece | the main piece is 98 % of the surface; loose pieces (touching nothing) are listed |
| every foot on the ground | no foot (a low point of the underside, lower than all round it, well outside where the others touch) stops short of the ground by more than 0.5 % of the height |
| base level | set down, it settles at most 2° onto the face of its hull under its centre of mass |
| centre over the base | its centre of mass lies 15 % of the base's radius inside where it touches the ground |
| left-right symmetry | mirrored across its best upright plane, it lies at most 2 % of its size from itself on average, and at most 5 % at the 98th percentile (measured to its triangles: a leg missing, out of place or on one side) |
| paint covers the surface | the painted views coloured 90 % of the surface as it is seen: each texel counted by the share of 32 directions round the object it is in view from (the inside of a jug's neck counts little); the plain share of texels is shown beside it |
| fill matches the paint round it | (when the paint was touched up) judged on what the finished model shows. The fill kept: in no touch-up view is more than 10 % of it a colour found nowhere in the paint round it (each filled point against the 32 painted points nearest it on the same surface, foreign when more than 20, delta E, from every one of them), and no joined foreign place is larger than 3 % of the object's size. The places left unpainted (the fill the touch-up left out, which shows the colour spread from round it): none larger than 3 % is a smear, where neither the paint round it nor the colour it shows is one colour (less than 90 % of each within 20 of its median). The fill as the inpainter painted it, before what was foreign was left out, is shown, not counted |

Also measured, not checked: likeness, the painted object seen from its picture's view against the
picture (the mean colour difference, delta E). It compares ways of painting the same shape; its
absolute value also carries the shape's small misfits to the picture.

Objects held in the hand (and long-handled tools, which have a shaft and a grip):

| Check | Passes when |
|---|---|
| blade, haft or shaft straight | its middle strays at most 1 % of the length from a straight line; the middle is the part on its line, followed slice to slice from the middle of its length (a head's corner or a beard beside the haft is not haft), and a haft or shaft is its longest unbroken stretch, stopping where the head begins |
| no kink | its direction turns at most 4° between neighbouring tenths of its length (a curved blade or a bow: beyond its own curve) |
| symmetric across the blade (or haft) | mirrored across it, it lies at most 3 % of its width from itself (measured to its triangles; a sword, a spear, a double-bitted axe) |
| limbs mirror each other | a bow mirrored about its grip lies at most 1.5 % of its length from itself |
| grip found | a stretch at least 7 cm long, 1.5 to 5.5 cm thick (its width the mean across it, however it is turned) and round, at the size set; a blade's beside its guard and at most 30 cm, and none over half a sword's or knife's length (that is the blade) |
| one main piece | no loose part at all (a floating guard or pommel fails) |

A haft's (or a shaft's) grip is on the haft proper: from the butt, it begins at the first stretch a hand wide (7 cm)
whose girth stays within 10 % of the haft's own, so a flared butt cap, its neck and a collar are not haft; the hand's
middle goes 8 cm in from there (a spear's at 40 % of its length).

A shield's grip is on a handle found on its back: a loop standing off the board (at least 3 cm off it, with at least
1.5 cm of open space under its middle for the fingers, its bar 1.5 to 5.5 cm round and at least 7 cm long; the board's
back a smooth surface fitted to the backmost points, what stands off it left out). With two handles the forearm goes
through one and the hand holds the other; by convention the shield is carried on the left arm, the forearm across the
back, so seen from behind (looking toward its front, +Z) the hand holds the handle on the right, at -X, and of two
handles one above the other, the upper. "handle found" (shown, not counted) says which handle the hand holds; with no
handle found it fails, saying so, and the grip is put on the back's middle by convention (the shape model sees only
the front, so a strap behind may not be made). The sidecar's `measures.weapon.handles` (the sidecar's key for the
measures of a thing held in the hand) lists each handle, and
`not_handles` what else stood off the back and why it is no handle.

## The marks

The `.glb` of a thing held in the hand has named empty nodes under the object's node: `grip` (where the hand's
middle goes), `grip_start` and `grip_end` (the grip's line), and `tip`; a shield's has `grip`, `grip_start`,
`grip_end` and `front`, and no `tip`. Each grip mark lies on the grip's own middle where it is (its slices' middles
within 3.5 cm of it), so a frame tilted by a one-sided head puts no mark off the haft. A shield's grip marks lie
along the bar of the handle the hand holds, and its nodes' +Y runs along that bar.
Each node's +Y points toward the tip and its +Z is the blade's flat side, so a game puts the hand's
palm at `grip` with its fingers round +Y. The sidecar lists them too.

## Thin parts

The shape model sees the object through a picture 224 pixels square, the object about 213 pixels
long, and finds its surface on a grid of 513 cells a side, a cell about 1/256 of the object's
largest size. So:

| Part | Its width in the picture the shape model sees | Grid cells across it |
|---|---|---|
| a chair's leg, 4 cm of 95 cm | about 9 pixels | about 11 |
| a broom's handle, 2.5 cm of 1.3 m | about 4 pixels | about 5 |
| a spear's shaft, 3 cm of 2 m | about 3 pixels | about 4 |
| a longbow's limb, 3 cm of 1.8 m | about 3.5 pixels | about 4 |
| a bowstring, 2 mm of 1.8 m | under a pixel | under one |

A part a few pixels wide is guessed by the shape model from its training, so it may come out
wobbly, too thick, or broken into pieces; the straightness, wholeness and grip checks measure
exactly that. On the first trial the spear's shaft, the axe's and hammer's hafts and the broom's
handle all came out straight and whole. The longbow did not, but for its view, not its thinness:
painted edge-on, its bend was out of sight and the shape model made it wave. Asked to face the
viewer, the picture model still painted it in profile with its string, and the shape model filled
the space between bow and string into a flat plate. **Bows are not made in release 1**: a bow prompt
is answered that they come later, and nothing is made. A prompt for a person, an animal or a creature is answered
that characters come in the second release (a statue or a carving of one is an object). How each prompt is read is
tested on one file of cases (`python/tests/prompt_cases.json`), which a program in front of the pipeline can test
its own reading against.

What is ready for them: a bowstring is below what the shape model can see. When a bow comes out as one curve to one side
of the line between its tips, standing at least 5 % of its length from it, the finish draws a
string (a cord 3 mm thick) between the tips with our own code: its own mesh named `string` in the
`.glb`, in one plain colour, marked in the sidecar as made by code, and not part of the checks,
which judge the shape model's work. A bow that is straight, or waves, gets none, and the sidecar
says why.

## The regression set

Every model a check caught, and every case where a check once misread a good model, is kept with
the verdicts it must keep (`tools/trial/regression.toml`): the chair whose back legs came out short,
the table with a web between its legs, the waving and the plate-like longbows, the barrel's seams and
folds, the lantern's hollow and its truly unpainted bars, the jug's seldom-seen neck, the chair and
table that match their pictures with thin legs, the chest and book whose shadows lie outside their
cut-outs, the sword that passes, the third trial's table with a leg 24.6 mm short (it must still fail
"base level" as made and pass once levelled), its chest painted with the lid up and its book painted open
(both must fail "shows what the words ask"), the 12 GB card's book painted open with both halves in the cover's leather (it
must fail it too), and the first trial's chest and book, shut, which must pass it; every book and chest picture the
trials painted, labelled open or shut by eye (`tools/trial/state-cases.toml`: each must pass or fail the state check as
its label says, judged as one case, "state-pictures"); every distinct picture the trials painted, labelled by its prompt (`tools/trial/prompt-cases.toml`:
each clear one must pass "shows the thing asked for" against its own prompt and fail it against every other, judged as
one case, "prompt-pictures"; see "The picture against the prompt"); the pictures painted with a quality and without it, labelled
by eye (`tools/trial/quality-cases.toml`: judged against the prompt that names the quality, each showing it must pass
"shows the thing asked for" and each showing its opposite must fail it, judged as one case, "quality-pictures"; see
"Several things, and qualities the words name"); the fix-check's and the fifth trial's jugs, a web in the handle's loop (as made they must fail the outline, carved
they must match it, closed and in one piece), and the chairs, stool, tables and lantern whose holes must not be carved;
the fix-check's book made standing on its edge (laid, it must stand), its chest whose picture is a photograph (it
must match through a lens) and its candle as the six views painted it (it must keep failing the paint's coverage); the touch-up's first candle, whose fill painted holes and grey patches (it must fail "fill matches the paint round it" while its coverage passes), and its chair, filled in its own wood (it must pass both); the fifth trial's candle, whose foot left out shows a smear of wax and brass (it must fail the fill check), and a clay jug's and a shield's handle's places left out, the model showing one colour there (they must pass it). The state cases need SigLIP 2 in `MODEL_MAKER_MODELS`; it runs on the CPU too, a few seconds a picture. `tools/trial/regression.py run DATA` judges them all on the CPU in a
few minutes; it runs before any change to the pipeline or its checks, and every case must hold.

The cases and their verdicts are here (`regression.toml`, `state-cases.toml`, `prompt-cases.toml`, `head-cases.toml`,
`quality-cases.toml`),
and so are the sums of their data files (`regression-sums.json`), but the data itself (the made shapes and pictures,
built from the trials' folders) is not in this repository: a reader can see what each case holds and why, but cannot
run `regression.py build` or `run` without it (the faults its made-up cases hold are tested in `python/tests/`).
`tools/trial/regression-prompts.txt` makes the same objects again on the card before a release.
`tools/trial/unseen-final.txt` holds a set of prompts kept aside during development and made once for the release's
check.

## Where the models are

`--models DIR` (or `MODEL_MAKER_MODELS`) holds a folder per model: `flux1-schnell`, `birefnet`,
`triposg`, `mv-adapter`, `sdxl-base-1.0`, `sdxl-vae-fp16-fix`, `siglip2-so400m-patch14-384`. `--sources DIR` (or
`MODEL_MAKER_SOURCES`) holds the view painter's own code, `mv-adapter`, at the commit `registry/models.toml`
names. The shape model needs no code of its own repository: model-maker runs TripoSG's weights with its own denoiser
and decoder (`shape_net`, `shape_vae`). Before a model is
loaded, every file the registry lists is checked against its sha256, and a folder with code the
registry does not list, or with none of its listed files, is refused; before MV-Adapter's code is imported, its
checkout is checked to be at the registry's commit with nothing changed or added.

## Without a card

The checks, the bake, the export and the report run on any computer:
`./model-maker inspect MESH --picture PNG --prompt TEXT --out DIR` checks a mesh made elsewhere
against its picture, and `./model-maker finish DIR` redoes step 4 for a made object.

With `--recut` (for a model made on another computer, its picture and cut-out sent with it) the picture is cut again
by our own BiRefNet on the CPU and every check is judged against that cut-out, never the one sent, so a cut-out made to
fit the model (a web between legs left out, say) cannot pass the outline. The one sent must also agree with ours, their
overlap at least 97 % of their union (*the cut-out sent is the picture's*). On the third trial's chair our CPU cut and
the card's agreed 99.9 %; the same chair's cut-out forged (the gap under its seat filled, 58 % more cut-out) agreed
63.3 % and was refused. Without any cut-out `inspect` tells the backdrop by colour, and its shadow counts as the
object: of eight good models, 6 then failed.

## The third trial's two faults (10 November 2026)

The third trial passed 11 of 15. Two of the four failures had causes the checks did not name.

### The picture did not show what the words said

"a leather-bound book, closed" was painted open in two tries of four, and "a wooden treasure chest" with its lid
up in one of four. Every picture check passed them (they judge the cut-out's pieces, the backdrop, shadows and
margins, not what the object is), and the open ones scored best, so they were chosen. The book then settled onto
its spread covers (base level failed); the chest's inside could not be painted from the six views (paint coverage
failed). Both failed only through these side effects, after a card's worth of shaping and painting.

The fix, in two parts:

* The words. A thing with a lid, a cover or doors is shut unless the prompt opens it, and the picture model is told
  so right after the prompt. An open lid hides nothing useful from the shape model, but its inside is seen by no
  painted view, and it is not mirror-symmetric.
* A picture check, before any meshing: "shows what the words ask". The established technique is zero-shot
  image-text scoring with a contrast pair (as CLIP classifies without training; VQAScore and TIFA ask a question
  of a picture-and-text model instead, stronger on prompts with many parts, but a state of one thing is a single
  contrast): the picture is compared with words for the state asked for ("a closed chest", "a chest with its lid
  shut", ...) and the same words for the other state, each pair giving a share (the sigmoid of their score
  difference), averaged over the pairs and over two views, the whole picture and its square crop round the
  cut-out. The model is **SigLIP 2** (google/siglip2-so400m-patch14-384, **Apache-2.0**; Google, trained on its
  WebLI data), 4.5 GB, in the registry with every file's sum. It runs in the pinned transformers through its own
  SigLIP model and the Gemma tokenizer it ships (its own processor class came in a later transformers, so the pin
  stays). A picture passes when the asked state is at least three times as likely as the other (a share of 0.75):
  a share near one half is no verdict (moving the crop by one pixel moved one open book from 0.52 to 0.58), and a
  picture in doubt fails safe: another try is painted.

Measured on every book and chest the trials painted (36 pictures; `tools/trial/judge_states.py`):

| | Shut (by eye) | Open (by eye) |
|---|---|---|
| books, asked closed | 10 of 10 pass (0.99 to 1.00) | 5 of 6 fail (0.18 to 0.51) |
| chests | 16 of 16 pass (0.94 to 0.99) | 4 of 4 fail (0.28 to 0.42) |

The third trial's own tries: the book's open 628200 (chosen) and 628201 fail (0.44, 0.18), its shut 628202 and
628203 pass (0.99, 1.00); the chest's open 962714 (chosen) fails (0.28), its three shut tries pass (0.94 to 0.98).
With the check the shut tries are chosen. The objects with no lid, cover or doors are not judged, and their
picture checks are as before. The one open picture it misses is a book standing on its spine, opened in a V (0.90).

What did not work:

* The first SigLIP (google/siglip-so400m-patch14-384, Apache-2.0): it told the chests apart (open 0.14 to 0.26,
  shut 0.96 and over) but not the books: the open ones scored 0.67 to 0.97 for "closed", with every wording tried
  (four sets, from "a closed book" to "a book lying open in two halves, its spine down the middle").
* Other wordings for SigLIP 2 ("its covers together", "a half-open book", "a book opened in a V") moved the scores a
  few hundredths and caught nothing more; the plain wordings stay, so the judge is not fitted to these pictures.
* The whole picture alone missed a small chest with its lid up (0.61); the crop alone scored an open book 0.52. Both
  views averaged catch both.

Not tried: a question-answering model (VQAScore's way). Of the small ones with a commercial licence (Qwen3-VL 2B,
SmolVLM2, moondream2, all Apache-2.0), each is a larger download and slower per picture; it is the next step if
open-or-shut pictures keep getting through.

### A leg short by a little

"a small round table" came out with one of its three legs 24.6 mm short (3.8 % of its 65 cm height) and its top
dead level: set down, it settled 3.6° onto the short leg and failed "base level" and "centre over the base". The
picture showed every leg to the floor; the shape model guessed the thin leg's end from a few pixels. "Every foot on
the ground" had not named it: three legs always all rest on the floor once it tips onto the short one, so only the
tilt shows it.

The fix is the one a furniture maker uses on a rocking table, and a 3D print's base gets (a plane cut, then the
cut capped): find the floor the object should stand on, a plane square to its up axis, and cut away what reaches
below it. The finish (`levelling.py`) looks for the underside's low places (each foot's end, from the base checks'
own height map), and tries the floors at their heights from the lowest up; the first at which the cut object passes
every base check is kept. Each cut leg is closed with a flat cap (its loop of cut edges, ear-clipped), so the
surface stays closed. It cuts at most 5 % of the height (the height the base checks look for a foot within: 3.25 cm
on the table, 4.75 cm on a 95 cm chair); a foot short by more is left as made and stays a failure, as is a thing
the cut does not set level, and a thing made leaning whole, whose flat top is square to the floor its feet settle it
on rather than to its own up (no foot is short; see "Defects found by measurement"). The shape the model made is kept unchanged in `shape.npz`; the sidecar's `levelled`
says what was found and cut, and the report shows it beside the checks ("feet trimmed to one floor", not counted).
The painted colours follow the cut: each corner takes the colour of the made corner nearest it.

On the trial's table (`./model-maker finish`, the CPU): the two long legs trimmed 24.6 mm; it settles 0.0°, its
centre 55 % inside its base, closed and in one piece; it now passes every check. The cost: the table is 2.5 cm
lower than its picture, so its outline's edges lie 0.80 % of the height apart on average (0.45 % before; the limit
is 1.9 %) and its likeness is a little lower (mean colour difference 13.2, 11.8 before). The first trial's chair,
its back legs 28.9 mm short, is trimmed the same way and passes; the regression set keeps it failing "base level"
as made, so the check still catches what the shape model got wrong.

What was not done: lengthening the short leg instead. It would keep the table's height, but it means finding that
leg's own cross-section and extending it down, a guess at its shape; the cut is exact, keeps the surface closed,
and costs the object a few centimetres of height at most.

## The fix-check's three faults (10 November 2026)

The ten regression prompts made again after the third trial's fixes passed 7 of 10. Each of the three left had a
different cause.

### The book stood on its edge: the picture's view, and our set-down

"a leather-bound book, closed" was painted shut, as asked, but from well above ("lying shut, only its cover showing"
draws the picture model nearly overhead). The shape model makes a thing as its picture shows it, in the picture's own
frame: its up is the picture's up. So it made a slab standing on its bottom edge, leaning back, its cover to the
viewer; the finish set it down with the shape model's up as its up, and it stood on an edge and settled 54.8° (base
level, centre over the base and the feet failed). The picture was not wrong (a book lying shut is seen from above),
nor the shape (it matches the picture closely, see below): what was wrong was taking the shape model's up as the
book's up.

A thing that lies has no up of its own: it rests on whichever broad face it falls on. The established way to find
how a rigid object comes to rest is its stable poses, each with its quasi-static probability (Wiegley, Rao and
Goldberg, 1992, "Computing a statistical distribution of stable poses of a polyhedron"; Kriegman, 1997; Moll and
Erdmann, 2002; Fu, Cohen-Or, Dror and Sheffer's upright orientation of man-made objects, 2008, starts from the same
candidates, the faces of the convex hull): set down slowly, the object rolls downhill on the height of its centre of
mass over the floor (its hull's support distance along the direction that points down) until it rests on a hull face
under its centre of mass; the share of all directions that roll to a pose is its probability. `lying.py` follows
6,000 directions (a Fibonacci lattice), each to its lowest neighbour until none is lower, joins resting directions
within 12° (a cover a little warped has several shallow minima), and lays the thing on the likeliest pose; of two
near-equal ones (a book's two covers) the face that looked toward the picture's viewer goes up.

Only a thing that lies is laid: `words.LYING` (a book, a coin, a plate, a tray), unless its words stand it up
("standing", "upright", "on its spine"); the report says which word decided. A chair, a table, a chest, a barrel stand
as the shape model made them, and one that tips fails its base checks; it is never laid on its back (a test holds a
leaning chair to that). The laying is done where the levelling is: the shape stage judges each shape try laid, the
finish lays the shape it exports; `shape.npz` keeps the shape as made, the paint's colours follow the turn, the
sidecar's `laid` says how far it turned and the poses found, and the report shows it beside the checks ("laid on its
broad face", not counted). The outline is judged in the frame the shape was made in, the picture's, so the views it
searches are those of any other object.

On the fix-check's book (`./model-maker finish`, the CPU): laid on its back cover (39 % of the ways it can fall; the
front cover 39 %, each edge 5 % or less), turned 56°; it settles 0.6°, its centre 57 % inside its base; closed and in
one piece; paint 97.2 %. It still fails "every foot on the ground": its back cover is not flat. The shape model made
the book in two lobes, after the bookmark's ribbon that splits the page block in the picture, and laid on that cover
it rests on a strip down its middle, the cover's edges 2 to 8 mm above the floor (16 low places 1.6 to 3.2 mm short;
the levelling, which may cut 5 % of the height, 3.7 mm on a book 7.5 cm thick, finds a low place 4 mm up and leaves it
as made). That is the shape's own fault, small, and it stays a
failure; the book is made again on the card.

The regression set keeps it as `book-made-on-its-edge`: as made it must fail "base level" and "centre over the base",
laid it must pass them, turned 50 to 62°.

### The chest's outline: the measure was wrong, twice

"a wooden treasure chest" was shut and looked right from every side, but failed "outline matches the picture" at the
95th percentile (4.95 % against 4 %; the mean 1.40 % passed). The outline picture showed where: the feet. The
picture's near front-right foot stood lower than its far back-right one, while the lid showed only as a sliver; in the
model's best view the feet were at its ends.

Two faults of the measure, not the shape:

* The outlines were compared on a canvas twice as wide as high, and an outline wider than that was squeezed to fit.
  The chest is 2.25 times as wide as high in its picture and 2.10 in the model: both were squeezed to 2, so their
  widths were never compared. Unsqueezed (a canvas six times as wide as high), the same view reads 3.65 % and 8.33 %;
  the view searched again, 2.17 % and 7.55 %. No other case in the regression set is that wide, so none other moved.
* The picture is a photograph, painted through a lens: near parts larger and lower than far ones. The measure drew the
  model straight on (orthographically). From any turn and tilt, straight on, the chest's feet cannot be where the
  picture has them while its lid shows that little. Drawn through a lens 6 times its size away (a field of view of
  about 20°), it matches: 0.68 % and 1.82 %, every foot in place. The book's outline was the same: 1.29 % and 5.47 %
  straight on (failed), 0.51 % and 1.30 % through a lens 4 sizes away; its picture, looking steeply down on a flat
  thing, is a trapezoid.

The established way to compare a model with its photograph is to fit the camera too (render-and-compare: the pose and
the focal length, as camera calibration from a silhouette does). The view search now tries the camera's distance with
the turn and tilt: straight on, and 6, 4 and 3 sizes away. That is more freedom, and it let the first trial's table
with a web between its legs through: its edge distances through a lens 6 sizes away were 1.10 % and 4.33 % (failing
by 0.33 % only), 4 sizes away 0.98 % and 2.66 % (passing). The web's edges lie near the legs' edges, so the edge
distances barely see it. So the check now also measures what they cannot: the thickest piece one outline has in a gap
of the other (within the other's convex hull: between a table's legs), at most 8 % of the height, twice the 95th
percentile's limit (the width such a piece, lying between two edges of the other outline, may have and still read
the 95th percentile's limit at its edges). The web is 11.5 % from every camera; a piece outside the other's hull (a
corner sticking out, the first trial's chest's at 8.1 % through a lens) reaches away from the other outline's edges,
and the edge distances judge it.

On every case (straight on and squeezed, before; now; mean, 95th percentile, thickest piece in a gap):

| Case | Before | Now | Verdict now |
|---|---|---|---|
| chair-good | 0.44 %, 1.07 % | 0.44 %, 1.07 %, 2.6 % (straight on) | passes, as it must |
| chair-short-back-legs | 0.45 %, 2.08 % | 0.52 %, 1.56 %, 2.1 % | passes |
| table-short-leg | 0.80 %, 1.84 % | 0.43 %, 1.09 %, 3.3 % | passes, as it must |
| table-good | 0.73 %, 2.08 % | 0.46 %, 1.30 %, 2.6 % | passes, as it must |
| table-web | 1.34 %, 5.47 % | 1.10 %, 4.33 %, 11.5 % | fails, as it must |
| sword-good | 0.09 %, 0.26 % | 0.09 %, 0.26 %, 1.0 % (straight on) | passes, as it must |
| book-shadow-left-out | 0.12 %, 0.26 % | 0.39 %, 1.30 %, 0.5 % | passes, as it must |
| barrel, stool, jug, lantern, candle | at most 0.49 %, 1.56 % | at most 0.51 %, 1.58 %, 5.2 % | pass |
| chest-shadow-left-out, chest-lid-open, book-painted-open | at most 1.28 %, 3.91 % | at most 0.61 %, 2.08 %, 1.6 % | pass |
| longbows | 0.16 %, 0.52 % and 0.23 %, 0.52 % | 0.19 %, 0.52 % and 0.25 %, 0.74 % | pass (not what fails them) |
| the fix-check's chest | 1.40 %, 4.95 % | 0.68 %, 1.82 %, 2.6 % | passes |
| the fix-check's book | 1.29 %, 5.47 % | 0.51 %, 1.30 %, 2.6 % | passes |

The likeness (not a check) is drawn from the same camera. The regression set keeps the chest as
`chest-through-a-lens`.

What did not work: the thickest piece of either outline alone, anywhere, without the hull. A good model's corner
sticking a little out of its picture's (the first trial's chest, 8.1 %) read like a web; edge distances already judge
those.

### The candle's paint: places seen from above at a slant, which none of the six views sees squarely

"a candle in a brass holder" passed every shape check, but the paint covered 85.8 % of the surface as it is seen
(limit 90 %). The coverage picture and a count by place and by the directions that see it (the check's own 32):

* what was missed is the dish's inner wall and the candle's lower part inside the dish (10.2 % of the surface as seen
  lies between a tenth and three tenths of the height, 3.7 % between three and five tenths), facing sideways (7.9 %) or
  up (6.3 %); the underside, painted by the view from below, is not among it (0.03 %);
* every missed place is seen from above: of the missed weight, 3.8 % of the whole comes from directions 5 to 30° above
  the horizon, 7.0 % from 30 to 60°, 3.4 % from 60 to 90°, and 0.01 % from below. The four side views look level, and
  the dish's rim hides its own inside and the candle's foot from them; the view from straight above sees the inner
  wall and the candle's sides edge-on.

So these places are not hidden: someone looking down on a candle on a table sees exactly them. The check is right
to count them.

The chair sits at the limit too: the fix-check's chair (made on the 16 GB card) reads 90.7 %, and the same prompt made
on the 12 GB card read 89.99 % and failed on that alone. Its misses are of another kind, but also geometric: 8.2 %
of its surface as seen is sideways-facing (the legs' and spindles' sides), hidden in the four level views behind the
legs in front of them (each level view looks straight along the chair's rows of legs); 5.4 % of the missed weight is
seen from below the horizon, 3.5 % from above. Counting only the directions above the horizon would read 92.5 % and
pass it: that would be the limit loosened by another name, so it is not done. The views chosen for it are the
diagonals the four level views lack: 45° and 315° round, 20° above (2.3 % and 1.8 % of the surface as seen). The
lantern (93.2 %, its cage bars' insides) gets two, 1.4 % and 1.0 %; every other object of the fix-check (jug, book,
table, shield, stool, barrel, chest: 0.8 to 4.9 % unpainted) gets none, as no view would add 1 %. A place no direction sees (the candle's foot where it meets the dish) already counts nothing. Counting
only the directions above the horizon, as for a thing that never leaves the floor, was measured: 74.8 %, stricter,
as it drops the underside, which the view from below paints. It is not done: a thing can be picked up or knocked
over, and the question of what can be seen in use is not the one this failure asked.

The established fix is views chosen by what is missed. MV-Adapter's own texturing (its paper, section 5.4) runs "view
coverage analysis to identify uncovered regions", renders the current texture from those views and inpaints them;
LumiTex ranks candidate views greedily by the uncovered area each sees; Hunyuan3D 2.0 paints from denser views (6 of
44 preset). MV-Adapter's image-and-geometry model was trained on the six fixed views only (four level, above and below),
so its views cannot simply be moved up. Measured on the candle's shape, what views could reach squarely (facing at
least 0.55, as the colour cloud keeps): the six, 84.9 % of the surface as seen; with four more at 20° above, 91.7 %;
at 30°, 94.0 %; at 45°, 96.6 %.

The paint stage now touches the paint up (`touch_up.py`, at most 4 views, `--touch-up-views`, 0 for none): candidate
views every 45° round and 20°, 35° and 50° above are ranked by the unpainted surface each sees squarely, weighted as
the coverage check weights it; the best is taken, the object drawn from it in its colours so far, the unpainted pixels
in it (grown by 3 pixels so the fill meets the paint round it) painted by SDXL's base model through diffusers' SDXL
inpainting pipeline (the base weights, already in the registry; no new model), and only the unpainted pixels' new
colours put back into the colour cloud; then the next view, which sees the first's fill, until a view would add less
than 1 % of the surface as seen. A box the six views see whole needs none. What no view sees squarely is left to the
spread, and counts as not covered.

On the CPU, the views chosen for the fix-check's candle are 180° round and 50° above (4.3 % of the surface as seen),
0° and 50° (4.1 %), 45° and 50° (2.2 %), 225° and 35° (2.0 %). With a stand-in painter in place of the inpainting
(each unpainted pixel the colour of the nearest painted one: a test of the plumbing, not of the paint) the finish
reads 98.4 % (96.6 % of all texels). How well SDXL fills those places is for the card to show; the regression set
keeps the candle as the six views left it (`candle-dish-unpainted`, it must keep failing), so the check still catches
what the six views miss.

What did not work, or was not done: moving MV-Adapter's own cameras up (it was trained on the six level, top and bottom
views only; its paper's arbitrary-view model is another, separately trained model); LaMa, the inpainting model
MV-Adapter's texturing uses (trained on Places, whose terms are for non-commercial research); a dedicated SDXL
inpainting model (a separate download and registry entry; the base model is tried first).

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
  strength 0.6 (`--touch-up-strength`; 1 paints from noise as before).
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
faster, with the same views and values (see "The outline's search made fast" below). The fill's judgement adds about
3 s a view and to the finish on the CPU; a retried view adds one more inpainting, and strength 0.6 runs 18 of the 30
steps.

## The fill judged on the model (11 November 2026)

Release 1's unseen run failed six of its ten objects on "fill matches the paint round it". Two of them, the rocking
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


## The jug's handle: a web where the picture shows a hole (10 November 2026)

"a clay jug" failed "outline matches the picture" in every trial since the outline's gap piece was measured, on both
cards and with both shape seeds: the shape model closed the loop of the handle with a web where the picture shows the
backdrop through it. The thickest piece of the model's outline in the picture's gap was 15.1 % of the height on the
fix-check's and the third trial's jug (seed 42), 13.1 % on the fifth trial's (seed 43; seed 42 15.1 %), 13.5 % on the
12 GB card's fourth trial (seed 43); the limit is 8 %. Everything else of the jug matched (edges 0.64 to 0.95 % apart on
average). A mug, a lantern's ring, a key's bow, a shield's grip seen from behind would meet the same fault.

### Why: the shape model was shown a grey surface in the loop

`cutout.for_shape`, the cut-out the shape model is given, filled every hole closed off inside the object. The picture
is laid on white where the cut-out is empty, so inside the filled loop the shape model saw the picture's own backdrop:
light grey, darkened by the jug's shadow (a mean of 214 and 213 of 255 in the loops of the two pictures the webbed jugs
were made from, one on each card), framed by the handle, on white all round. To a model that sees a picture of an
object on white, that is a grey surface, and it made one. The first trial's jug, made from another picture, whose
handle came out open, had a lighter backdrop in its loop (239). TripoSG's own preparation (its `image_process.py`)
keeps the cut-out's holes and puts the object on white; it drops only specks of the cut-out under 200 pixels.

The webs are thin: 0.6 to 1.1 cm deep along the picture's lines of sight, across a loop 4.7 cm wide.

### The fix, in two parts

* **The shape model is given the holes** (`cutout.for_shape`): only a pinhole of the cut-out (under 200 pixels of
  the 1024-pixel picture, TripoSG's own speck size) is filled; the loop of a handle shows white, as the rest of the
  backdrop does. Tried on the CPU (TripoSG in 32-bit floats, its 50 steps, seed 42, the fifth trial's picture; 85
  minutes a shape on 4 threads, so its shape is not the card's, but both runs differ only in what they were shown):
  given the loop filled, the shape model made the web again (the gap piece 12.2 %, the edges 0.74 % and 2.39 %;
  carved: 2.3 %); given the loop as the picture shows it, it made the handle open (0.22 %, 0.52 %, 0.7 %), every
  shape check passing, nothing to carve. How it does on the card, and on every other object with a hole (the chairs'
  backs, the lantern's ring), is for the card's next run to show.
* **A web the shape model still makes is carved away by the picture's silhouette** (`carving.py`, in the shape stage's
  judgement of each try and in the finish). The picture is the one view we have, and where it shows the backdrop
  through the object, nothing lies along those lines of sight: the object lies within its silhouette's cone from the
  camera, its visual hull (Martin and Aggarwal, 1983; Laurentini, 1994), and what lies outside it is carved away
  (space carving, Kutulakos and Seitz, 2000). Only a web is carved, never the object's body:

  1. a hole of the picture's outline, closed round by the object, at least 7.8 % of the height wide (room for a piece
     thicker than an edge's misfit and the margin on both sides);
  2. the camera found as the outline check finds it (the same turns, tilts and lens distances), but with the
     outlines framed by their boxes and the holes' inner parts left out of the count: the web fills the shape's hole,
     and must not pull the camera to a view where it fits;
  3. the shape's piece in the hole thicker than 4 % of the height (the 95th percentile's limit on the edges: the most
     an edge may be out of its place; a chair's spindles a little off their picture's leave pieces up to 2.1 %);
  4. along every line of sight through the hole's inner part, the shape no deeper than the hole is wide: a web spans a
     hole and is thinner than it; the object's body behind the hole would be deeper (a test mug whose loop is filled as
     deep as its body, 20 cm in a loop 5 cm wide, is left as made);
  5. the cut keeps 1.9 % of the height (the edges' mean limit) inside the hole's edge, so an edge off by what the check
     allows is never cut; the triangles near the cut's line are split until none is longer than 3 canvas pixels (the
     simplifier leaves large flat triangles on a web, and one cut straight across cuts a chord of a curved hole), then
     the surface is cut along the line where it crosses them (as the levelling cuts along a plane);
  6. each opening the cut leaves is closed by a wall along the lines of sight: the web's front and back edges joined by
     a strip of triangles, its steps chosen so its new edges are shortest in all (as contours are joined into a surface,
     Fuchs, Kedem and Uselton, 1977), whether the web closed the loop whole (two openings, front and back) or ended
     inside it (one opening that runs along the front and back along the back); shreds of web the cut leaves hanging
     are dropped, a stray small opening on one face is filled as the repair fills a small hole;
  7. kept only if the surface is closed (no open or torn edge more than before), in no more pieces than before,
     encloses a volume, and, seen from the camera, nothing is left in the hole's core (twice the margin in): else that
     hole is left as made, and the sidecar says why.

  The shape the model made is kept unchanged in `shape.npz`; the sidecar's `carved` says what was found in each hole
  and what was cut, and the report shows it beside the checks ("webs in the picture's holes cut away", not counted).
  The painted colours follow as for the levelling: each corner takes the colour of the made corner nearest it.

On every jug the trials made, judged as the finish judges (the outline's edges: mean and 95th percentile; the thickest
piece in a gap; the shape closed and in one piece in every case):

| Jug | As made | Carved | What was cut |
|---|---|---|---|
| the fix-check's (also the third trial's and `jug-handle-web`), 16 GB card, seed 42 | 0.95 %, 3.12 %, 15.1 % (fails) | 0.53 %, 2.03 %, 3.3 % (passes) | a web 1.1 cm deep across the loop's lower part |
| the fifth trial's (`jug-handle-carved`), 16 GB card, seed 43 | 0.69 %, 2.08 %, 13.1 % (fails) | 0.37 %, 1.84 %, 2.6 % (passes) | a web 0.6 cm deep, ending inside the loop |
| the 12 GB card's fourth trial, seed 43 (from its `.glb`, no shape kept) | 0.64 %, 1.40 %, 13.5 % (fails) | 0.36 %, 1.67 %, 2.6 % (passes) | a web 1.3 cm deep |

Finished again on the CPU with their paint (`./model-maker finish` on copies), the fix-check's and the fifth trial's
jugs now pass every check: the paint covers 96.5 % of the surface as it is seen (97.0 % and 96.8 % with the web), the
likeness is as before (mean colour difference 6.0 and 6.5; 5.8 and 6.5). The cut adds about a thousand triangles to a
jug (61 100 to 61 600) where the line was refined.

Of the 67 objects with a kept shape (the regression set, the fix-check, the third and fifth trials, the three fails,
the touch-up), judged before and after: the four jugs above changed (outline fails to passes), and the first trial's
round table (`table-web`), whose picture shows the backdrop through two gaps between its legs and stretchers where the
shape model put webs 0.5 and 0.6 cm thick, is carved too and now matches its picture (1.10 %, 4.33 %, 11.5 % as made;
0.93 %, 2.60 %, 3.7 % carved; 74 300 triangles, the long cut lines refined). Every other object kept every verdict and
every value: the chairs' backs, whose holes the shape keeps open (pieces of at most 2.1 % along the spindles), the
lanterns (the ring's hole open in the shape; the gaps over the base 4.7 % wide, too narrow to hold a web), the
stools, tables, barrels, chests, books, candles, shield, swords, spears and axes (no hole wide enough, or none). The regression set holds 23 of 23:
`jug-handle-web` and `table-web` must fail the outline as made and pass it carved, `jug-handle-carved` is new, and the
chairs, stool, tables, lantern, sword and chest must have nothing carved.

What the carving leaves: a rim of web 1.9 % of the height wide along the loop's inside (the margin), which keeps the colour the
view painter gave the web; on these jugs the painter painted the web nearly black (it read as the loop's shadow), so a
dark sliver shows on the handle's inner edge. A carving sees only what the picture's view sees through: a web seen
edge-on from the picture's camera (the table's webs between its far legs) is not in any hole of the picture's outline,
and stays. With the shape model shown the holes, the web should not be made at all; the carving is the net under it.

### What did not work

* Cutting away whole triangles whose corners all lay in the hole: the simplifier leaves a web's flat face in a few
  large triangles, so the cut stopped far from the hole's edge on one face and close to it on the other.
* Cutting along the line without splitting the triangles near it first: on the fifth trial's jug the front face's cut
  ran as a straight chord across the curved loop and the back face's along it, and the wall between them crossed the
  hole (it was refused: 399 pixels of the hole's core still covered).
* The outline check's own camera, and the camera found from the outlines with their holes closed: a web whole across a
  test mug's loop drew the first 4° above and through a lens (the cut then took the top of the handle's lower bar), and
  on the fifth trial's jug the second turned 11° off (the cut fell beside the loop). The outline check's canvas
  centres an outline on its area's middle, which the web moves (11 canvas pixels on the test mug); the carving frames
  both outlines by their boxes.
* Joining the front and back edges by their arc length: where the two faces' edges are cut into different numbers of
  pieces it twisted the wall; the shortest strip does not.

Not tried: more shape seeds when only the gap piece fails. Both seeds on both cards made the web (four jugs of four),
as the input made it likely; another seed costs a shape stage (about a minute and a half on the 16 GB card) for a fault
that is not the seed's. The shape model's own settings (its guidance and steps) were not changed: the cause was what
it was shown.

## The state check's words: the state's look, not its name (10 November 2026)

On a run of the 12 GB card "a leather-bound book, closed" was painted open: both halves in the cover's leather, side
by side, the page edges in two blocks, a valley down the middle. "Shows what the words ask" scored it 0.88 shut, it was
chosen, the shape model built an open book in two lobes, and only "every foot on the ground" caught it, 9 card minutes
later. The same seed on the 16 GB card paints the book shut (one cover, a ribbon between the page edges), which is why
that card's runs of the same prompts passed it rightly.

### Why

The check compared the picture with the states' names: "a closed book", "a photo of a closed book", "a book lying
shut" against the same words open. Contrastive picture-text models recognise an object's state far worse than the
object (Newman et al. 2024, "Do pre-trained vision-language models encode object states?": about 30 points lower,
which they lay to weak localisation and to the text side), and a state's name alone is scored largely by the noun's
usual look. An open book in pictures is mostly white pages; a picture of brown leather and no white page reads as
"book, cover", so shut. The open books the check caught before were the ones with white pages (0.07 to 0.18); those
painted in the cover's leather were near its limit (0.40, 0.51) or over it (0.88).

### The fix

The established way past a class name's bias is classification by description (Menon and Vondrick 2023, "Visual
classification via description from large language models"; CuPL, Pratt et al. 2023): each class is scored by
sentences naming what it looks like, keeping the class's name in each (without it such descriptors lose most of their
worth). Each state is now described in pairs that name the same evidence both ways, the share of each pair averaged
as before (`words.STATE_JUDGE_TEXTS`):

| A thing's | Shut | Open |
|---|---|---|
| cover | its front cover one flat panel with no fold across it | its two halves side by side with a fold down the middle |
| | the edges of its pages in one even block | the edges of its pages split in two at the middle |
| | its covers pressed together | its covers spread apart |
| | no pages showing | its pages showing |
| | its spine straight | bent open along its spine |
| lid | its lid down on top of it | its lid raised up behind it |
| | its inside hidden | its inside showing |
| doors | its doors shut flat across its front | its doors swung out |
| | its inside hidden | its inside showing |

Each begins "a closed {thing}," or "an open {thing},". The score is the same share as before (a sigmoid of a pair's
score difference, averaged over pairs and over the two views), so the limit stays at 0.75. No cabinet or wardrobe has
been painted yet; the doors' pairs are written the same way and untested.

Measured on every book and chest picture the trials painted, one copy of each (52; 17 open by eye, 35 shut;
`tools/trial/state-cases.toml`), asked shut:

| | Before: the states' names | After: their look |
|---|---|---|
| shut books pass | 15 of 15 (0.98 to 1.00) | 15 of 15 (0.91 to 1.00) |
| open books fail | 7 of 9 (0.07 to 0.51; missed 0.88, 0.90) | 8 of 9 (0.06 to 0.61; missed 0.81) |
| shut chests pass | 20 of 20 (0.93 to 0.99) | 20 of 20 (0.88 to 0.98) |
| open chests fail | 8 of 8 (0.28 to 0.42) | 8 of 8 (0.15 to 0.34) |

The leather-coloured open book goes from 0.88 to 0.61 and fails; its shut twin from the 16 GB card stays at 0.91. The other
two leather-coloured open books fall from 0.51 and 0.40 to 0.41 and 0.36. The book standing open in a V on its spine is
still missed (0.90 to 0.81): it is kept in the labelled set as a known miss, and the run says when it is caught. The
regression set holds (25 of 25 cases, the labelled pictures as one), and with the old words the new cases break (the
open book passes at 0.88).

### What did not work, and what was tried

* Adding the descriptions to the states' names (three names and two descriptions a side): it failed the leather book
  too (0.61), but the names pull towards the noun's usual look, and nothing they add is evidence of a state; the
  descriptions alone are kept.
* Averaging the wordings' text vectors before scoring (CLIP's prompt ensembling) or averaging their scores per state:
  the same verdicts on every picture, the V book scored higher (0.89, 0.90 against 0.81); the pairs stay.
* The two strongest pairs alone (the fold down the middle, the page edges split): the leather book fell to 0.20, but its
  shut twin to 0.80, five hundredths from the limit. Five pairs keep the shut books at 0.91 and over.
* A third lid pair ("its lid resting flush on its box" against "tilted back on its hinges") brought a shut chest down
  to 0.84 and caught nothing more; it was left out.
* A pair for the V ("its two covers flat against each other" against "apart in a V") moved the V book by a hundredth
  and the leather book up to 0.67; it was left out.

Five sets of descriptions were scored in all, each on the same 52 pictures; the limit was not moved. Not tried: a
question-answering model (VQAScore's way), still the next step if the V book or another state gets through.


## The picture against the prompt (10 November 2026)

A model made on another computer comes with the picture it was made from. Before the model is shown, its picture is
checked to show the thing the prompt asks for (`model-maker picture-check`, `subject_judge.py`):
a chair for "an old looking chair", not a table, a vase or a person. A computer that swaps in another picture, or whose
picture model painted something else, is caught here.

### How it judges

This uses established zero-shot image-text classification (CLIP's, SigLIP's): the picture is compared with words for
the thing asked for and with words for every other thing it might show instead, and the thing asked for must be the
likeliest. The other things are every other kind of thing release 1 makes (`words.SUBJECT_KINDS`, 55 rows), each by a
description of its look with its name first ("a table, a flat top standing on legs"; classification by description,
as the state check's words are), and five distractors: a person, an animal, text, an empty backdrop, several things.
The thing asked for is the prompt itself ("a photo of an old looking chair") and its kind's description. Each class's
wordings are averaged into one direction (CLIP's prompt ensembling). A kind any word of the prompt names is never a
contrast ("a chest with a sword carved on it" is judged against neither), and near kinds a prompt may name by either
word share a row (a jug and a vase, a sword and a dagger), so a jug is never refused for looking like a vase. A
likeness of a creature ("a stone statue of a knight") is not judged against a person or an animal, and a prompt that
asks for letters not against text. The share is the sigmoid of the score difference between the thing asked for and
the likeliest other, averaged over the whole picture and its crop round the cut-out, as the state check's is. The
model is the state check's, SigLIP 2, on the CPU.

### Measured

Every distinct picture the trials painted (268, from 13 runs of 16 prompts; `tools/trial/prompt-cases.toml`): each
against its own prompt, and against every other prompt of the trials (the "shuffled" wrong prompts). The 16 longbows
are left out because bows are not made in release 1. Labelled by eye, 13 pictures do not clearly show their thing:
all 8 war hammers (no clear hammer face; labelled blind later, one is an axe head and three are hybrids, each copied on
two cards: see "War hammers") and 5 tables painted from straight
above (a round top with at most its feet's tips showing, so a table and a stool look alike). They are shown, not held.

| | First try: a row for halberds | Now: halberds in the axe's row |
|---|---|---|
| own prompt, 239 clear pictures | lowest 0.05 (a war hammer, "a halberd") | **lowest 0.926** (a barrel seen from above, "a helmet") |
| another prompt, 3,346 pairs | highest 0.94 (a longbow asked as a spear) | **highest 0.500** (a three-legged stool asked as "a small round table") |
| an empty backdrop, each prompt | | highest 0.041 |

The limit is **0.75**, the state check's: the thing asked for at least three times as likely as the likeliest other.
In score terms it lies near the middle of the gap (logit 1.10 between 0 and 2.53): 0.25 above the highest wrong
prompt and 0.18 under the lowest own prompt, in share. The nearest wrong pairs are a stool and a small round table
(0.50, 0.46, 0.37), as is fair: a three-legged stool is a small round table's look-alike.

The first try kept halberds, glaives and poleaxes as a row of their own. The picture model paints a battle axe on a
long haft with a spike on top, which SigLIP 2 scores as a halberd: four of eight battle axes fell to 0.15 to 0.38.
A halberd is an axe on a pole, so it now shares the axe's row, as a vase shares the jug's.

The war hammers' scores (0.49 to 0.97) show the check sees what they are: the trials' two chosen war hammer pictures
score 0.49 and 0.58 (the copy on the other card 0.59), with the likeliest other "an axe".

### A known miss: the prompt's words written out

A picture of the prompt's words in black letters scores as the thing they name (the typographic attack on
picture-text models): "a wooden treasure chest" written out scored 1.00 against the general "text". Classes naming
the prompt's own words were tried. Each one that caught the written words also pulled real pictures down, because its
words sit next to the thing asked for:

| Written-words class | Own prompt, lowest | Written prompt, highest | Its noun written, highest |
|---|---|---|---|
| none (kept) | 0.926 | 1.000 | 1.000 |
| `the word "chest" written in letters` | 0.744 | 0.999 | 0.120 |
| `the words "<prompt>" written out in letters` | 0.218 | 0.190 | 0.065 |
| both | 0.328 | 0.332 | 0.002 |
| a page of typed text with the word | 0.223 | 1.000 | 0.862 |
| a sign with the word | 0.718 | 0.999 | 0.643 |

None is kept. The regression shows the written pictures and does not hold them. A picture of letters would also have
to pass the other picture checks, and its model an outline matched to it, but that is not measured. The next step is
a question-answering model ("is this a photo of an object or of text?", VQAScore's way).

### In the pipeline's choice of try

Every try is judged, and a try the check refuses is not chosen, as with the other picture checks (see "War
hammers"). It refuses the trials' two chosen war hammer pictures, and rightly: they show an axe head (0.49, 0.58).
Measured on every picture the trials painted that could be reached (268, one copy each; the same
pictures as the labelled set and its 16 longbows), the 239 clear pictures still pass, lowest 0.926. Of the 65 pictures
the trials chose, the check refuses the two war hammers with the axe head and one table from an early comparison of
leads (0.54, seen from straight above, labelled unclear), whose job never got past its pictures.

The labelled set is a regression case, "prompt-pictures" (`tools/trial/regression.py`): each clear picture must pass
its own prompt and fail every other, and an empty backdrop must fail every prompt. The whole regression set holds with it
(26 of 26; "prompt-pictures" took 673 s on 12 CPU threads, most of it SigLIP 2 seeing 252 pictures twice). The
written words are shown as they are: 24 of 30 still pass.

## War hammers (10 November 2026)

Every war hammer the trials chose had an axe-like head: a flared blade with a square poll and a spike (seed 141081,
chosen on both cards). It scored best on the other picture checks, and its models passed every other check. The subject
check refused it (0.49 and 0.58, the likeliest other "an axe"), but did not then take part in the choice of try.

### Why

A close look at the four designs shows the cause is mostly our wording:

* The view asked for "its point or head at the top", which invites a spike.
* Seen straight from the side, a hammer face is edge-on, and from there a face and an axe blade have the same outline.
* The concept itself is mixed: many war hammers in pictures have a blade or a beak behind the face.
* FLUX.1 [schnell] runs with no guidance, so nothing can be asked away with negative words.

### The fix

* **The check chooses.** "Shows the thing asked for" (`subject_judge`, at its limit of 0.75) is now one of the picture
  checks of every try. A try it refuses is not chosen, and when no try passes, another round is painted, as with the
  other picture checks. SigLIP 2 is loaded for every job's pictures. Without it, the tries go unjudged, the run says
  so, and a thing with a lid, cover or doors is still not made. The finish keeps the verdict from the record, and the
  sidecar lists SigLIP 2 among the models used.
* **The words.** A head on a haft (an axe, a hammer, a mace) is painted "standing upright, its head at the top, seen
  straight from the side". "point" is dropped; a blade and a spear keep it. The war hammer family (war hammer, maul,
  mallet, sledgehammer) is told what its head looks like, as a chest is told its lid is shut: "its head a heavy blunt
  block with a flat square striking face". The words name no material, so "a wooden mallet" stays wooden. "mallet" and
  "sledgehammer" are now known words (before, a mallet was an unknown object, painted lying on the floor from above).

### Measured

Each prompt was painted with 8 seeds on the 12 GB card, the pipeline's own seeds for its own words, pictures only
(`tools/trial/paint_tries.py`), in three wordings:

* "old": the words before the change;
* "new": the words above;
* "turned": the new words with the view turned a little toward the viewer, so the face shows.

The axe prompts were painted old and new. The first trial's war hammers and battle axes from the other card were added.
The 12 GB card's copies of the trial's tries are bit-identical to the "old" ones. Every head was labelled by eye before its
score or wording was seen: the top of its cut-out cropped, shuffled under a random code (`tools/trial/head_sheets.py`).
The labels are:

* hammer: blunt faces; a spike on top or a pointed beak is allowed;
* axe: a broad sharp blade;
* hybrid: a blade and a face, or flared ends between the two, or a hammer and a blade on one haft;
* other: neither.

Then the codes were unblinded. In eleven pictures the crop missed what mattered: a second blade or block at the foot
of the haft, or a mallet lying head down. Those labels were corrected from the whole picture, and the case set says so
for each (`tools/trial/head-cases.toml`, 152 pictures).

| Hammer prompts (war hammer, maul, mallet, sledgehammer) | Old | New | Turned | First trial, other card |
|---|---|---|---|---|
| clear hammers | 19 of 32 (59 %) | **30 of 32 (94 %)** | 31 of 32 (97 %) | 0 of 4 |
| of them, "a war hammer" | 1 of 8 | 7 of 8 | 8 of 8 | 0 of 4 |
| axe heads | 1 | 0 | 0 | 1 |
| hybrids, others | 9, 3 | 0, 2 | 0, 1 | 3, 0 |
| check at 0.75: passes, refusals | 27, 5 | 32, 0 | 31, 1 | 3, 1 |
| clear hammers passing | 19 of 19 | 30 of 30 | 31 of 31 | |
| axe heads refused | 1 of 1 (0.58) | | | 1 of 1 (0.49) |

| Axe prompts (battle axe, double-bitted battle axe, hatchet) | Old | New | First trial |
|---|---|---|---|
| axe heads, all passing | 18 (0.97 and over) | 20 (0.97 and over) | 3 |
| hybrids, hammers, others | 4, 0, 2 | 3, 1, 0 | 1, 0, 0 |
| check at 0.75: passes, refusals | 22, 2 | 23, 1 (the hammer head, 0.44) | 4, 0 |

Every rule holds:

* Every clear hammer passes: 80 of 80, lowest 0.935.
* Every axe head asked as a war hammer is refused: 2 of 2, at 0.49 and 0.58.
* With the new view words, every axe head asked as an axe still passes: 41 of 41, lowest 0.967.
* The trials' 239 clear pictures stay at 0.926 or over.

By prompt, with the new words: war hammer 7 of 8 clear hammers, maul 7, mallet 8, sledgehammer 8. With the old words
the numbers are 1, 2, 8 and 8.

With the check in place, the first round of four picks a clear hammer for all four prompts with the new words, and an
axe for all three axe prompts. Without the check, the new words' hatchet round would have chosen its hammer-headed try.
With the old words and the check, "a war hammer" gets a hybrid (two flared ends and a spike, 0.93) in place of the
axe head.

Shown, not held: two "other" pictures from the new words pass. One is a war hammer with two blocks on its haft
(1.00). The other is a maul try that is a pole with a brush-like end (0.86). The check judges what kind of thing a
picture shows, not whether it is well made.

### The second layer was not needed

A second layer was considered in case the subject check alone was too weak: "a hammer head with a flat blunt striking
face" against "an axe head with a sharp curved blade" (`tools/trial/judge_tries.py` scores it for every picture). The
subject check alone holds every rule above, so it is not added. The pair would also be a poor check:

* The clear hammers score from 0.21 up and the axe heads at most 0.05, so its limit would have to lie in that narrow
  gap.
* It would catch nothing the subject check misses: the two "other" pictures that pass score 1.00 on it.

No limit was moved.

### What did not work, and what was not kept

* **The three-quarter turn.** It made 31 clear hammers of 32 against 30, but one turned try put a second large block
  at the foot (refused, 0.63). It also changes the view the shape checks expect: a hafted thing's length along the up
  axis, and its width, seen from the side. It is not kept.
* **Labelling from the top of the cut-out alone.** It missed a second head at the foot in six pictures and two mallets
  lying head down. The labels were corrected from the whole picture, and the case set says so for each.
* **Reading seed 141080 by eye, unblinded, as "a real war hammer".** Labelled blind, both cards' copies are hybrids: two thin
  flared ends and a spike, no clear face. Its score (0.95, 0.97) passes either way.

The labelled heads are a regression case, "head-pictures" (`tools/trial/regression.py`). A clear head of the prompt's
own kind must pass, and a clear head of the other kind must fail. Hybrids and others are shown, not held. The old war
hammers in `prompt-cases.toml` keep their "unclear" label, now with their blind labels.

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
   as-painted fill is shown, not counted, and what was left out is judged on the model; see "The fill judged on the
   model".)
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
    and eleven such prompts are in them with five characters that must stay characters ("a giant", "a frost
    giant", "a dragon sleeping on a chest", "a horse grazing", "a knight kneeling").
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
20. **Offline.** `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are set by the `model-maker` command before Python starts (and
    by `__main__` before any of the hub's libraries is imported).

### Licences

21. **What the wheels bundle.** The licence check lists every shared library a package bundles and compares it with an
    allow-list; 93 in the locked environment, 90 allowed by their licences, and **libquadmath** (LGPL-2.1-or-later, in
    numpy's and scipy's wheels) allowed as the project's one recorded exception (CONTRIBUTING.md, rule 1; NOTICE;
    docs/setup.md, "What the wheels bundle").
22. **TripoSG's code.** None of TripoSG's repository is fetched or run: its folder of code holds the text of the
    Tencent Hunyuan FlashVDM Community License, and the repository's `NOTICE` names code derived from HunyuanDiT and
    FlashVDM. Its weights run with our own code ("The shape model's own decoder", below).

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
| Carving a real thin panel | Confirmed and fixed for an opaque panel: a hole whose colour in the picture lies more than 10 (delta E) from the backdrop's is a part the cut-out missed, never carved and never judged open (the made-up mirror, its glass the cut-out missed, reads 63, every real hole of the trials at most 2.2). A clear pane shows the backdrop through it and cannot be told from a hole by colour; the lanterns' small panes are too small to be judged (see "Holes the picture shows") |
| Length of a sword or spear | Confirmed (a box's largest side was scaled to the length: a spear at 45 degrees came out 1.41 times too long) and fixed: a thing whose size is its length is set to it along its principal axis. Test: a spear made at 45 degrees is 2.0 m |
| Wrong paint passes | Confirmed for a black view (the coverage counts a texel painted whatever its colour): a new check, "painted views not black", fails a view more than 25 % black; on the trials' 654 views the most was 1.7 %. A view of near-uniform colour is not judged: the end views of a spear are near-uniform by right |
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

## Holes the picture shows (10 November 2026)

An early clay jug (made before the handle fix) was found with its handle's loop filled: it had passed every check. A filled hole thinner than the outline's gap piece (8 % of the height) or narrower than the carving's
least hole (7.8 %) was caught by nothing: a lantern's ring, a key's bow, the gaps between a chair's spindles.

**"holes the picture shows are open"** (counted). Each hole of the cut-out the shape model was given is judged when it
has an inner part (more than 1.9 % of the height, the outline's mean limit, in from its edge) of at least 50 canvas
pixels and shows the backdrop's colour: the share of that inner part the model's outline covers, seen from the camera
the carving finds (the outline check's single view search, the outlines framed by their boxes and compared by their
overlap with the holes' inner parts left out, so a web cannot pull the camera). The converse is judged too: each hole of
the model's own outline with such an inner part, and how much of it the picture covers (a hole the picture does not
have). A hole too small to judge is listed "not judged", never passed; when no hole was judged and the model has none
of its own, the check is shown, not counted. The sidecar's `measures.holes` lists every hole: its width, its inner
part, its colour against the backdrop's, judged or not and why, how much is covered; and the model's own.

Measured on every made job of the trials (on both cards) and the regression set, as the finish judges them:

| | Most covered | |
|---|---|---|
| models that pass every check | 6.4 % (the 12 GB card's chair, a gap of its back), 2.6 % (jugs), 1.7 % (chairs) | pass |
| the first trial's round table, carved | 8.3 % | passes |
| the early jug, its loop filled (as made, as `inspect --recut` judges a model made elsewhere) | 100 % | fails |
| made-up: a ring on a box filled by a web 1 cm thick (its hole 6.5 % of the height) | 100 % | fails |
| made-up: a chair's back with a panel between its spindles (gaps 5.6 cm, 5.9 %) | 100 % (each of five gaps) | fails |
| any model's own hole covered by its picture | 0 % | passes |

The limit, a third of the inner part, lies in that gap (8.3 % to 100 %), for both ways. The trials' lanterns show the
backdrop through a pane of glass between two bars, a hole 4.7 % wide whose inner part is 20 pixels, and the model's
glass covers it, rightly: under 50 pixels an inner part is not judged (its cover is as much an edge's misfit as a web;
the made-up ring's is 108). Its colour tells nothing there: through clear glass it is the backdrop's (2.2 delta E).

**Before carving, the hole's colour.** A hole whose colour in the picture (the median over the hole, CIE76 delta E in Lab)
lies more than 10 from the backdrop's just outside the object (12 to 48 pixels out) is a part the cut-out missed (a
mirror, a framed canvas), not a hole: it is never carved, and never judged open. On the trials every hole read at most
2.2 (most under 1; the jugs' loops 0.4, though the shadow darkens them, as the backdrop next to the object is darkened
alike); the made-up mirror whose glass the cut-out missed reads 63 and is left whole (before, its glass was carved
away: 47,680 triangles).

**A carved try ranks below an uncarved one** passing as many checks: the carving leaves a flat wall along the lines of
sight where the web was, and a seed that made the hole itself is better.

The regression set keeps `example-jug-handle-filled` (built from that jug, kept in the development history and not in
this repository: its picture cut again by our BiRefNet on the CPU, its `model.glb` read as the shape), which must fail
the check as made; `made-up-ring-filled` and `made-up-chair-back-panel`, which must
fail it, and their open twins, which must pass it; and `made-up-panel-miscut`, whose panel must not be carved.

What it costs: when the picture has a hole to judge, one more view search per judgement (the shape stage's tries and
the finish), about as long as the outline's; without one, the outline check's own view is read again.


## The shape model's own decoder (10 November 2026)

TripoSG's weights are MIT, at the revision the registry pins. The folder its repository's code sits in, `triposg/`,
holds the text of the Tencent Hunyuan FlashVDM Community License, which excludes the European Union, the United
Kingdom and South Korea, so no code of that repository is fetched or run. model-maker runs the weights with its own
code only: `python/model_maker/shape_net.py` holds the denoiser, and `python/model_maker/shape_vae.py` the decoder from
the latent to the field the surface is found in, the attention processor and the point embedding (docs/setup.md).

**What it is, and what it follows.** The attention processor follows diffusers' `AttnProcessor2_0` (Apache-2.0), with
each head taking its own run of the joined projections' channels (query, key and value joined in self-attention; key
and value in cross-attention), the order the weights were trained with. The point embedding is the frequency encoding
the TripoSG paper describes: the point, then the sines and cosines of each coordinate times 1, 2, 4, ... 128 (8
frequencies, `vae/config.json`). The decoder is built from the weights' tensor names and shapes on model-maker's own
block and diffusers' `Attention`, `FeedForward` and `FP32LayerNorm`: the latent's 2048 tokens through `post_quant` and
16 self-attention blocks, the query points looking at them in one cross-attention block, a layer norm and `proj_out`.
Only `decode` was ever called (the narrow-band search asks the field at points), so the VAE's encoder is not built: its
tensors (`encoder.*`, `quant.*`) are the only ones of the file not loaded, and a missing or left-over tensor is refused.
The FlashVDM decoder is neither written again nor used.

**How what the weights do not say was settled.** The upstream code was run as a black box, before it was removed
from setup: the arguments its VAE passed to our block were recorded (layer norms with eps 1e-5, no query and key norms, a layer norm on
the cross-attention's input), and its outputs compared with ours on the same inputs. Two things were found that way:
the heads' order (with diffusers' own order the output differed by up to 9.95 where it reached 6.2, on random inputs; with the joined order by 0), and the
field's sign (the old decoder gives the negative of `proj_out`, so the field is positive inside).

**Measured on the CPU**, the old modules against ours, each on the same inputs:

| What | Inputs | Largest difference |
|---|---|---|
| attention processor | random inputs and weights; self- and cross-attention; widths 2048 (16 heads), 1024 (8), 256 (4); no query and key norm, RMS norm, layer norm without bias; 32 and 16 bits (36 cases) | 0 in every case |
| the denoiser, one call | the chair's picture features, noise of seed 42, time 500, every weight real | 0 |
| point embedding | 60,000 points in the shape's cube | 0 |
| decoder's field, 32 bits | random latents (seed 42) and the chair's real latents (below); 60,000 points each | 0, its sign the same at every point |
| decoder's field, 16 bits (as on the card) | the same latents, 4,096 points each | 0 |
| the field with the latent's tokens worked out once (the new `surface`) | 20,000 of those points | 0 |
| a whole shape | the chair's real latents, the narrow-band search as the pipeline runs it (a coarse grid of 256, one finer level: 513) | identical: 296,942 corners and 593,904 triangles in both, every corner and triangle the same (largest corner distance 0); 1,391 s the old way, 1,037 s the new on 8 CPU threads |
| a whole shape, coarser | the same latents, a coarse grid of 128 and one finer level (257) | identical: 73,838 corners and 147,696 triangles, every one the same; 149 s and 110 s on 12 threads |

The real latents are the regression set's `chair-good`: its picture and cut-out through our denoiser on the CPU (32 bits,
seed 42, 50 steps, guidance 7; 830 s). The field is the same to the last bit, so every shape the pipeline made with the
old code it makes again with this code. The new `surface` also works out the latent's tokens (its 16 self-attention
blocks) once per shape rather than once per chunk of points, which gives the same field and saves that work on every
chunk (a quarter less time for the whole shape on the CPU). A full make on the card was not needed to prove the
equivalence. The regression set holds all its 41 cases with the new code (it judges stored shapes, so it checks that
nothing else moved), and the Python tests pass.

## Held things: the haft followed, the grip on the haft, a shield's handles (10 November 2026)

Eight objects made again showed three faults in the checks and marks of things held in the hand.

**The axe failed falsely.** "a double-bitted battle axe" failed "haft straight" (9.37 % against 1 %) and "no kink"
(28.4° against 4°), but its haft is straight within 4 mm. Each slice across the length was measured on its largest
piece; in one slice just under the head the largest piece was the lower corner of one bit (its middle 80 mm off the
haft, 44.8 mm wide, narrow enough to pass for haft). Now the haft is followed from slice to slice: from the middle of
the length outward, each slice is measured on the piece nearest where the line of the narrow slices before it points
(a slice with one piece is that piece, however far it lies, so a bent haft is followed round its bend and still
fails). A haft or a shaft is then its longest unbroken stretch: it stops where the head begins (the spike above the
axe's head is not haft). The axe now reads 0.5 mm (0.05 %) and 1.6°.

**The axe's grip mark sat on its butt.** The hand's middle was 14 cm from the end, on the neck of its black butt
piece (a flared knob, a neck 2.8 cm round under the haft's 3.3 cm, a collar), thin and round enough to hold. Now a
haft's grip begins, from the butt, at the first stretch a hand wide whose girth stays within 10 % of the haft's own;
the hand's middle is 8 cm in from there, on the wood: 26 cm from the end. The grip marks also lie on the grip's own
middle where they are (its slices' middles within 3.5 cm), not on the run's mean middle: a frame tilted by a
one-sided head put a made-up bearded axe's hold 3 cm off its haft, and the war hammer's grip line ran off its haft.

**The shield's grip mark was between its handles.** The grip was on the middle of the back by convention; the remade
shield has two D-handles, left and right of its middle, so the grip lay on bare boards. Now the handles are found: the
board's back is a smooth surface (quadratic) fitted to the backmost point of each cell (1/40 of the shield's size),
what stands more than 1 cm off it left out of the fit, four times; what stands more than 1.2 cm off it is grouped into
pieces, and a piece is a handle when it stands at least 3 cm off, its bar (what stands at least 70 % as far off as its
furthest) is at least 7 cm long and 1.5 to 5.5 cm round across its middle third, with at least 1.5 cm of open space
under that middle third for the fingers. With two handles the hand holds the one on the right as seen from behind
(-X), the forearm through the other: the convention of a shield carried on the left arm, the forearm across the back;
of two handles one above the other, the upper. "handle found" (shown, not counted) names the handle held; with none,
it fails, saying so, and the grip stays on the back's middle by convention.

### What did not work

* Taking the haft proper as its longest even stretch (girth within 10 % of its own middle) cut the remade spear's
  shaft at a band 13 % thicker, 61 cm from its butt: its grip shrank from 157 to 53 cm and its hold left 40 % of its
  length. The butt end alone is trimmed now; a band or a binding further up is held over.
* A straight line fitted through the grip's slices placed its marks on the haft's line, but a bow's grip curves: the
  first trial's bow (which fails anyway) had its grip 9 cm off its stave. The marks take the slices' middles near them.

### The trials finished again

Every held thing of the trials with a kept shape (31 jobs: the first trial and its refinish, the fix-check, the
third and fifth trials, the remake; the 12 GB card's trials keep no shape and their bows were not made) finished on the CPU
with the code before and after, and every verdict and mark compared:

| Model (jobs) | Verdict | Marks, judged by eye |
|---|---|---|
| remade axe (1) | fails "haft straight", "no kink" → passes every check | grip from the black butt piece's neck (14 cm from the end) to the wood above its collar (26 cm) |
| the trials' axe (4) | passes, as before | the same: its butt piece has a collar too; 14 → 26 cm |
| war hammer (4) | passes, as before | the hold 4.7 cm further up (12 → 16 cm), still on the wood just above its cap; its grip line now on the haft, before it ran off it |
| shields (6) | passes, as before; "handle found" new | grip from the boards between two handles to the right-hand one seen from behind: 13.5 cm (the trials' shield), 15.5 cm (the remade) |
| broom (4) | passes, as before | 2 to 4 mm across, onto the handle's middle |
| first trial's bow (2) | fails, as before | the hold on the stave where it was 9 cm off it |
| swords, spears (10) | as before | as before (the remade spear's grip 156.6 → 154.9 cm long, its hold where it was) |

The 12 GB card's trials' shield (not re-finishable) has three pieces on its back: a long bar in the middle, which the hand
holds, a handle to its left seen from behind (the forearm's), and a stub 6 cm long, too short for a hand.

The regression set keeps the remade axe (`axe-head-corner-beside-haft`: straight, no kink, its grip 20 to 32 cm from
the butt) and shield (`shield-two-handles`: two handles found, the grip on one), and made-up cases: a haft bent 6° at
its middle (it must fail straight and kink), a bearded axe (its beard a larger piece than the haft beside it: it must
pass), an axe with a butt cap like the remade one's (its grip 19 to 26 cm from the butt) and a shield with two handles.
With the code before, the remade axe, the bearded axe, the butt cap and both shields break; the bent haft fails both.

## Several things, and qualities the words name (11 November 2026)

Release 1's unseen run made two objects the prompt did not describe:

* **"a pair of leather boots" made one boot.** Three of its four picture tries drew two boots, touching, so each
  passed "one piece" as one lump of two; the fourth drew one boot, scored best on the other picture checks and was
  chosen. Either way the words were not met: release 1 cannot make two boots as two objects.
* **"a short curved dagger" came out straight.** None of its four tries was curved, and nothing checked a quality the
  prompt names.

Those ten prompts were looked at to understand the two faults; nothing below was measured or tuned on them.

### Several things: answered before anything is painted

Release 1 makes one object in one piece. A prompt that asks for more than one thing is now answered at once, before
any card time, with the one thing to ask for instead (`words.several_of`):

    model-maker makes one object at a time, and these words ask for more than one. Try one: "a leather boot".

It asks for several when its head phrase (the words before "of", "with", "in", "for", "by", ...):

| asks for several | example | suggested |
|---|---|---|
| groups things | "a pair of boots", "a set of three knives", "a couple of mugs", "a collection of bottles" | "a boot", "a knife", "a mug", "a bottle" |
| counts them | "two wooden chairs", "2 lanterns", "a few goblets", "three iron swords with gold hilts" | "a wooden chair", "a lantern", "a goblet", "an iron sword with gold hilts" |
| names a known thing in the plural | "boots", "leather boots, brown", "knives", "bookshelves" | "a boot", "a leather boot, brown", "a knife", "a bookshelf" |
| joins two known things | "a sword and a shield", "a table and chairs" | "a sword", "a table" |

What is made as before, each a case in the shared prompt cases:

* things that are one object though named as a pair or a set: "a pair of scissors", "a pair of iron tongs", "a pair
  of glasses", "glasses", "trousers", "a pair of bellows", "a set of drawers", "a set of bagpipes";
* a count that describes: "a two-handed sword", "a two handed sword", "a three legged stool", "a four-poster bed", "a
  20 cm dagger", "a 200 year old chest", "a bench for two people";
* a plural after "of" or "with", which is what a thing holds or is made of: "a bag of coins", "a chest of drawers",
  "a basket of apples", "a ring of keys";
* a pile, heap, stack or bundle, which is one object (its words already make it not mirror-symmetric): "a pile of
  rocks", "a stack of books", "a bundle of arrows";
* two words joined where one is no known thing: "a black and white chair".

A plural is known by the word lists: a thing's word with s, es (after s, x, z, ch, sh: "boxes"), ves ("knives",
"shelves") or its own ("staves"). A plural of a word the lists do not know is not caught ("three widgets" is, by its
count). The suggestion makes each known plural one and drops the count, the group and the article; with no known
thing it makes the last word one by the plain rules ("a pair of earrings": "an earring"), and when that word is no
plural it suggests nothing ("a set of armour": "Ask for one of them."). A suggestion that would itself ask for several
or for a character is not given.

`python/tests/prompt_cases.json`, shared with the private service's Rust copy, has a fifth verdict, `several`, and 55
new cases (26 `several`, 29 `make`); no case changed. The Rust side must read the same cases the same way.

Not caught: "a table with four chairs" (the head phrase ends at "with": a table is made, and the chairs are left to
the picture checks).

### Qualities the words name: judged in the picture

A quality the prompt gives the whole thing is now judged in "shows the thing asked for", as the state check judges a
lid (`words.qualities_of`, `subject_judge`). Each quality has two pairs of descriptions, the same evidence both ways,
the thing's word kept in each (classification by description, as the state check's words):

| Quality | Asked | Its opposite |
|---|---|---|
| curved | a curved {thing}, its whole length bending in one smooth arc | a straight {thing}, its whole length one straight line |
| | a {thing} with a curved outline, bent like a crescent | a {thing} with a straight outline, its edges parallel lines |
| round ("round", "circular") | a round {thing}, its outline a circle | a square {thing}, its outline a square with four corners |
| | a round {thing} with no corners, curved all the way round | a square {thing} with straight sides meeting at corners |

A pair's share is the sigmoid of the two scores' difference, averaged over the pairs and over the two views (the whole
picture and its crop round the cut-out), as the state check's is. The thing asked for and each quality must each reach
**0.75**, the limit of both checks, and the check's value is the least of their shares; its words give each share
("... with a share of 99% against the likeliest other thing, a stool; round with a share of 100% against square").
So a try without the quality is not chosen, and when no try of two rounds has it, the job ends at its pictures, failed
on that check, before any shape is made.

A quality counts when its word is in the head phrase, said of the whole thing: not denied ("a sword, not curved"), not
joined to the next word ("a round-bottomed flask", "a curved-handled cane"), not followed by a past participle ("a
round bottomed flask"; a word in -ing is the thing's use: "a curved hunting dagger" is curved), not part of the thing's
own word ("a short sword") or kind (a sabre, a scimitar, a sickle is curved by its kind), and not contradicted ("a
round square table"). A quality after "with" names a part ("a chest with a round
lid") and is not judged.

Two changes came with the curved blade:

* **A blade the words call curved is a curved blade.** "a curved dagger" is checked as a sabre is (its curve without
  a kink; no straight line, no mirror across it): before, it was a straight blade, whose "blade straight" check a
  curved dagger fails. In the picture judge it is judged with the curved blade's description ("a curved blade with a
  hilt") and not against the sabre: with the straight blade's description and the sabre as a contrast, the first four
  tries of "a curved bronze dagger" labelled curved scored 0.03 to 0.69 as the thing asked for, the likeliest other
  "a sabre".
* **A curved blade is painted with its curve in view**, as a bow is: "its blade curving in one smooth arc like a
  sabre's", and the view "standing upright, its point at the top, seen from the side so its whole curve shows". The
  picture model seldom paints a curve. Tries labelled by eye, the same six prompts ("a curved bronze dagger", "a curved
  hunting knife", "a curved iron sword", "a curved silver dagger", "a curved short sword", "a curved steel sabre"), 8
  each: with release 1's words 1 of 48 curved (one bronze dagger), with these words and view 7 of 48 (the hunting
  knife 4 of 8, the bronze dagger, the iron sword and the short sword 1 each, the silver dagger and the sabre none).
  Tried and not kept: a description of the curve alone, the view unchanged ("its whole length bending in one smooth
  arc": the bronze dagger 3 of 8, the iron sword none). A small sample, and still few curved tries: a job whose tries
  never show the curve now ends at its pictures, failed on "shows the thing asked for", where release 1 made a
  straight dagger.

### Measured

New picture tries, painted with the pipeline's own picture words and seeds (`tools/trial/paint_tries.py`): 384 on both
cards, 8 for each prompt and wording. The prompts name a quality ("a curved hunting knife", "a round mirror", "a tall
clay vase") or its opposite or nothing ("a straight bronze dagger", "a hunting knife", "a square mirror", "a low
wooden stool", "a wooden table"). None is one of release 1's unseen ten. Every try was labelled by eye before its
score or prompt was seen: cropped round its cut-out and shuffled under a random code (`tools/trial/quality_sheets.py`).
"tall" and "squat" were labelled by the cut-out's proportions (tall at least twice as high as wide, squat at most 1.2
times, unclear between), "square" as four corners, square or oblong. Each picture was judged against the prompt that
names the quality: its own, or its twin's ("a hunting knife" as "a curved hunting knife").

| Quality | Showing it, passing the pair | Showing its opposite, refused by the pair | The whole check passes |
|---|---|---|---|
| curved: daggers, swords, a short sword, benches | **9 of 9** (lowest 0.77) | **98 of 98** (highest 0.60) | 4 of the 9 curved |
| round: boxes, mirrors, trays, tables, plates, shields | **48 of 48** (lowest 0.997) | **31 of 31** (highest 0.31) | 46 of the 48 round |

The whole check also judges the thing. It refuses 5 of the 9 curved tries for that: each is a bare curved blade with
no guard or grip to speak of, taken for a scythe's or a spear's blade (0.002 to 0.32); and one round shield and one
round mirror (0.35, 0.61, the likeliest other "a coin"), as release 1's check did. The pair is not to blame there.

The 48 tries painted after the wording was chosen ("a curved hunting knife", "a curved silver dagger", "a curved short
sword" and "a curved steel sabre" with the words and view kept, the dagger and the short sword also with release 1's
words) were held out as a check of it: every straight dagger and short sword is refused (28 of 28), the one curved
short sword passes the pair (0.99), and the curved knives showed the limit below. The sabres are not judged for the
curve (their kind is curved).

A knife's curve is not judged. Of the 7 hunting knives labelled curved (their blades sweep up, their handles
straight), the pair refused 5 (0.07 to 0.56); the straight knives were all refused (24 of 24, highest 0.45). A second
wording was tried for every blade on the first tries, the blade named in place of the whole length ("a curved {thing},
its blade bending in one smooth arc" against "its blade one straight line"): it refused the same two knives (0.46,
0.53) and brought two straight swords closer to the limit (0.71, 0.73 against 0.58, 0.60); it was not kept. A
curved knife is still a curved blade, painted with its curve in view (4 of 8 of the held-out knives came out curved),
and its picture is judged for the thing only (`words.QUALITY_NOT_JUDGED`).

The trials' own pictures whose prompts name a quality kept (`tools/trial/judge_tries.py --only-qualities`, every
distinct picture of every trial, labelled by its own prompt): "a small round table" (20 pictures, round 0.96 and over)
and "a round wooden shield with an iron boss" (20, round 0.99 and over). None is newly refused. The 5 round tables
labelled unclear in the prompt cases (seen from straight above, a table and a stool alike) are refused for the thing,
as before; their round shares are 0.96 and over too.

Measured and left out, on the first 336 tries and the trials' own pictures (each quality by two pairs, as written):

| Quality | Showing it, passing | Showing its opposite, refused | Why left out |
|---|---|---|---|
| tall ("a tall {thing}, much taller than it is wide" against "a squat {thing}, wider than it is tall"; "a tall narrow {thing}, high and slender" against "a low squat {thing}, short and broad") | 32 of 46 | 17 of 17 | stools, vases, bottles and candles pass, but lanterns (4 of 13) and cabinets (0 of 4) twice as high as wide are refused: "tall" is judged against the thing's usual proportions, and a lantern is tall already |
| squat (the same pairs turned round) | 10 of 17 | 46 of 46 | low stools refused (0.40 to 0.50) |
| square (the round pairs turned round) | 29 of 31 | 48 of 48 | two square boxes seen in perspective refused (0.69, 0.70) |
| straight (the curved pairs turned round) | 82 of 84 | 8 of 8 | a straight sword (0.43) and a straight sabre (0.74) refused; the picture model paints blades straight anyway |
| broken ("a broken {thing}, a part of it snapped off or smashed" against "a whole {thing}, every part of it intact"; "cracked and damaged" against "an undamaged {thing} in good repair") | 0 of 1 (0.57) | 12 of 12 | the picture model painted 1 broken chair in 8 tries of "a broken wooden chair": too few to measure |
| rusty ("a rusty {thing}, its metal covered in orange-brown rust" against "a clean {thing}, its metal bare and free of rust"; "a corroded {thing}, rust flaking from it" against "a polished {thing}, shining like new") | keys 8 of 8 (0.80 and over), the trials' "a rusty iron lantern" 17 of 20 | 8 of 8 (0.04 and under) | three of the trials' rusty lanterns, rusty by eye, refused (0.64, 0.64, 0.70) |

Known limits: a size ("small", "big", "short", "long") is not judged, as a picture has no scale (`--size` sets it); a
quality of a part ("a chest with a round lid") is not judged; a knife's curve is not judged; "open" and "closed" are
judged for things with a lid, cover or doors only (the state check). The picture model seldom paints a curved blade,
and a bare curved blade with no guard or grip is refused as the thing: most jobs asking for a curved dagger or sword
end at their pictures, failed on "shows the thing asked for", rather than making a straight one.

The labelled set is a regression case, "quality-pictures" (`tools/trial/quality-cases.toml`, 240 pictures, the curve
and round tries): judged against the prompt that names the quality, the quality's pair must pass a picture showing it
and refuse one showing its opposite; the whole check is shown beside it, and "unclear" is shown, not held. The whole
regression set holds (49 of 49 with it; "quality-pictures" took about 8 minutes on 12 CPU threads). In
"prompt-pictures" one written prompt fewer passes, a known miss shown and not held (23 of 30, 24 before); every
other number is as before (own prompts lowest 0.926, other prompts highest 0.500).
