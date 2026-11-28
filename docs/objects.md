# 3-D objects

model-maker makes 3-D objects, not animals or characters: "an old looking chair", "a rusty iron lantern",
"an old iron sword", "a double-bitted battle axe". Animals come in a later release, characters (people and fantasy
creatures) after the animals.

```
./model-maker make "an old looking chair"
./model-maker make "an old iron sword" --size 0.9
```

Each object gets a folder with its `.glb`, its sidecar (`<name>.provenance.json`: the prompt, every
model used with its version and licence, the seeds, every check) and its report
(`report/report.html`).

This document is the reference: the steps, how a prompt is read, every check with its limit, the marks, the content
credentials, where the models are, what runs without a card, and offline. What each limit and fix was measured on,
what did not work, and the trials' faults are in [`docs/history/`](history/README.md), one file per topic.

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
small (at most 24 edges and 5 % of the object's size round); hollows sealed inside it dropped; a loose piece that
comes within 3 mm (and 1 % of the object's size) of the main piece moved by that gap to touch it (a candelabra's arm
the shape model left a hair apart was refused by "one main piece"; a gap that shows is left).

Settings worth knowing: `--tries`, `--attempt N` (a refused object made again: its pictures, shapes
and views from seeds after every seed the attempts before it used; 0, the default, draws the same seeds as before),
`--require-lettering` (every picture must be read for painted
lettering: without a usable tesseract no job is made; the measured runs use it).

`--tries` and `--texture` are not in the job folder's name either. A run with another `--tries` on a folder whose
pictures are painted is refused, saying how to go on (the same `--tries`, `--attempt N` or another `--out`); a `batch`
run again with another `--texture` bakes again, as `make` does every time.

The trials' settings are not options of `make` and `batch`: the lead words, the view painter's steps (20), the shape
tries, the picture rounds, the triangle budget and the touch-up's views and strength. They are not in the job folder's
name, so as options a second run with another value found its stages done and returned the first object. Every
published number is for their defaults; the trial tools and `reproduce` set them through `pipeline.Settings` in
Python ([measured limits](history/measured-limits.md), "The steps").

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
* its usual size: the largest side of an object, the length of one held in the hand. `--size` sets it. Since 0.3.0
  about 60 names of things of today are known as well (a smartphone, a laptop, a sofa, a fridge, a bicycle,
  sneakers), each with its usual size, whether it stands or lies (a robot vacuum lies on its broad face) and whether
  it is mirror-symmetric; sneakers and running shoes are a left and a right, and "a pair of headphones" is one thing;
* for a thing with a lid, a cover or doors (a chest, trunk, box, crate, book, cabinet, wardrobe),
  whether it is shut or open: shut unless the words say "open", "opened" or "ajar" (and not "never opened", "not
  open", "cannot be opened": a negation up to three words before the word denies it). The picture model
  is told so after the prompt ("its lid shut"; a book "lying shut, only its cover showing").

* whether it asks for two of one thing ("a pair of boots", "two candlesticks"): such a pair is made once, as "a boot",
  and placed twice in the .glb, the second the first's mirror image when the two are a left and a right (boots,
  shoes, gloves, gauntlets) and an exact copy otherwise (below, "Pairs");
* whether it asks for more than one thing otherwise: a count ("three chairs", "3 swords"), a group ("a set of knives"), a known thing in the plural ("leather boots") or two known things joined ("a sword and a
  shield"). model-maker makes one object at a time, so such a prompt is answered before anything is painted, with the
  one thing to ask for instead: `model-maker makes one object at a time, and these words ask for more than one. Try
  one: "a leather boot".` A thing that is one object though named as a pair or a set is made ("a pair of scissors",
  "a pair of tongs", "glasses", "trousers", "a set of drawers"), and so is a count that describes ("a two-handed
  sword", "a three legged stool"), a plural after "of" or "with" ("a bag of coins", "a chest of drawers") and a
  pile, heap, stack or bundle (one object). See [Several things, and qualities the words
  name](history/several-and-qualities.md);
* the qualities its words give it that the picture must show ("a curved dagger", "a round box"): see the same
  section. A curved blade ("a curved dagger") is checked as a sabre is (its curve without a kink; no straight line,
  no mirror across it).

`--stands yes/no` and `--symmetric yes/no` override the words.

**Bows are not made yet**: a bow prompt
is answered that they come later, and nothing is made. A prompt for an animal is answered that animals come in a
later release, and one for a person or a fantasy creature that characters come after the animals; a prompt that names
both ("a knight on a horse", "a horse with a knight") is answered as a character, the later release, and a statue
or a carving of either is an object. An animal made a fantasy creature ("a zombie dog", "a skeleton horse") is a
character; a person only described ("a horse for a knight", "a horse and a statue of a knight", "a cat like a king")
leaves an animal an animal; what is left of an animal ("a fish skeleton", "a wolf skull") is an object; and a head word
the pipeline does not know does not hide the creature it is on ("a sheriff on a horse" is answered as an animal).
How each prompt is read is
tested on one file of cases (`python/tests/prompt_cases.json`), which a program in front of the pipeline can test
its own reading against.

## The checks

Every check has a measure and a limit, written in the report beside what it means.

Pictures:

| Check | Passes when |
|---|---|
| one piece | the cut-out's largest piece holds 97 % of it |
| plain backdrop | at most 2 % of the backdrop is something else, more colourful than it (its own grey in shadow, on a floor or in a vignette is judged in the cut-out) |
| no shadow or floor in the cut-out | at most 0.5 % of the cut-out is backdrop taken in with the object: backdrop-coloured pixels in its lowest tenth, joined to a stretch of its lower edge with no colour step across it (the edge need not run through a shadow on the backdrop: the stricter reading) |
| not cut off | the object keeps 1 % of the picture from every edge |
| shows the thing asked for | SigLIP 2 finds the thing asked for (the prompt and its kind's description) at least three times as likely as the likeliest other kind model-maker makes or a distractor (a person, text, an empty backdrop, several things), from the whole picture and its crop round the cut-out (see [The picture against the prompt](history/picture-check.md) and [War hammers](history/war-hammers.md)); and each quality the words name ("curved", "round") at least three times as likely as its opposite (see [Several things, and qualities the words name](history/several-and-qualities.md)); without SigLIP 2 the tries go unjudged and the run says so |
| shows what the words ask | a thing with a lid, cover or doors: SigLIP 2 finds it shut (or open, as asked) at least three times as likely as the other way, judged by descriptions of what each state looks like ("the edges of its pages in one even block" against "split in two at the middle"), from the whole picture and from its crop round the cut-out |
| no painted lettering | tesseract reads the cut-out's region, and its words that count (read with a confidence of 70 or more, at least two letters and three letters and digits, at least 8 pixels of 1024 tall) hold at most 4 characters between them; a prompt that asks for lettering ("a sign that says OPEN") is not refused. Without a usable tesseract (optional: version 5 or newer with its English data) it says "not judged" and why, shown, not counted, and the job says it was not read (see [Painted lettering](history/lettering.md)) |

Shown, not counted: the shadow on the backdrop (how much darker it is near the object than far from it) and a floor
line on it. What would harm the model is a shadow inside the cut-out, and that is what is checked. The measure cannot
tell an object as pale and grey as the backdrop from a shadow: those fail safe, and another try is painted
([measured limits](history/measured-limits.md), "The checks").

Objects:

| Check | Passes when |
|---|---|
| outline matches the picture | judged against the cut-out the shape model was given (its largest piece, its pinholes filled), from the view that matches best (its turn, its tilt and the camera's distance: straight on, or through a lens 6, 4 or 3 times the object's size away, as the picture is a photograph; the views compared by the edges' mean distance, each camera distance refined; the tilts tried from level to 24° closer together for a deep, low thing, see [False refusals](history/false-refusals.md)), on a canvas as wide as the picture needs (a model wider still from that view fails), the outlines' edges lie at most 1.9 % of its size apart on average and 4 % at the 95th percentile (its size: its height, or for a thing wider than high the square root of its height times its width, so a long, low bed frame or rolling pin is not refused for a few pixels at its ends) (a compact shape offset all round by 1.9 % overlaps its picture 85 %), and no piece one outline has in a gap of the other (a web between legs) is thicker than 8 %; the overlap itself is shown, as it is unfair to thin parts: a chair leg a pixel off counts wholly against it |
| holes the picture shows are open | each hole of the cut-out with an inner part (1.9 % of the height in from its edge) of at least 50 canvas pixels, showing the backdrop's colour (within 10, delta E), is covered by the model's outline at most a third, seen from the camera found with those inner parts left out; and no hole of the model's own outline is covered by the picture more than a third. Counted only when a hole was judged; holes too small say "not judged" (see [Holes the picture shows](history/holes.md)) |
| no holes | no edge is used by one triangle only |
| one main piece | the main piece is 98 % of the surface; loose pieces (touching nothing) are listed |
| every foot on the ground | no foot (a low point of the underside, lower than all round it, well outside where the others touch) stops short of the ground by more than 0.5 % of the height |
| base level | set down, it settles at most 2° onto the face of its hull under its centre of mass |
| centre over the base | its centre of mass lies 15 % of the base's radius inside where it touches the ground |
| closed underneath | (a thing that stands on its own base) seen from straight below, at most 40 % of what is seen is a hollow closed in all round by the base; the space under a table, which reaches the outline, does not count (see [Open underneath](#open-underneath)) |
| left-right symmetry | mirrored across its best upright plane, it lies at most 2 % of its size from itself on average, and at most 5 % at the 98th percentile (measured to its triangles: a leg missing, out of place or on one side) |
| paint covers the surface | the painted views coloured 90 % of the surface as it is seen: each texel counted by the share of 32 directions round the object it is in view from (the inside of a jug's neck counts little); the plain share of texels is shown beside it |
| fill matches the paint round it | a note, not a refusal, since 0.3.0: it is shown in the report with its measures and the picture of the touch-up view where the fill fits worst, and counts toward nothing (alone it had refused 3 right objects of 50 on the 12 GB card and 4 on the 16 GB card, judged blind). It would pass when (the paint was touched up) judged on what the finished model shows, the fill kept: in no touch-up view is more than 10 % of it a colour found nowhere in the paint round it (each filled point against the 32 painted points nearest it on the same surface, foreign when more than 20, delta E, from every one of them), and no joined foreign place is larger than 3 % of the object's size. The places left unpainted (the fill the touch-up left out, which shows the colour spread from round it): none larger than 3 % is a smear, where neither the paint round it nor the colour it shows is one colour (less than 90 % of each within 20 of its median). The fill as the inpainter painted it, before what was foreign was left out, is shown, not counted |

The outline is measured against its own size since 0.3.0, so a long, low thing is not refused for a few pixels at its
ends. A known limit came with it: a wooden rolling pin with lettering stamped on its roller (a paint fault) had been
refused by the outline only by chance, and now passes every check, the question judge saying yes (0.80). No check reads
lettering painted on the finished model (the lettering check reads the picture); one is later work.

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

## Open underneath

Since 0.3.0 a thing that stands on its own base (a chest, a chair, a table, a cabinet: the words decide, see "How a
prompt is read") is refused when it is open underneath: a box made with no bottom, its hollow seen from below.

What it measures (`object_checks.ringed_hollow`): the object is drawn from straight below on a grid of 200 by 200
cells over its footprint, each cell keeping how high the lowest surface above it is. A cell more than 15 % of the
object's height above its lowest point is part of a hollow. A hollow that reaches the outline of what is seen from
below is the space under a table or between a chair's legs, and does not count. The largest hollow closed in all round
by the base, as a share of all that is seen from below, is the measure; over 0.4 (40 %) the check "closed underneath"
fails, with "it is open underneath" in the report.

The limit was measured on 309 stored objects made on both cards: no object judged right came over 0.4
([measured limits](history/measured-limits.md), "Open underneath").

The check also takes part in choosing between shape tries: a try open underneath loses to one that is not.

Known limits:

* a right thing whose base is closed all round, with its underside more than 15 % of its height up, is refused: it
  looks from below like a box with no bottom. Measured on plain test shapes: a chair or a stool with low stretchers on
  all four sides (0.76 and 0.69), a chest on a deep skirt, its bottom 12 cm up of 60 (0.80, the same as no bottom at
  all), a platform bed with rails to the floor (0.92), a sofa with a skirt to the floor (0.90), a cot with low rails
  all round (0.85). None of the 309 stored objects was such a thing. `--stands no` leaves the check out. The tests keep
  these shapes as the check measures them now;
* a bowl-like open side that reaches the outline is not caught, as it looks the same from below as the space under a
  table: a stone pestle made as a mortar passes it;
* it runs only for things that stand on their own base, so a thing of another kind or of no known kind (a toolbox, a
  mortar, a step ladder) is not checked;
* it was chosen after the 0.3.0 measurement's list was opened (`results.md`): on that list it is a recount, not a
  measurement.

## Pairs

Words that ask for two of one thing and nothing more ("a pair of boots", "two iron candlesticks", "a pair of boots
with brass buckles") make the one ("a boot") and place it twice. It is checked, judged and painted as one object; the
question judge is asked about the one ("Is it a boot?"), as the pictures show one. The `.glb` holds its mesh once
and two nodes:

* a left and a right (boots, shoes, gloves, gauntlets, sandals, slippers, clogs, mittens, greaves, vambraces,
  bracers, sabatons, spurs): the second is the first's mirror image, an exact match. For footwear it stands on the
  side the first's arch faces (the footprint's middle third is inset more there), so the inner sides face each other;
* anything else: an exact copy.

The two stand side by side across the object's narrower width on the floor, a small gap between them, toes the same
way; a flat thing (a dagger standing on its point) stands beside its pair in its own plane, not face to face. The
second carries the first's marks and parts (a dagger's grip) as its own children.

Anything more is answered as before, one object at a time: a set or three, a second thing ("a pair of boots and a
hat"), a count ("two dozen eggs"), where the two stand or how they differ ("two chairs at a table", "two swords
crossed", "one red and one blue"), lettering (a mirror would turn the letters round), or a name whose one cannot be
said ("a pair of boot racks").

## The marks

The `.glb` of a thing held in the hand has named empty nodes under the object's node: `grip` (where the hand's
middle goes), `grip_start` and `grip_end` (the grip's line), and `tip`; a shield's has `grip`, `grip_start`,
`grip_end` and `front`, and no `tip`. Each grip mark lies on the grip's own middle where it is (its slices' middles
within 3.5 cm of it), so a frame tilted by a one-sided head puts no mark off the haft. A shield's grip marks lie
along the bar of the handle the hand holds, and its nodes' +Y runs along that bar.
Each node's +Y points toward the tip and its +Z is the blade's flat side, so a game puts the hand's
palm at `grip` with its fingers round +Y. The sidecar lists them too.

## Content credentials

Every picture model-maker writes carries content credentials: a signed C2PA manifest inside the file that says it
was made by AI models, and by which. That is the picture tries and their cut-outs, the chosen picture, the six painted
views and the touch-up's pictures, the texture inside the `.glb`, and the report's pictures. It is on by default and
needs no network. Please do not strip it: a picture shown or passed on without it no longer says it was made by AI.

A manifest holds:

* one action, `c2pa.created`, with its digital source type from the IPTC vocabulary: `trainedAlgorithmicMedia` for a
  picture that is the models' work alone (a try, a painted view, a turntable of the made object),
  `compositeWithTrainedAlgorithmicMedia` for a report picture that draws model-maker's own marks over it (the outline,
  the grip and tip, the paint's coverage), and `algorithmicMedia` for a touch-up's mask, which our own code makes;
* the generator, model-maker and its version, and the models whose work the picture shows, by name;
* the signature and its certificate chain.

It never holds the prompt, a path, a file name or the user's or the computer's name. The prompt is in the object's
sidecar: a picture travels alone and its manifest is signed, so what is in it could not be taken out again. The
sidecar's `content_credentials` and the report say how many of the job's pictures carry credentials that hold,
whether a test certificate signed them, and any picture whose mark no longer holds (changed since it was signed).
`model-maker finish` signs the report's pictures it writes, and any picture of the job made before pictures were
signed (their pixels as they are). A bug report keeps each PNG's credentials. The `.glb` also says in plain words, in
`asset.extras`, that it was made by AI models: a mesh need not be marked, but a note costs nothing.

**The certificate.** `--signing-key` and `--signing-cert` (or `MODEL_MAKER_SIGNING_KEY` and
`MODEL_MAKER_SIGNING_CERT`) give your own key and certificate chain, both PEM; without them the pictures are signed
with a test certificate `setup.sh` makes on your computer (`.tools/c2pa-test`, its key readable by you alone). C2PA
checkers report a picture signed with it as signed by an untrusted certificate. That is true: the mark is still there
and still machine-readable. `docs/setup.md`, "Your own signing certificate", says how to get one checkers trust. Your
own certificate's names (its organisation, for one) are shown with every picture it signs. `MODEL_MAKER_TIMESTAMP_URL`
names a time-stamping authority, so signatures stay valid after the certificate ends (every signing then uses the
network).

    ./model-maker credentials                     which certificate signs, and a trial signature read back
    ./model-maker credentials --read PICTURE...   what each picture's manifest says

Pictures made before (a website's, this README's) get their credentials with `tools/sign_pictures.py`, from a list
that names each AI-made picture and why; every other picture is left alone (`docs/images/made-by-ai.toml`).

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
exactly that. A bowstring is below what the shape model can see; what the finish has ready for bows, and how the first trial's
thin parts came out: [thin parts](history/thin-parts.md).

## The regression set

Every model a check caught, and every case where a check once misread a good model, is kept with the verdicts it
must keep (`tools/trial/regression.toml`), with labelled pictures for the picture checks (`state-cases.toml`,
`prompt-cases.toml`, `quality-cases.toml`, `lettering-cases.toml`); every case is listed in
[the regression set's cases](history/regression-set.md). `tools/trial/regression.py run DATA` judges them all on
the CPU in a few minutes; it runs before any change to the pipeline or its checks, and every case must hold.

The cases and their verdicts are here (`regression.toml`, `state-cases.toml`, `prompt-cases.toml`, `head-cases.toml`,
`quality-cases.toml`, `lettering-cases.toml`),
and so are the sums of their data files (`regression-sums.json`), but the data itself (the made shapes and pictures,
built from the trials' folders) is not in this repository: a reader can see what each case holds and why, but cannot
run `regression.py build` or `run` without it (the faults its made-up cases hold are tested in `python/tests/`).
`tools/trial/regression-prompts.txt` makes the same objects again on the card before a release.
`tools/trial/unseen-final.txt` holds a set of prompts kept aside during development and made once for the release's
check, with `--require-lettering` (as `tools/trial/run_trial.sh` makes its trial), so no pass is counted with its
pictures unread.

## Where the models are

`--models DIR` (or `MODEL_MAKER_MODELS`) holds a folder per model: `flux1-schnell`, `birefnet`,
`triposg`, `mv-adapter`, `sdxl-base-1.0`, `sdxl-vae-fp16-fix`, `siglip2-so400m-patch14-384`. `--sources DIR` (or
`MODEL_MAKER_SOURCES`) holds the view painter's own code, `mv-adapter`, at the commit `registry/models.toml`
names. The shape model needs no code of its own repository: model-maker runs TripoSG's weights with its own denoiser
and decoder (`shape_net`, `shape_vae`). Before a model is
loaded, every file the registry lists is checked against its sha256, and a folder with code, a pickle or a
configuration (a `.json` file, a `.jinja` chat template) the registry does not list, or with none of its listed files,
is refused. Code is anything Python imports a module from (`.py`, `.pyc`, `.pyo`, `.pyw`, and the extension modules
`.so` and `.pyd`, which Python tries before a `.py` of the same name) and a pickle under any of its usual names
(`.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl`, `.pickle`, `.joblib`); the hub's cache folder (`.cache`) is walked like the
rest. A linked folder inside a model folder is refused, as the check does not follow it (setup makes none; the model
folder itself may be a link), and so is a folder the check cannot read. Such a refusal says to remove the file or set
the models up again with `./setup.sh` in an empty folder: setup writes only the registry's files into a model folder.
Before MV-Adapter's code is imported, its checkout is checked to be at the registry's commit with nothing
changed or added. `--no-verify` skips reading the files for their sums only: the folders are still walked, and what
the registry does not list is still refused (0.3.1).

## Without a card

The checks, the bake, the export and the report run on any computer:
`./model-maker inspect MESH --picture PNG --prompt TEXT --out DIR` checks a mesh made elsewhere
against its picture, `./model-maker finish DIR` redoes step 4 for a made object, and `./model-maker repair-texture`
mends the texture of a `.glb` made before 0.3.2 ([Thin dark strokes](history/thin-dark-strokes.md)).

With `--recut` (for a model made on another computer, its picture and cut-out sent with it) the picture is cut again
by our own BiRefNet on the CPU and every check is judged against that cut-out, never the one sent, so a cut-out made to
fit the model (a web between legs left out, say) cannot pass the outline. The one sent must also agree with ours, their
overlap at least 97 % of their union (*the cut-out sent is the picture's*; [measured
limits](history/measured-limits.md), "Without a card"). Without any cut-out `inspect` tells the backdrop by colour,
and its shadow counts as the object: of eight good models, 6 then failed.

## Offline

`HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are set to 1 by the `model-maker` command before Python starts (and by the
package's `__init__`, before any of the hub's libraries is imported, so for every way in: the command, setup's smoke
test, the trial tools, a program that imports the pipeline), whatever the environment held: a 0 left there is
overwritten, so no configuration can fetch code from the hub (0.3.1; [defects found by
measurement](history/defects.md), 20).

## History

How each check, limit and fix came to be, dated as written, with what did not work: [`docs/history/`](history/README.md).
