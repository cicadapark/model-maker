# Painted lettering

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## Painted lettering (11 November 2026)

Version 0.1.0's unseen run passed "an iron anvil" with three lines of made-up words painted on its side, and they went on
into its views and its paint. The picture model paints letters, logos or a signature on some objects, and nothing
checked for them. The picture words already end "no text, no signature", but the picture model's first text encoder
(CLIP) reads only 77 tokens and cuts those words off (docs/setup.md, "Warnings you can ignore"; its second encoder, T5,
reads the whole prompt, and the letters come anyway). That prompt was looked at to understand the fault; nothing
below was measured or tuned on 0.1.0's unseen ten.

### The check

"no painted lettering" (`lettering.py`) joins the picture checks in the choice of try: a try with lettering the words
do not ask for is not chosen, and when no try of two rounds is free of it, the job ends at its pictures, failed on it.

* **The reader** is tesseract (Apache-2.0, version 5 or newer, with its English data), an optical character
  recogniser, run as its own program. It is probed once a run: its version (version 4 takes the option for Sauvola's
  threshold without using it, says "Could not set option", and reads with Otsu's alone: by the table below, 10 of the
  32 caught instead of 15) and its English data. The
  region round the cut-out is read in grey, the backdrop made white, its contrast stretched over the object's own
  greys, at twice its size, as painted and inverted (light letters cut into dark iron, dark letters on light wood),
  each with one threshold for the whole region (Otsu's) and with one set from the light round each pixel (Sauvola's:
  grey letters on a black blade beside a pale handle are lost to one threshold), in tesseract's sparse-text mode. A
  try takes about 0.45 seconds on the CPU (the median of 20; at most 1.2).
* **What counts.** Grain, stone joints, rivets, hoops and bristles are read as short scraps ("iit", "Wil", "SSS",
  "ANY") at middling confidence. A word counts when it is read with a confidence of at least 70, has at least two
  letters and three letters and digits, is nearly all letters and digits once the punctuation round it is trimmed,
  and is at least 8 pixels of 1024 tall. A picture has lettering when its counted words hold more than 4 letters and
  digits between them: one word of five, or two short ones.
* **Words that ask for lettering** (`words.lettering_asked`: "a sign that says OPEN", "a book with a title on its
  cover", "a gravestone with an inscription", "a plaque engraved with a name", "a mug printed with HELLO", "a crate
  marked FRAGILE", "a book called Dune", a rune, a label, a logo; a word in capitals, "a tombstone with RIP"; words in
  single quotes, "a sign 'inn'") are not refused: the letters are read and listed. Such a picture is also painted
  without "no text, no signature". A word that only names a thing that often bears letters ("a sign", "a plaque", "a
  coin") does not ask for them, nor does a lettering word where it means something else ("an engraved silver cup", "a
  pile of letters", "a number of coins", "a well-read book", "a title deed", "a reading lamp"). A denial ("a sign with
  no text", "neither text nor logo", "a text-free sign", "free of text") reaches only its own words: "a box with no
  lid, labelled FRAGILE" and "a no smoking label" ask. Double quotes are dropped when a prompt is read, so a word
  asked for in them is asked only when it is in capitals ('a sign "OPEN"'): 'a sign "inn"' is not read as asking.
* **Without a usable tesseract** (it is optional; not installed, version 4 or older, no English data, or a reading
  that fails or takes over a minute) the check says "not judged" and why: shown, not counted, never a pass; the run
  says so once. The job says so too: its record, its sidecar (`lettering_judged`, `tesseract_version`), its report and
  its ledger line, and `tools/trial/summary.py` never counts such a job as a pass (it is shown apart, "not read"). A
  measured run requires the reader (`--require-lettering`): without one its jobs are "not made", and a try it could
  not read is counted as failed. A refinish (`model-maker finish`) reads the chosen picture again, as the other
  picture checks are measured again (it needs no card); without a reader there, the reading made when the try was
  chosen is kept. `./setup.sh --check-only` lists tesseract, its version and whether it can be used, with how to
  install it; setup installs no system package.

### In the picture, not in the views

The six views are painted from the chosen picture, and a view cannot change the choice: by then the try is chosen.
Read the same way, a view takes about 1 second (6 seconds a job, against 0.45 a try), and on the 684 views of the
trials' 114 made objects nothing was read as lettering. On 0.1.0's anvil (looked at, not tuned on), the picture's
try with the letters is refused (18 characters read, "EANITIES", "NUISNHIETS"), its three other tries pass, and its
view from the front, which shows the same words larger and softer, is read as one scrap ("TAL"). So the picture is
where the check runs.

### Measured

Every distinct picture try of every trial reachable here (808: the trials' pictures folders, the war hammer and the
quality trials, the game props' pictures; not 0.1.0's unseen ten, nor the reproduction of its anvil), and 160 new
tries painted for this with the pipeline's own words and seeds (`tools/trial/paint_tries.py`), 8 each: on the 16 GB
card "an old iron anvil", "a wooden crate", "a brass plaque", "a wooden shipping crate", "a stone gravestone", "a gold
coin", "a glass potion bottle", "a wooden barrel", "a wooden signboard", "an old leather-bound book"; on the 12 GB card
"a wooden sign that says OPEN", "a book with a title on its cover", "a brass plaque engraved with a name", "a stone
gravestone with an inscription", "a tin box", "a bronze bell", "a stone milestone", "a wooden tea chest", "a bronze
medal", "a wooden chest of drawers".

Labelled blind, by eye (`tools/trial/lettering_sheets.py`: cropped round the cut-out, shuffled under a random code):
every trial try that looser limits refused (87: confidence 40, height 6), 100 drawn at random from the other 721, and
all 160 new tries. "letters": letters, words, numbers or a signature large enough to read as such; "unclear": a mark
too small to read (a maker's stamp a few pixels tall on a handle or a blade, a logo without letters).

| | Trials' tries, labelled (187) | New tries (160) | Every try (968) |
|---|---|---|---|
| showing none, refused | 0 of 173 | 0 of 98 | 0 (and none of the 621 not labelled) |
| a try a trial chose, refused | | | **0 of 65** |
| unclear, refused | 0 of 8 | 0 of 5 | |
| letters the words do not ask for, refused | 3 of 6 | 12 of 26 | **15 of 32** |
| letters the words ask for, passed (by construction) | | 31 of 31 (read in 13) | |

Caught: the trials' leather-bound book with "BLHOM BOORY" on its cover (both cards' tries of that seed; neither was
chosen), a hunting knife with a maker's name on its blade, and of the new tries every flat inscription: six
gravestones and two milestones (painted with made-up inscriptions though none was asked), a shipping crate with a
made-up logo and name, a signboard with a word, an old book, a coin. Missed (17), each a known limit:

* **curved lettering round a rim** (11: coins and medals, and 1 bell): tesseract reads straight lines of text;
* **letters a few pixels tall** (5: a word on a knife's blade, letters up a hammer's handle, stamps on an axe's blades,
  a small signature on a bell, the print on an open book's pages), under the height limit or too small to read.

In the random 100, 4 tries showed letters, all small: one is now caught (the knife), 3 are not. So about 3 in 100 of
the trials' tries carry small lettering the check lets through, as well as the trials' book it refuses.

What these numbers are: the limits were set on these same labelled tries, and there is no held-out set, so "0 of 173"
and "0 of 98" showing none refused say the limits fit the tries they were set on, not how often a new try showing
none will be refused. "31 of 31 asked for, passed" holds by construction: when the words ask for lettering the check
never refuses; only "read in 13" is measured.

### The limits

No try showing none, and no try a trial chose, is refused. The margin is one-sided: the most any try showing none
holds is one scrap word of 3 characters (a shield, a broom, a closed book), one character under the limit ("more than
4"); the least any caught try holds is 5, just over it, so the caught side has no margin.
Changing one limit at a time, on all 968 tries (letters not asked for refused / tries showing none
refused):

| | refused letters | refused none |
|---|---|---|
| **as kept: confidence 70, height 8, more than 4 characters** | **15 of 32** | **0** |
| confidence 60 | 15 | 4 (a broom, a shield, a mirror, a table: grain and frames read as scraps) |
| confidence 80 | 12 | 0 |
| more than 2 characters | 16 | 6 (one 3-letter scrap each) |
| height 6 or 10 | 15 | 0 |
| only Otsu's threshold (as painted, inverted) | 10 | 0 |
| Sauvola's on the inverted region left out | 13 | 0 |

Sauvola's threshold found 5 of the 15 that Otsu's missed (the knife's name, the signboard's word). Measured and left
out: the same readings at half size, for letters over 60 pixels tall: they found more only on signs whose words asked
for their letters ("OPEN", 4 characters). Measured with tesseract 5.5.0; no run on another version is recorded.

Measured again on 11 November 2026 after the review's fixes (a word counted before it is kept once, and kept once by
its text as well as its box; the words that ask for lettering widened and their denials narrowed), on the same 968
tries' readings and the same labels, nothing read again (the readings are the same; only what is done with them
changed): every verdict is as before. Letters the words do not ask for: 15 of 32 caught, 17 missed (the 17 known
misses), as before; showing none, unclear, and every try a job chose (64 found by their jobs' records): none refused,
as before. Six lettered tries count other numbers of characters (a gravestone's 39 now 36, a book's 8 now 11), each
on the same side of the limit. No prompt of the labelled tries is asked for lettering otherwise than before. The
regression case "lettering-pictures", read again with tesseract 5.5.0, gives the same tally: letters not asked for
15 refused and the 17 known misses passed, letters asked for 31 passed, none 267 and unclear 13 passed.

The labelled set is a regression case, "lettering-pictures" (`tools/trial/lettering-cases.toml`, 343 pictures; the
20 game props' pictures are left out): every picture showing none and every picture whose prompt asks for lettering
must pass, every picture showing letters its prompt does not ask for must be refused, but the 17 known misses
(`known_miss`, shown, not held), and "unclear" is shown. It needs tesseract 5 or newer with its English data: without
a usable one the case counts as broken; the version that read it is printed and kept in the regression ledger.
The whole regression set holds with it (53 of 53; "lettering-pictures" took about 4 minutes on one CPU thread), and
every other number is as before; so it did again after the review's fixes (53 of 53, its lettering read by tesseract
5.5.0).
