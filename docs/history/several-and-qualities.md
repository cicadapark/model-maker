# Several things, and qualities the words name

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## Several things, and qualities the words name (11 November 2026)

Version 0.1.0's unseen run made two objects the prompt did not describe:

* **"a pair of leather boots" made one boot.** Three of its four picture tries drew two boots, touching, so each
  passed "one piece" as one lump of two; the fourth drew one boot, scored best on the other picture checks and was
  chosen. Either way the words were not met: 0.1.0 cannot make two boots as two objects.
* **"a short curved dagger" came out straight.** None of its four tries was curved, and nothing checked a quality the
  prompt names.

Those ten prompts were looked at to understand the two faults; nothing below was measured or tuned on them.

### Several things: answered before anything is painted

model-maker makes one object in one piece. A prompt that asks for more than one thing is now answered at once, before
any card time, with the one thing to ask for instead (`words.several_of`):

    model-maker makes one object at a time, and these words ask for more than one. Try one: "a leather boot".

It asks for several when its head phrase (the words before "of", "with", "in", "for", "by", ...):

| asks for several | example | suggested |
|---|---|---|
| groups things | "a set of three knives", "a collection of bottles" | "a knife", "a bottle" |
| counts them | "three wooden chairs", "5 lanterns", "a few goblets", "three iron swords with gold hilts" | "a wooden chair", "a lantern", "a goblet", "an iron sword with gold hilts" |
| names a known thing in the plural | "boots", "leather boots, brown", "knives", "bookshelves" | "a boot", "a leather boot, brown", "a knife", "a bookshelf" |
| joins two known things | "a sword and a shield", "a table and chairs" | "a sword", "a table" |

### Pairs

Two of one known thing ("a pair of boots", "a couple of mugs", "two iron candlesticks", "2 swords") is made once, from
the words for one ("a boot"), checked and painted as one object, and placed twice in the .glb: the same mesh a second
time, beside the first with a gap of 15 % of its width (at least 2 cm). When the two are a left and a right (boots,
shoes, sandals, slippers, gloves, mittens, gauntlets, greaves, bracers, spurs) the second is the first's mirror
image; otherwise an exact copy. A mirror image is an exact match, where a second one made would differ in every
detail. Lettering asked for on them is copied, not mirrored (a mirror would turn the letters round). The .glb's
`extras` say `"pair": "mirror"` or `"copy"`; the report shows the one made.

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
  each: with 0.1.0's words 1 of 48 curved (one bronze dagger), with these words and view 7 of 48 (the hunting
  knife 4 of 8, the bronze dagger, the iron sword and the short sword 1 each, the silver dagger and the sabre none).
  Tried and not kept: a description of the curve alone, the view unchanged ("its whole length bending in one smooth
  arc": the bronze dagger 3 of 8, the iron sword none). A small sample, and still few curved tries: a job whose tries
  never show the curve now ends at its pictures, failed on "shows the thing asked for", where 0.1.0 made a
  straight dagger.

### Measured

New picture tries, painted with the pipeline's own picture words and seeds (`tools/trial/paint_tries.py`): 384 on both
cards, 8 for each prompt and wording. The prompts name a quality ("a curved hunting knife", "a round mirror", "a tall
clay vase") or its opposite or nothing ("a straight bronze dagger", "a hunting knife", "a square mirror", "a low
wooden stool", "a wooden table"). None is one of 0.1.0's unseen ten. Every try was labelled by eye before its
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
round mirror (0.35, 0.61, the likeliest other "a coin"), as 0.1.0's check did. The pair is not to blame there.

The 48 tries painted after the wording was chosen ("a curved hunting knife", "a curved silver dagger", "a curved short
sword" and "a curved steel sabre" with the words and view kept, the dagger and the short sword also with 0.1.0's
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
