# Names the picture model gets wrong

A living list, added to whenever we or anyone else finds an object's name that the picture model does not know. As
far as we could find (November 2026), no such list exists anywhere else for this model.

model-maker's first step paints a picture of the prompt with **FLUX.1 [schnell]** (run in 4 bits, with model-maker's
own picture words around the prompt). For many names it paints the right thing. For some it paints something close
instead: asked for "a garden hoe", it painted a digging fork. Everything after the picture builds on it, so a wrong
picture makes a well-made wrong object.

For each name on this list we write a **description**: the same thing said in words the model does know ("a garden
hoe" could become "a flat metal blade set across the end of a long wooden handle"). A later version of model-maker
will say a name's measured description when it paints that name; until then, write the description into your own
prompt.

## How a name is measured

A name goes on the list only after it is measured, so the list stays true:

1. **8 plain tries.** The name is painted 8 times, with model-maker's own seeds, without a description.
2. **Judged blind.** Each picture is labelled by eye as right, wrong or unclear, shuffled, with no name, score or
   description shown. The question judge (Qwen 3.5 9B: "Does this picture show a garden hoe?") is recorded next to
   it, but the label by eye decides.
3. **On the list** when at most 4 of the 8 are right.
4. **8 described tries.** The description is painted 8 times (with its own seeds: model-maker's seeds come from the
   words it paints) and judged blind, shuffled together with the plain tries. The description is kept only if at least
   4 more tries are right than with the name alone; the name's count is the higher of its two blind labellings.
   A name whose description does not help stays on the list with "no description yet", so everyone knows the model
   cannot paint it.

The counts hold for the model, the seeds and the picture words of the version named in each row. A later version
measures its rows again.

## The list

The first round (15 November 2026): 353 names of things people keep at home, in a workshop, on a farm or in a
castle, 8 plain tries each. 89 of them came out right in at most 4 of 8; in 56 of those, not once. "Unclear" counts
the tries that could not be told by eye (a long axe for "a hatchet": the right kind, but nothing shows it is small),
which never count as right. Only the 107 names the question judge passed in at most 5 of 8 were labelled by eye;
the other 246, which it passed in 6 or more, were not, and a few of them may belong here too (it said yes to 1 in 9
of the wrong tries it was shown). For each of the 89 names two descriptions were tried, one written by the question judge's model (Qwen 3.5 9B)
and one by hand, 8 tries each, all 24 pictures of a name shuffled and labelled blind. Of 712 tries, 77 were right with
the name alone, 138 with the model's descriptions and 460 with the hand-written ones. The better description
gained at least 4 right tries for 57 names (12 of them the model's). The other 32 keep "no description yet"; for 20
of them no version came out right in more than 2 of 8. The hand-written descriptions and the labels are by the same
person, blind to which picture came from which. The question judge, asked about all 2,136 pictures, said yes to 161
with the names alone, 329 with the model's descriptions and 518 with the hand-written ones: more lenient than the
eye, but the same order and a gain of the same size.

| Name | Right of 8, plain | Description | Right of 8, described | Measured with | Reported by |
|---|---|---|---|---|---|
| A brace and bit | 0 | no description yet | 2 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A bread knife | 0 | "a long straight kitchen knife with a serrated blade, its cutting edge lined with many small wavy teeth, and a wooden handle" | 4 | 15 Nov 2026, before 0.2.2 | |
| A buckler | 0 | "a small round metal shield, no larger than a dinner plate, with a big domed centre and a handle behind it" | 8 | 15 Nov 2026, before 0.2.2 | |
| A chamber pot | 0 | "a small round lidless china pot with a wide rolled rim and one handle on its side, about the size of a large mixing bowl" | 4 | 15 Nov 2026, before 0.2.2 | |
| A chess knight piece | 0 | "a single chess piece: a carved horse's head and neck, mane and ears, rising from a round turned base" | 8 | 15 Nov 2026, before 0.2.2 | |
| A club | 0 | "a primitive wooden club, short and heavy, thick and knobby at its striking end, narrowing to a grip at the other" | 4 | 15 Nov 2026, before 0.2.2 | |
| A cow bell | 0 | no description yet | 2 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A crowbar | 0 | no description yet | 3 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A dirk | 0 | no description yet | 1 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A doctor's bag | 0 | no description yet | 0 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A drawknife | 0 | no description yet | 0 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A fireplace bellows | 0 | no description yet | 0 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A fishing net float | 0 | "a round glass ball float about the size of a melon, wrapped in a loose net of knotted rope" | 8 | 15 Nov 2026, before 0.2.2 | |
| A flail | 0 | no description yet | 0 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A flail weapon | 0 | no description yet | 0 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A font | 0 | "a church baptismal font: a large carved stone basin on a thick stone pillar standing on the floor" | 8 | 15 Nov 2026, before 0.2.2 | |
| A froe | 0 | no description yet | 2 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A gauntlet | 0 | "an armoured glove: a steel glove with articulated plates over each finger and a flared steel cuff covering the wrist" | 8 | 15 Nov 2026, before 0.2.2 | |
| A glaive | 0 | no description yet | 2 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A grappling hook | 0 | "a hook with four curved sharp prongs spreading out from one central shaft like an anchor, with a rope tied to its ring" | 5 | 15 Nov 2026, before 0.2.2 | |
| A great helm | 0 | "a medieval knight's helmet: a closed flat-topped steel cylinder covering the whole head, with narrow eye slits and small breathing holes" | 8 | 15 Nov 2026, before 0.2.2 | |
| A hobby horse | 0 | no description yet | 2 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A hurdy-gurdy | 0 | no description yet | 0 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A lobster pot | 0 | "a fishing trap: a half-round cage of wooden slats and netting on a flat wooden base, with a netted funnel opening on each end" | 6 | 15 Nov 2026, before 0.2.2 | |
| A machete | 0 | no description yet | 3 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A morning star | 0 | "a weapon: a short wooden handle topped by a heavy round metal ball covered in long sharp spikes" | 8 | 15 Nov 2026, before 0.2.2 | |
| A prayer bench | 0 | no description yet | 2 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A sabre | 0 | no description yet | 1 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A scroll case | 0 | "a long closed cylindrical tube of leather with a fitted cap on one end and a carrying strap, used to hold rolled scrolls" | 7 | 15 Nov 2026, before 0.2.2 | |
| A sleigh bell | 0 | "a small hollow metal ball with a narrow slit across its lower half and a loop on top, a jingle bell, several on a leather strap" | 6 | 15 Nov 2026, before 0.2.2 | |
| A spokeshave | 0 | no description yet | 0 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A stretcher | 0 | "a medical stretcher for carrying a person: two long parallel poles with a canvas sheet stretched between them" | 8 | 15 Nov 2026, before 0.2.2 | |
| A sword in a scabbard | 0 | no description yet | 0 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A torch on a wall bracket | 0 | "a burning wooden torch with an open flame at its top, held upright in an iron ring bracket fixed to a wall, no glass and no lamp" | 7 | 15 Nov 2026, before 0.2.2 | |
| A wooden puzzle box | 0 | "a small closed block-like box covered all over in inlaid geometric patterns, its sides made of sliding panels, with no visible lid, hinge or lock" | 8 | 15 Nov 2026, before 0.2.2 | |
| A wooden recorder | 0 | "a straight cylindrical tube with a flared mouthpiece at one end and a series of finger holes along its side, capped by a foot joint at the opposite end" | 8 | 15 Nov 2026, before 0.2.2 | |
| A carpenter's square | 0 (1 unclear) | no description yet | 3 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A dice cup | 0 (1 unclear) | "a tall straight-sided leather cup, open at the top, slightly wider at the mouth, with a few dice beside it" | 8 | 15 Nov 2026, before 0.2.2 | |
| A riding crop | 0 (1 unclear) | no description yet | 3 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A set of brass weights | 0 (1 unclear) | "several solid brass cylinders of graded sizes, each with a small knob on top, standing in a row from large to small in a wooden block" | 8 | 15 Nov 2026, before 0.2.2 | |
| A drawbridge | 0 (2 unclear) | "a castle drawbridge: a heavy wooden plank bridge hinged at one end, half raised by two chains running up to a stone gatehouse" | 4 | 15 Nov 2026, before 0.2.2 | |
| A harpoon | 0 (2 unclear) | "a long throwing spear with a barbed iron head, sharp backward-pointing barbs behind its point, and a rope tied near its head" | 6 | 15 Nov 2026, before 0.2.2 | |
| A portcullis | 0 (2 unclear) | "a heavy castle gate grille of crossed wooden and iron bars, with sharp spikes along its bottom edge, made to slide up and down" | 6 | 15 Nov 2026, before 0.2.2 | |
| A toasting fork | 0 (2 unclear) | "a fork for toasting bread over a fire: a very long thin metal handle with a hanging loop at its end, and two or three long prongs" | 8 | 15 Nov 2026, before 0.2.2 | |
| A water clock | 0 (2 unclear) | "a vessel that tells time by dripping water: an upper bowl dripping through a small spout into a tall marked cylinder below, on a stand" | 8 | 15 Nov 2026, before 0.2.2 | |
| A wooden shoe last | 0 (2 unclear) | "a tapered wooden form: a long narrow shaft widening into a rounded foot section with a distinct heel and a flat sole base" | 6 | 15 Nov 2026, before 0.2.2 | |
| A battering ram | 0 (3 unclear) | "a siege engine: a heavy log with an iron cap on its front end, hung by chains from a wooden roofed frame on wheels" | 8 | 15 Nov 2026, before 0.2.2 | |
| A catapult | 0 (3 unclear) | "a medieval siege engine: a wooden frame on wheels with a long throwing arm ending in a bucket, pulled back by ropes" | 8 | 15 Nov 2026, before 0.2.2 | |
| A dibber | 0 (3 unclear) | "a garden tool: a short thick wooden stick tapering to a blunt point, with a D-shaped handle on its top end" | 8 | 15 Nov 2026, before 0.2.2 | |
| A candle clock | 0 (4 unclear) | "a tall candle marked with evenly spaced numbered rings down its side to measure the hours, in a candle holder" | 8 | 15 Nov 2026, before 0.2.2 | |
| A shield with a boss | 0 (5 unclear) | "a round wooden shield with a domed metal boss in its exact centre" | 8 | 15 Nov 2026, before 0.2.2 | |
| A water pump | 0 (6 unclear) | "an old cast-iron hand pump for water: an upright column with a curved spout on its front and a long lever handle on top" | 7 | 15 Nov 2026, before 0.2.2 | |
| A hatchet | 0 (7 unclear) | no description yet | 1 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A music box | 0 (7 unclear) | "a small open wooden box with a visible brass music cylinder with tiny pins and a metal comb inside, and a winding key on its side" | 8 | 15 Nov 2026, before 0.2.2 | |
| A tinderbox | 0 (7 unclear) | "a small round tin box with a hinged lid, open, holding a flint, a C-shaped fire steel and some tinder" | 8 | 15 Nov 2026, before 0.2.2 | |
| A throwing axe | 0 (8 unclear) | no description yet | 0 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A corner shelf | 1 | "a set of quarter-round wooden shelves made to fit into the corner of a room, two flat back sides meeting at a right angle" | 5 | 15 Nov 2026, before 0.2.2 | |
| A crutch | 1 | "a walking crutch: a padded bar at the top to go under the arm, two long side rails meeting at a single rubber-tipped foot, and a hand grip between them" | 5 | 15 Nov 2026, before 0.2.2 | |
| A leather waterskin | 1 | "a flexible bladder: an elongated rounded body with a narrow neck at the top, a small loop handle attached to the neck, and a spout extending from the side near the opening" | 8 | 15 Nov 2026, before 0.2.2 | |
| A scimitar | 1 | no description yet | 2 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A bedroll | 1 (1 unclear) | "a blanket and a sleeping pad rolled up tightly into a long cylinder and tied with two leather straps" | 8 | 15 Nov 2026, before 0.2.2 | |
| A sundial | 1 (1 unclear) | "a flat circular base with a central vertical rod casting a shadow onto engraved hour lines radiating from the center" | 8 | 15 Nov 2026, before 0.2.2 | |
| A chisel | 1 (2 unclear) | "a woodworking tool: a wooden handle and a flat steel blade with a straight square cutting edge across its end, like a narrow flat bar" | 7 | 15 Nov 2026, before 0.2.2 | |
| A spyglass | 1 (2 unclear) | "a hand-held telescope: a long brass tube made of three sections that slide into one another, wider at one end, with a small eyepiece at the other" | 7 | 15 Nov 2026, before 0.2.2 | |
| A board game box | 1 (3 unclear) | "a flat rectangular cardboard game box with a game board and pieces shown printed on its lid" | 8 | 15 Nov 2026, before 0.2.2 | |
| A milestone | 1 (3 unclear) | "a short standing stone by a road, with a rounded top, about knee high, carved with a number and an arrow" | 7 | 15 Nov 2026, before 0.2.2 | |
| A cat basket | 1 (7 unclear) | "a pet carrier: a woven wicker basket with a domed lid, a carrying handle on top and a small barred door at one end" | 7 | 15 Nov 2026, before 0.2.2 | |
| A katana | 1 (7 unclear) | "a long curved single-edged blade attached to a straight handle, featuring a circular handguard and a simple crossguard" | 8 | 15 Nov 2026, before 0.2.2 | |
| A magic wand | 1 (7 unclear) | "a short slender wooden wand, about as long as a hand and forearm, with a carved handle" | 7 | 15 Nov 2026, before 0.2.2 | |
| A quilt folded | 2 | "a patchwork quilt folded into a neat thick square, its layers of stitched fabric stacked on one another" | 7 | 15 Nov 2026, before 0.2.2 | |
| A cutlass | 2 (1 unclear) | no description yet | 0 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A round shield | 2 (3 unclear) | "a large circular wooden shield made of planks, with a metal rim and a domed metal boss at its centre" | 8 | 15 Nov 2026, before 0.2.2 | |
| A strongbox | 2 (4 unclear) | "a small heavy iron chest with riveted metal bands, a big keyhole and lock plate on its front, and handles on its sides" | 8 | 15 Nov 2026, before 0.2.2 | |
| A sugar bowl | 2 (5 unclear) | "a small round lidded china bowl with two little handles on its sides and a knob on its lid" | 8 | 15 Nov 2026, before 0.2.2 | |
| A serving platter | 2 (6 unclear) | no description yet | 5 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A book stand | 3 | no description yet | 2 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A quiver of arrows | 3 | no description yet | 6 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A shop sign | 3 (2 unclear) | "a wooden board sign hanging from two chains below an iron bracket sticking out from a wall" | 8 | 15 Nov 2026, before 0.2.2 | |
| An umbrella stand | 3 (3 unclear) | "a tall open cylinder on the floor, like a tall bucket, with several closed umbrellas and walking sticks standing in it" | 8 | 15 Nov 2026, before 0.2.2 | |
| A cannonball | 3 (5 unclear) | no description yet | 4 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A laundry tub | 3 (5 unclear) | no description yet | 8 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A log basket | 3 (5 unclear) | "a sturdy woven basket with a carrying handle, filled with chopped firewood logs" | 8 | 15 Nov 2026, before 0.2.2 | |
| A pair of boots | 3 (5 unclear) | "two boots standing side by side, a left one and a right one" | 8 | 15 Nov 2026, before 0.2.2 | |
| A salt box | 3 (5 unclear) | "a small wooden box made to hang on a kitchen wall, with a sloping hinged lid and a hole at the top of its back board" | 7 | 15 Nov 2026, before 0.2.2 | |
| An astrolabe | 3 (5 unclear) | no description yet | 3 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A wooden barrel cart | 4 (1 unclear) | "a two-wheeled wooden cart with handles, a wooden barrel lying on its side on its bed" | 8 | 15 Nov 2026, before 0.2.2 | |
| A hitching post | 4 (2 unclear) | no description yet | 7 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A rolling board | 4 (4 unclear) | no description yet | 2 (best of two tried) | 15 Nov 2026, before 0.2.2 | |
| A wizard's staff | 4 (4 unclear) | "a tall gnarled wooden walking staff, taller than a person, its top twisted into a knot holding a glowing crystal" | 8 | 15 Nov 2026, before 0.2.2 | |

## Adding a name

Anyone can help:

- **Found a name it gets wrong?** Open an issue with the form "A name the picture model gets wrong". Give the
  prompt, and if you can, the pictures model-maker kept (`pictures/` in the job's folder). We measure it as above
  and add its row, with your name or handle in "Reported by" if you want it there.
- **Have a description that works better?** Open the same form with the description, or a pull request that adds
  it to the row under "Description" with "(not measured)". We measure it before it replaces the description in use.
- A row is changed only by a new measurement, never by eye alone or by one good picture.
