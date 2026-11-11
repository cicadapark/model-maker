"""What a prompt asks for, read from its words: what kind of 3-D object (held in the hand or not), which checks apply
to it, its usual size, and the words the picture model is given.

Nothing here guesses beyond the words. A word list decides each choice, so a user can see why a check ran or did
not (the report says which word decided it), and `model-maker make --stands no` or `--symmetric no` overrides it.
"""

import re
from dataclasses import dataclass, field

# A prompt is at most this many characters; the rest is not read.
MOST_PROMPT_CHARACTERS = 300

# The picture model is told about the light before the subject: its text encoder weighs the first words most, and
# an object painted in light from one side keeps that shadow in its colours, and a shadow under it becomes a dark
# slab in the shape.
LEAD = ("lit evenly from every side like a catalogue scan, no shadow anywhere, floating in empty space with no floor "
        "and no ground shadow, on a seamless pale grey backdrop: one single object alone, the whole of it in view:")
# The first trial (2026-11-09): with LEAD, 37 of 76 tries passed the picture checks, most failing for a floor or a
# shadow; "floating" drew objects hanging above a floor with their shadow on it (the chest, the shield, the book).
# Other leads, measured against it on the card before one becomes the default (tools/trial/picture_words.sh).
LEADS = {
    "floating": LEAD,
    "isolated": ("one single object isolated on a plain seamless light grey background, flat even light from every "
                 "side, no floor, no horizon, no ground, no shadow at all, the whole object in view:"),
    "cutout": ("one single object shown alone as a cut-out on a flat plain light grey background, evenly lit from "
               "every side, no shadow, no floor and no horizon, the whole object in view:"),
}
DEFAULT_LEAD = "floating"
# Asked for "slightly from above", the first trial's chair came from the front at eye level: its far legs ended higher
# in the picture, and the shape model made them 3 cm short (both shape seeds), so it settled 3.6 degrees back. Seen
# from well above, every foot shows where it stands.
OBJECT_VIEW = "centred, seen from the front and a little to one side, looking down on it from about thirty degrees above"
# A long weapon is painted upright, its length up the picture, its flat side to the viewer: the shape model then
# sees its whole length and the blade's width, and the checks find its length along the up axis.
WEAPON_VIEW = "standing upright, its point or head at the top, seen straight from the side, the whole length in view"
# A head on a haft is painted without "point": asked for "its point or head at the top", the picture model gave every
# war hammer of the trials a spike on top and an axe-like blade (docs/objects.md, "War hammers"). A blade and a spear
# keep WEAPON_VIEW: their tip is a point.
HAFTED_VIEW = "standing upright, its head at the top, seen straight from the side, the whole length in view"
# A bow is painted facing the viewer, its curve in view: painted edge-on (the first trial), its bend was out of sight
# and the shape model gave it an S-shaped wiggle in that direction (a 39 degree kink, limbs that did not mirror).
BOW_VIEW = "standing upright, facing the viewer so its whole curve shows from tip to tip, the whole length in view"
# A long-handled tool stands as it is used to: a broom's bristles down.
TOOL_VIEW = "standing upright on its working end, its handle up, seen straight from the side, the whole length in view"
SHIELD_VIEW ="seen straight from the front, its face to the viewer, upright"
STYLE = "clean silhouette, true materials, crisp detail, no text, no signature, no hands, no people"

# Each kind of weapon, its words (longest first wins), and its usual length in metres (a person's weapons: a sword
# about a metre, a longbow about a person's height, a spear above it).
WEAPON_KINDS = {
    # A straight blade: its line straight, mirror-symmetric across the blade.
    "blade": {
        "greatsword": 1.4, "longsword": 1.1, "broadsword": 1.0, "sword": 0.95, "rapier": 1.05, "dagger": 0.4,
        "knife": 0.3, "dirk": 0.45, "shortsword": 0.7, "short sword": 0.7,
    },
    # A curved blade: no straight line asked, its curve without a kink.
    "curved blade": {"sabre": 0.95, "saber": 0.95, "scimitar": 0.9, "falchion": 0.85, "cutlass": 0.8, "sickle": 0.5},
    # A head on a haft: the haft straight.
    "hafted": {
        "battle axe": 0.9, "battleaxe": 0.9, "axe": 0.8, "hatchet": 0.4, "war hammer": 0.8, "warhammer": 0.8,
        "hammer": 0.6, "mace": 0.7, "club": 0.8, "flail": 0.8, "morning star": 0.8, "halberd": 2.0, "glaive": 2.0,
        "poleaxe": 1.8, "pick": 0.9, "staff": 1.8, "quarterstaff": 1.9, "maul": 1.0, "sledgehammer": 0.9,
        "sledge hammer": 0.9, "mallet": 0.35,
    },
    "spear": {"spear": 2.0, "pike": 3.0, "lance": 3.0, "javelin": 1.2, "trident": 1.8},
    # A bow is curved by design: its curve without a kink, mirror-symmetric about its grip.
    "bow": {"longbow": 1.8, "shortbow": 1.1, "recurve bow": 1.3, "bow": 1.4},
    "shield": {"shield": 0.7, "buckler": 0.4},
}
# What a kind's head looks like, said to the picture model after the user's words, as STATE_PICTURE_WORDS says a lid
# is shut: asked for "a war hammer" alone, the picture model painted an axe-like blade with a square poll or a spike in
# every try of the trials (the concept is mixed: many war hammers pictured have a blade or beak behind the face). The
# words name no material, so "a wooden mallet" stays wooden. FLUX.1 [schnell] runs with no guidance, so nothing can be
# asked away: the head is described as it should look. Measured in docs/objects.md, "War hammers".
HEAD_PICTURE_WORDS = {
    "hammer": "its head a heavy blunt block with a flat square striking face",
}
HEAD_KINDS = {"war hammer": "hammer", "warhammer": "hammer", "hammer": "hammer", "maul": "hammer",
              "mallet": "hammer", "sledgehammer": "hammer", "sledge hammer": "hammer"}
# Kinds of thing not made yet, and why, said to whoever asks for one. Bows (the second trial, 2026-11-09): painted
# edge-on, the shape model made the bow wave; asked to face the viewer, the picture model still painted it in profile
# with its string, and the shape model filled the space between bow and string into a flat plate.
NOT_YET = {
    "bow": ("Bows are not made yet: the picture model does not paint a bow from the side that shows its curve, and the "
            "shape model fills the space between a bow and its string. They come in a later release."),
}
# Words that make "bow" or "staff" not a weapon (a bow tie, a ship's bow).
NOT_WEAPON = ("bow tie", "ship", "violin", "cello", "rainbow", "elbow", "bowl")

# Things that stand on their own base, with their usual largest size in metres.
STANDING = {
    "chair": 0.95, "armchair": 1.0, "throne": 1.5, "stool": 0.5, "bench": 0.5, "table": 0.75, "desk": 0.8,
    "barrel": 0.9, "keg": 0.5, "cask": 0.7, "crate": 0.6, "chest": 0.8, "trunk": 0.8, "box": 0.4, "cabinet": 1.6,
    "wardrobe": 1.9, "bookshelf": 1.8, "shelf": 1.5, "bed": 2.0, "jug": 0.3, "jar": 0.25, "pot": 0.3, "vase": 0.4,
    "urn": 0.5, "bottle": 0.3, "flask": 0.2, "mug": 0.12, "cup": 0.1, "goblet": 0.2, "tankard": 0.18, "bowl": 0.2,
    "bucket": 0.35, "basket": 0.4, "lantern": 0.35, "lamp": 0.4, "candle": 0.25, "candlestick": 0.3,
    "candelabra": 0.5, "book": 0.25, "anvil": 0.5, "cauldron": 0.6, "statue": 1.2, "bust": 0.5, "clock": 0.4,
    "globe": 0.5, "pedestal": 1.0, "chess": 0.1, "coin": 0.03, "plate": 0.25, "tray": 0.45, "teapot": 0.2,
    "kettle": 0.25, "stump": 0.5, "rock": 0.6,
    "boulder": 1.2, "mushroom": 0.3, "well": 1.6, "fountain": 1.5, "signpost": 2.2, "mailbox": 1.2,
}
# Things among those that stand that lie on their broad face: set down, they rest on whichever face they fall on, and
# the shape model, which makes a thing as its picture shows it, may make one standing on an edge (the fix-check's book,
# painted from well above, came out tilted toward the viewer and settled 54.8 degrees). The finish lays such a thing on
# its likeliest resting face (lying.py); a thing that stands on a base of its own (a chair) is never laid down.
LYING = {"book", "coin", "plate", "tray"}
# Words that stand a thing that lies on its edge or end on purpose ("a book standing upright"): it is left as made.
UPRIGHT_WORDS = ("standing", "upright", "on its edge", "on edge", "on its end", "on end", "on its spine")
# Words that make "plate" not a dish ("plate armour").
NOT_PLATE = ("armour", "armor", "mail")
# Things that do not stand on their own (they lean, hang or lie loose), with their usual largest size.
NOT_STANDING = {
    "broom": 1.3, "shovel": 1.2, "spade": 1.1, "rake": 1.5, "pitchfork": 1.6, "hoe": 1.4, "scythe": 1.6,
    "saw": 0.7, "wrench": 0.3, "key": 0.12, "rope": 0.6, "chain": 0.6, "flag": 1.5, "banner": 1.5, "rug": 2.0,
    "carpet": 2.0, "painting": 0.8, "mirror": 0.8, "wreath": 0.5, "necklace": 0.4, "ring": 0.03, "crown": 0.2,
    "hat": 0.35, "helmet": 0.3, "boot": 0.3, "glove": 0.25, "bag": 0.4, "sack": 0.6, "scroll": 0.3, "map": 0.5,
    "wand": 0.35, "torch": 0.5, "pipe": 0.25, "feather": 0.3, "bone": 0.4, "skull": 0.2, "horn": 0.4,
}
# Long-handled tools have a shaft and a grip as a hafted weapon does: those checks apply.
HANDLED_TOOLS = {"broom", "shovel", "spade", "rake", "pitchfork", "hoe", "scythe"}
# A thing made by hand to a plan is mirror-symmetric unless its words say it is broken or lying in disorder.
NOT_SYMMETRIC_WORDS = (
    "broken", "tipped", "fallen", "toppled", "leaning", "melted", "crumpled", "spilled", "pile", "heap", "tangled",
    "twisted", "half-burnt", "burnt", "cracked", "smashed", "torn", "open", "opened", "asymmetric", "lopsided",
)
# Things whose shape is mostly not mirror-symmetric even when whole.
NOT_SYMMETRIC_THINGS = {
    "broom", "rock", "boulder", "stump", "rope", "chain", "flag", "banner", "feather", "bone", "horn", "scythe",
    "saw", "wrench", "key", "pipe", "boot", "glove", "bag", "sack", "scroll", "map", "torch", "teapot", "kettle",
    "pick", "sickle", "hatchet", "axe", "battle axe", "battleaxe", "war hammer", "warhammer", "hammer", "flail",
    "halberd", "glaive", "poleaxe", "club", "statue", "bust", "mushroom", "signpost",
}
# Two-sided heads make a hafted weapon mirror-symmetric across its haft.
TWO_SIDED_WORDS = ("double-bitted", "double bitted", "double-headed", "double headed", "double-bladed",
                   "double bladed", "two-headed", "labrys")
# Things with a lid, a cover or doors, shut or open, and which part opens. On the third trial (2026-11-10) the picture
# model painted "a wooden treasure chest" once of four with its lid up (that try was chosen: the views could not paint
# its inside) and "a leather-bound book, closed" open in two of four (one chosen; it would not stand). So a thing is
# shut unless its words open it, the picture model is told so, and the picture judge checks it (state_judge).
OPENING_PARTS = {"chest": "lid", "trunk": "lid", "box": "lid", "crate": "lid", "book": "cover", "cabinet": "doors",
                 "wardrobe": "doors"}
OPEN_WORDS = ("open", "opened", "ajar")
SHUT_WORDS = ("closed", "shut", "locked")
STATE_PICTURE_WORDS = {
    "lid": {"closed": "its lid shut", "open": "its lid open"},
    "cover": {"closed": "lying shut, only its cover showing", "open": "lying open, its pages showing"},
    "doors": {"closed": "its doors shut", "open": "its doors open"},
}
# What the picture judge compares: pairs of descriptions, the same evidence seen shut and seen open ({noun} is the
# thing's word). Each names what the state looks like, not only its name (classification by description: Menon and
# Vondrick 2023, CuPL): the bare wordings ("a closed book", "a photo of a closed book") scored the noun's usual look, so
# a book painted open with leather-coloured halves scored 0.88 shut (a trial on the 12 GB card). Each keeps
# the noun, which carries most of what such descriptions are worth. See docs/objects.md.
STATE_JUDGE_TEXTS = {
    "lid": [("a closed {noun}, its lid down on top of it", "an open {noun}, its lid raised up behind it"),
            ("a closed {noun}, its inside hidden", "an open {noun}, its inside showing")],
    "cover": [("a closed {noun}, its front cover one flat panel with no fold across it",
               "an open {noun}, its two halves side by side with a fold down the middle"),
              ("a closed {noun}, the edges of its pages in one even block",
               "an open {noun}, the edges of its pages split in two at the middle"),
              ("a closed {noun}, its covers pressed together", "an open {noun}, its covers spread apart"),
              ("a closed {noun}, no pages showing", "an open {noun}, its pages showing"),
              ("a closed {noun}, its spine straight", "an open {noun}, bent open along its spine")],
    "doors": [("a closed {noun}, its doors shut flat across its front", "an open {noun}, its doors swung out"),
              ("a closed {noun}, its inside hidden", "an open {noun}, its inside showing")],
}
# Several things in one prompt. Release 1 makes one object in one piece, and the picture checks hold a picture to one
# piece: in release 1's unseen run "a pair of leather boots" was painted as two boots in three tries of four, and the
# fourth, one boot, was chosen and made. So a prompt that asks for more than one thing is answered before anything is
# painted, with the one thing to ask for instead. It asks for several when its head phrase (the words before "of",
# "with", "for", ...) names a known thing in the plural ("leather boots"), counts ("two chairs", "3 swords"), groups
# ("a pair of boots", "a set of knives") or joins two known things ("a sword and a shield"). A count joined to the
# next word or followed by a participle is a description ("a two-handed sword", "a three legged stool"), and the
# plural of a thing after "of" or "with" is what it holds or is made of ("a bag of coins", "a pile of rocks", "a
# chest of drawers"). A pile, heap, stack or bundle is one object (it is not mirror-symmetric: NOT_SYMMETRIC_WORDS).
SEVERAL_THINGS = "model-maker makes one object at a time, and these words ask for more than one. Try one: \"{one}\"."
SEVERAL_THINGS_NO_ONE = ("model-maker makes one object at a time, and these words ask for more than one. Ask for one "
                         "of them.")
COUNT_WORDS = ("two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "dozen",
               "several", "many", "multiple", "numerous", "few", "couple")
GROUP_WORDS = ("pair", "pairs", "set", "sets", "collection", "group", "assortment", "selection", "row", "couple")
# Things one object although named as a pair or a set ("a pair of scissors", "a set of drawers").
ONE_THING_IN_PAIRS = (
    "scissors", "shears", "tongs", "pliers", "pincers", "tweezers", "snips", "clippers", "secateurs", "nutcrackers",
    "glasses", "spectacles", "goggles", "binoculars", "trousers", "pants", "breeches", "shorts", "jeans", "tights",
    "bellows", "compasses", "calipers", "callipers", "dividers", "handcuffs", "manacles", "shackles", "drawers",
    "shelves", "stairs", "steps", "scales", "bagpipes", "gallows", "antlers",
)
# Plurals after a count that measure, not count: "a 100 years old chest".
MEASURES = ("years", "days", "weeks", "months", "hours", "centuries", "decades", "inches", "feet", "metres", "meters",
            "centimetres", "centimeters", "millimetres", "millimeters", "yards", "miles", "pounds", "kilos",
            "kilograms", "grams", "gallons", "litres", "liters", "cms", "mms", "lbs", "kgs")
# A plural not made by adding s: a known thing's own.
IRREGULAR_PLURALS = {"staff": ("staves",), "quarterstaff": ("quarterstaves",)}
# Qualities a prompt names that decide a thing's shape or look ("a curved dagger", "a round box"), each judged in the
# picture as a state is (state_judge): the picture must show the quality asked for at least three times as likely as
# its opposite. In release 1's unseen run "a short curved dagger" was painted straight in four tries of four and one was
# made: nothing checked the word "curved". A quality counts when its word is in the head phrase, not denied ("not
# curved"), not joined to the next word ("a round-bottomed flask") and not followed by a participle ("a tall backed
# chair"): there it describes a part. Each quality has contrast pairs: the same evidence both ways, the thing's name
# kept in each (classification by description, as STATE_JUDGE_TEXTS). Only the qualities whose pairs held on labelled
# pictures are judged; "straight", "square", "tall", "squat", "broken" and "rusty" were measured and left out
# (docs/objects.md, "Qualities the words name").
QUALITY_WORDS = {"curved": "curved", "round": "round", "circular": "round"}
QUALITY_OPPOSITES = {"curved": "straight", "round": "square"}
# Words that contradict a quality: with both said ("a round square table"), the quality is not judged.
QUALITY_CONTRADICTED_BY = {"curved": ("straight",), "round": ("square", "rectangular", "oblong")}
QUALITY_JUDGE_TEXTS = {
    "curved": [("a curved {noun}, its whole length bending in one smooth arc",
                "a straight {noun}, its whole length one straight line"),
               ("a {noun} with a curved outline, bent like a crescent",
                "a {noun} with a straight outline, its edges parallel lines")],
    "round": [("a round {noun}, its outline a circle", "a square {noun}, its outline a square with four corners"),
              ("a round {noun} with no corners, curved all the way round",
               "a square {noun} with straight sides meeting at corners")],
}
# A quality the thing's own word already has is not judged again: a sabre, a scimitar or a sickle is curved by its
# kind, and the thing asked for is judged with its kind's description ("a curved blade with a hilt"). Painted as a
# crook, the two sickles of the trial labelled curved scored 0.60 and 0.73 on the curved pair.
QUALITY_OF_THE_KIND = {"curved": {"sabre", "saber", "scimitar", "falchion", "cutlass", "sickle", "scythe"}}
# Things whose quality the picture judge cannot see, measured: of 7 hunting knives labelled curved, the curved pair
# refused 5 (0.07 to 0.56; a knife's curve is in its short blade, its handle straight), and a second wording, tried on
# two of them, refused them too. A curved knife is still a curved blade, painted with its curve in view; its picture is
# not held to the curve.
QUALITY_NOT_JUDGED = {"curved": {"knife"}}
# A curved blade (a sabre, or a blade the words call curved) is painted facing the viewer with its curve in view, as a
# bow is: with the blade view ("standing upright ... seen straight from the side"), 1 of 24 tries of a curved dagger,
# a curved hunting knife and a curved iron sword was curved, and 0 of 8 of "a curved steel sabre"; with these words
# and view, 5 of 24 (docs/objects.md, "Qualities the words name").
CURVED_BLADE_VIEW = ("standing upright, its point at the top, seen from the side so its whole curve shows, the whole "
                     "length in view")
CURVED_BLADE_WORDS = "its blade curving in one smooth arc like a sabre's"
# The judge's kind for a blade the words call curved (SUBJECT_KINDS' row with this word).
CURVED_BLADE_KIND = "sabre"
# The thing's word in the judge's descriptions when no known word names it.
UNKNOWN_THING = "object"
# People, animals and creatures: characters, which come in the second release. A prompt whose head noun is one of
# these is answered so and not made; "a statue of a knight",
# "a wolf skull" and "a king's crown" are objects.
CHARACTERS_LATER = ("People, animals and creatures come in the second release; this one makes 3-D objects. "
                    "For a statue or a carving of one, say so: \"a stone statue of a dragon\".")
CHARACTER_WORDS = (
    "person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "children", "baby", "lady", "human",
    "humanoid", "character", "hero", "heroine", "adventurer", "king", "queen", "prince", "princess", "knight",
    "warrior", "soldier", "guard", "archer", "wizard", "witch", "mage", "sorcerer", "priest", "monk", "nun", "farmer",
    "merchant", "blacksmith", "pirate", "sailor", "hunter", "thief", "bandit", "villager", "peasant", "elf", "elves",
    "dwarf", "dwarves", "gnome", "halfling", "orc", "goblin", "troll", "ogre", "giant", "dragon", "wyvern", "griffin",
    "unicorn", "demon", "angel", "ghost", "zombie", "skeleton", "vampire", "werewolf", "mermaid", "fairy", "golem",
    "robot", "monster", "creature", "beast", "animal", "wolf", "wolves", "dog", "puppy", "cat", "kitten", "horse",
    "pony", "cow", "bull", "pig", "sheep", "goat", "deer", "stag", "bear", "fox", "rabbit", "lion", "tiger",
    "elephant", "monkey", "ape", "bird", "eagle", "owl", "raven", "crow", "chicken", "rooster", "fish", "shark",
    "whale", "dolphin", "snake", "serpent", "lizard", "frog", "toad", "turtle", "spider", "scorpion", "insect",
    "beetle", "rat", "squirrel", "boar",
)
# Words that make a creature's likeness an object: "a dragon statue", "a chess knight", "a rocking horse"; and things
# named after a person's word: "a hand guard" (a guard for a hand, not a guard).
NOT_A_CHARACTER_WORDS = ("statue", "statuette", "figurine", "sculpture", "carving", "carved", "toy", "chess",
                         "rocking", "plush", "stuffed", "puppet", "ornament", "weathervane", "hand guard", "arm guard",
                         "shin guard", "knee guard", "neck guard", "mouth guard", "finger guard", "fire guard",
                         "cross guard")
# What the picture-against-prompt judge (subject_judge) compares a picture with: the prompt itself, against every other
# kind of thing release 1 makes and a few things a picture might show instead of the one asked for. Each kind is its
# words (any of them in a prompt makes it the prompt's own kind, not a contrast) and one description of its look that
# keeps its name first (classification by description: Menon and Vondrick 2023; see docs/objects.md, "The picture
# against the prompt"). Near kinds that a prompt may name by either word share a row (a jug and a vase), so a jug is
# never refused for looking like a vase.
SUBJECT_KINDS = (
    (("chair", "armchair", "throne"), "a chair, a seat with a back rest and legs"),
    (("stool",), "a stool, a small seat on legs with no back rest"),
    (("bench",), "a bench, a long seat for several people"),
    (("table", "desk"), "a table, a flat top standing on legs"),
    (("barrel", "keg", "cask"), "a barrel, a round wooden cask bound with hoops"),
    (("chest", "trunk", "box", "crate", "coffer"), "a chest, a box with a lid"),
    (("cabinet", "wardrobe", "cupboard", "bookshelf", "shelf", "dresser"), "a cabinet, tall furniture with doors"),
    (("bed",), "a bed, a frame with a mattress to sleep on"),
    (("jug", "pitcher", "jar", "pot", "vase", "urn", "amphora"), "a jug, a round vessel with a handle and a spout"),
    (("bottle", "flask", "potion", "vial"), "a bottle, a narrow-necked vessel with a stopper"),
    (("mug", "cup", "goblet", "tankard", "chalice"), "a cup, a small vessel to drink from"),
    (("bowl",), "a bowl, a round open dish"),
    (("plate", "tray", "dish"), "a plate, a flat round dish"),
    (("bucket", "pail"), "a bucket, an open pail with a hoop handle"),
    (("basket",), "a basket, woven from wicker"),
    (("cauldron", "kettle", "teapot"), "a cauldron, a big round metal pot"),
    (("lantern", "lamp"), "a lantern, a lamp in a frame with glass sides and a handle on top"),
    (("candle", "candlestick", "candelabra"), "a candle, a stick of wax with a wick, in a holder"),
    (("book", "tome"), "a book, its pages bound between two covers"),
    (("scroll", "map", "parchment"), "a scroll, a rolled sheet of parchment"),
    (("anvil",), "an anvil, a heavy iron block with a horn"),
    (("statue", "statuette", "bust", "figurine", "sculpture"), "a statue, a figure carved in stone or cast in metal"),
    (("clock",), "a clock, a dial with hands"),
    (("globe",), "a globe, a ball with a map of the world on a stand"),
    (("coin",), "a coin, a small round disc of metal"),
    (("rock", "boulder", "stone"), "a rock, a rough lump of stone"),
    (("stump", "log"), "a stump, a cut piece of tree trunk with roots"),
    (("mushroom",), "a mushroom, a cap on a stalk"),
    (("well", "fountain"), "a well, a round stone wall round a shaft"),
    (("signpost", "sign"), "a signpost, a post with a board on it"),
    (("sword", "longsword", "greatsword", "broadsword", "rapier", "shortsword", "short sword", "dagger", "knife",
      "dirk", "blade"), "a sword, a long straight blade with a hilt"),
    (("sabre", "saber", "scimitar", "falchion", "cutlass"), "a sabre, a curved blade with a hilt"),
    (("axe", "battle axe", "battleaxe", "hatchet", "halberd", "glaive", "poleaxe", "polearm", "bardiche"),
     "an axe, a broad blade on a wooden handle"),
    (("hammer", "war hammer", "warhammer", "maul", "mallet", "sledgehammer", "sledge hammer"),
     "a hammer, a heavy block of a head on a handle"),
    (("mace", "flail", "morning star", "club"), "a mace, a heavy spiked or flanged head on a short handle"),
    (("spear", "pike", "lance", "javelin", "trident"), "a spear, a long pole with a pointed blade at its tip"),
    (("staff", "quarterstaff", "wand", "rod"), "a staff, a long plain wooden pole"),
    (("bow", "longbow", "shortbow", "crossbow"), "a bow, a curved limb of wood with a string"),
    (("shield", "buckler"), "a shield, a broad plate held on the arm"),
    (("broom",), "a broom, bristles bound to a long handle"),
    (("shovel", "spade"), "a shovel, a broad scoop on a long handle"),
    (("rake", "pitchfork", "hoe"), "a pitchfork, long tines on a long handle"),
    (("scythe", "sickle"), "a scythe, a long curved blade on a pole"),
    (("helmet", "helm"), "a helmet, a metal cover for the head"),
    (("crown", "tiara"), "a crown, a ring of gold with points"),
    (("hat",), "a hat, a cloth cover for the head with a brim"),
    (("boot", "shoe"), "a boot, footwear of leather"),
    (("bag", "sack", "pouch", "backpack"), "a bag, a sack of cloth or leather"),
    (("key",), "a key, a small metal bar with teeth"),
    (("ring", "necklace", "amulet"), "a ring, a piece of jewellery"),
    (("torch",), "a torch, a wooden stick with a flame at its top"),
    (("skull", "bone"), "a skull, the bones of a head"),
)
# What a picture might show instead of a thing: each with its description and the words of a prompt that ask for it
# (then it is no contrast: "a stone statue of a knight" may look like a person).
SUBJECT_DISTRACTORS = {
    "a person": ("a person, a living human being with a face, arms and legs", CHARACTER_WORDS),
    "an animal": ("an animal, a living creature with a head and legs", CHARACTER_WORDS),
    "text": ("text, written words and letters on a plain background",
             ("text", "letter", "letters", "lettering", "word", "words", "writing", "written", "inscription",
              "inscribed", "rune", "runes", "sign", "label", "glyph", "glyphs")),
    "an empty backdrop": ("an empty plain grey backdrop with nothing on it", ()),
    "several things": ("several different objects together in a pile",
                       ("pile", "heap", "stack", "set", "pair", "collection", "several", "group", "bundle")),
}


def subject_contrast(subject):
    """The picture-against-prompt judge's classes: (the words for the thing asked for, {each other kind's name: its
    words}). The thing asked for is the prompt itself and, when one of its words is a kind, that kind's description;
    every kind a word of the prompt names is left out of the others, and so is a distractor the prompt asks for."""
    text = subject.prompt.lower()
    asked = [f"a photo of {subject.prompt}"]
    others = {}
    # A blade the words call curved ("a curved dagger") is judged as a curved blade, a sabre's kind, not against it:
    # with the straight blade's description and the sabre's as a contrast, the trial's curved daggers scored 0.03 to
    # 0.69 as the thing asked for, the likeliest other "a sabre" (docs/objects.md, "Qualities the words name").
    curved_blade = subject.weapon_kind == "curved blade" and subject.noun not in WEAPON_KINDS["curved blade"]
    for nouns, description in SUBJECT_KINDS:
        own = subject.noun in nouns and not curved_blade
        if curved_blade and CURVED_BLADE_KIND in nouns:
            own = True
        if own or subject.noun in nouns or any(_found(text, noun) for noun in nouns):
            if own:
                asked.append(f"{subject.prompt}, {description.split(', ', 1)[1]}")
            continue
        others[description.split(", ", 1)[0]] = [description]
    for name, (description, asked_by) in SUBJECT_DISTRACTORS.items():
        if not any(_found(text, word) for word in asked_by):
            others[name] = [description]
    return asked, others


# A word after a creature's name that still leaves it the subject: "a dragon sleeping", "a bear standing". A word ends
# in -ed or -ing as a participle only when what is left holds a vowel (the Porter stemmer's condition for taking the
# ending off, its step 1b: "sleeping", "spiked", but not "red", "sled", "shed", "king", "wing", "string").
PARTICIPLE = re.compile(r"[a-z]*[aeiouy][a-z]*(?:ing|ed)")
# A thing of no word above: its largest size.
UNKNOWN_SIZE_M = 0.5


@dataclass
class Subject:
    """What the prompt asks for and which checks apply, each with the word that decided it."""

    prompt: str
    kind: str  # "object", "weapon" or "character" (not made: the second release)
    noun: str  # the word that decided the kind ("" when none did)
    weapon_kind: str = ""  # blade, curved blade, hafted, spear, bow, shield ("" for an object)
    handled: bool = False  # a long handle (a hafted weapon or a long-handled tool): shaft and grip checks
    stands: bool = False
    symmetric: str = ""  # "mirror" (an object), "across" (across a blade or haft), "along" (a bow), "" none
    size_m: float = UNKNOWN_SIZE_M  # the largest size (an object) or the length (a weapon)
    not_yet: str = ""  # why this kind of thing is not made yet ("" when it is made)
    state: str = ""  # "closed" or "open" for a thing with a lid, cover or doors ("" for others)
    lies: bool = False  # it rests on its broad face (a book, a coin, a plate): laid on its likeliest resting face
    several: str = ""  # the words that ask for more than one thing ("" for one): not made (SEVERAL_THINGS)
    qualities: list = field(default_factory=list)  # the qualities the words name ("curved", "round"), judged
    why: dict = field(default_factory=dict)

    @property
    def is_weapon(self):
        return self.kind == "weapon"

    @property
    def slug(self):
        return slug_of(self.prompt)


# Markup and signs a description does not need are read as spaces, so nothing hides a word from the checks
# ('a "dragon" <b>' asks for a dragon).
MARKUP = re.compile(r"<[^>]*>")
NOT_PROMPT_SIGNS = re.compile(r"[^\w\s',.\-’]")


def cleaned(prompt):
    """The prompt as read: markup and stray signs gone, one line, its spaces single, at most
    `MOST_PROMPT_CHARACTERS`."""
    text = NOT_PROMPT_SIGNS.sub(" ", MARKUP.sub(" ", str(prompt)))
    return " ".join(text.split())[:MOST_PROMPT_CHARACTERS].strip(" .,")


# A slug is at most this many characters (a word of 256 letters made a folder name too long for the file system).
MOST_SLUG_CHARACTERS = 60


def slug_of(prompt):
    """A short name for the prompt's files: its first four words that say most, at most `MOST_SLUG_CHARACTERS`."""
    words = re.sub(r"[^a-z0-9 ]", " ", cleaned(prompt).lower()).split()
    small = {"a", "an", "the", "in", "of", "with", "and", "looking", "old", "small", "big", "large"}
    kept = [w for w in words if w not in small] or words
    return "-".join(kept[:4])[:MOST_SLUG_CHARACTERS].strip("-") or "object"


def _found(text, word):
    """`word` in `text` as whole words (a plural s allowed)."""
    return re.search(rf"(?<![a-z]){re.escape(word)}s?(?![a-z])", text) is not None


# Words that deny the word after them, a few words on ("a book, never opened", "a chest that cannot be opened").
NEGATIONS = ("not", "never", "cannot", "can't", "cant", "won't", "isn't", "wasn't", "no", "without")
NEGATION_REACH = 3


def _said(text, word):
    """`word` in `text` as whole words (`_found`), at least once not denied by a negation in the
    `NEGATION_REACH` words before it."""
    for match in re.finditer(rf"(?<![a-z]){re.escape(word)}s?(?![a-z])", text):
        before = re.findall(r"[a-z']+", text[:match.start()].replace("\u2019", "'"))[-NEGATION_REACH:]
        if not any(w in NEGATIONS for w in before):
            return True
    return False


# The words that end a prompt's head phrase (with a comma).
HEAD_PHRASE_WORDS = ("in", "with", "of", "on", "under", "holding", "filled")
HEAD_PHRASE_END = re.compile(rf"\b(?:{'|'.join(HEAD_PHRASE_WORDS)})\b|,")
# A character's head phrase also ends at these: "a sword held by a knight" is a sword, "a lamp shaped like an owl" a
# lamp, "a cat and a chair" asks for a cat.
CHARACTER_PHRASE_WORDS = HEAD_PHRASE_WORDS + ("by", "for", "from", "like", "held", "and", "or")
CHARACTER_PHRASE_END = re.compile(rf"\b(?:{'|'.join(CHARACTER_PHRASE_WORDS)})\b|,")
# The head phrase for counting things also ends at these: "a bench for two people" is one bench.
COUNT_PHRASE_WORDS = HEAD_PHRASE_WORDS + ("for", "by", "from", "like", "held", "to", "as", "at")
COUNT_PHRASE_END = re.compile(rf"\b(?:{'|'.join(COUNT_PHRASE_WORDS)})\b|,")
# Words that join two things: "a sword and a shield".
JOINED = re.compile(r"\s(?:and|&|plus)\s")


def _head_phrase(text):
    """The words before the first "in", "with", "of", "on" or comma: English puts the head noun there ("a candle
    in a brass holder" is a candle, "a wooden barrel with iron hoops" a barrel)."""
    return HEAD_PHRASE_END.split(text, maxsplit=1)[0]


def _named(text, table):
    """The table's word that is the text's head noun: within the head phrase if any is (else anywhere), the one
    ending last (the noun after its adjectives: "a wooden treasure chest"), the longer of two ending together
    ("battle axe" over "axe")."""
    for part in (_head_phrase(text), text):
        found = [w for w in table if _found(part, w)]
        if found:
            return max(found, key=lambda w: (part.rfind(w) + len(w), len(w)))
    return ""


def known_things():
    """Every word of a thing release 1 knows: the objects, the things held in the hand, and the judge's kinds."""
    return {*STANDING, *NOT_STANDING, *(w for table in WEAPON_KINDS.values() for w in table),
            *(noun for nouns, _ in SUBJECT_KINDS for noun in nouns)}


def plurals_of(noun):
    """A known thing's plural forms: with s; with es after s, x, z, ch or sh ("boxes", "benches"); with ves for f or
    fe ("shelves", "knives"); and its own irregular ones ("staves")."""
    forms = {noun + "s", *IRREGULAR_PLURALS.get(noun, ())}
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        forms.add(noun + "es")
    if noun.endswith("fe"):
        forms.add(noun[:-2] + "ves")
    elif noun.endswith("f") and not noun.endswith("ff"):
        forms.add(noun[:-1] + "ves")
    return forms


def _last_thing(phrase, things):
    """The known thing named last in `phrase` as (its word as written, the thing, whether it is a plural), or None:
    the one ending last, the longer of two ending together ("battle axes" over "axes")."""
    found = []
    for thing in things:
        for form, plural in ((thing, False), *((p, True) for p in plurals_of(thing))):
            for match in re.finditer(rf"(?<![a-z]){re.escape(form)}(?![a-z])", phrase):
                found.append((match.end(), len(form), form, thing, plural))
    if not found:
        return None
    _, _, form, thing, plural = max(found)
    return form, thing, plural


def _article(words_after):
    return ("an " if words_after[:1] in tuple("aeiou") else "a ") + words_after


def _one_of(text, things):
    """The prompt asking for one of its things: the count, the group and the article dropped, each known plural made
    one, the words after the head phrase kept ("three iron swords with gold hilts": "an iron sword with gold hilts");
    or "" when no known thing is named."""
    head = COUNT_PHRASE_END.split(text, maxsplit=1)[0]
    group = re.match(rf"^(.*?)\b(?:{'|'.join(GROUP_WORDS)})\s+of\s+(.*)$", text)
    if group:
        before, text = group.group(1), group.group(2)
        head = before + COUNT_PHRASE_END.split(text, maxsplit=1)[0]
        rest = text[len(COUNT_PHRASE_END.split(text, maxsplit=1)[0]):]
    else:
        rest = text[len(head):]
    head = JOINED.split(head)[0]
    plural_to_one = {p: thing for thing in things for p in plurals_of(thing)}
    kept = []
    for word in re.findall(r"[a-z0-9'\-]+", head):
        if word in ("a", "an", "the", "some", "both", "of") or word in COUNT_WORDS or word in GROUP_WORDS or \
                word.isdigit():
            continue
        kept.append(plural_to_one.get(word, word))
    if kept and _last_thing(" ".join(kept), things) is None:
        # No known thing: the last word made one by the plain rules, so the user is still shown what to ask; when it
        # is no plural by them ("a set of armour"), nothing is suggested.
        one = re.sub(r"(?:(?<=[sxz])|(?<=[cs]h))es$|(?<![su])s$", "", re.sub(r"ies$", "y", kept[-1]))
        if len(kept[-1]) <= 3 or one == kept[-1]:
            return ""
        kept[-1] = one
    if not kept:
        return ""
    one = " ".join(kept)
    rest = rest.strip()
    return _article(one) + ("" if not rest else rest if rest.startswith(",") else " " + rest)


def several_of(text):
    """Why the (lowered) text asks for more than one thing ("" when it asks for one), and the prompt for one of them
    ("" when none can be named). See SEVERAL_THINGS."""
    things = known_things()
    head = _head_phrase(text)
    words_of_head = re.findall(r"[a-z0-9'\-]+", head)
    after_head = text[len(head):].lstrip()
    if words_of_head and words_of_head[-1] in GROUP_WORDS and after_head.startswith("of "):
        grouped = COUNT_PHRASE_END.split(after_head[len("of "):], maxsplit=1)[0]
        if not any(_found(grouped, word) for word in ONE_THING_IN_PAIRS):
            return f"'{words_of_head[-1]} of'", _one_of(text, things)
    phrase = COUNT_PHRASE_END.split(text, maxsplit=1)[0]
    for match in re.finditer(r"(?<![a-z0-9\-])([a-z]+|\d+)(?![a-z0-9])(-?)", phrase):
        word, joined = match.group(1), match.group(2)
        counts = word in COUNT_WORDS or (word.isdigit() and int(word) >= 2)
        if not counts or joined:
            continue
        after = re.findall(r"[a-z']+", phrase[match.end():])
        # "a three legged stool": a count before a participle describes; a count before a plural asks for several.
        if after and PARTICIPLE.fullmatch(after[0]):
            continue
        if any(w.endswith("s") and not w.endswith(("ss", "us", "is")) and w not in MEASURES for w in after):
            return f"the count '{word}'", _one_of(text, things)
    last = _last_thing(phrase, things)
    if last and last[2]:
        return f"the plural '{last[0]}'", _one_of(text, things)
    named = [part for part in JOINED.split(phrase) if _last_thing(part, things)]
    if len(named) >= 2:
        first, second = _last_thing(named[0], things)[1], _last_thing(named[1], things)[1]
        return f"'{first}' and '{second}'", _one_of(text, things)
    return "", ""


def qualities_of(text, noun):
    """The qualities (QUALITY_WORDS) the head phrase of the (lowered) text gives the thing, each once and in the order
    said, with the word that said it: [(quality, word)]. Not one denied, joined to the next word, followed by a
    participle, or part of the thing's own word ("a short sword") or kind ("a curved sabre"); one the words also
    contradict ("a round square table") is left out."""
    head = _head_phrase(text)
    found = {}
    for word, quality in QUALITY_WORDS.items():
        for match in re.finditer(rf"(?<![a-z\-]){word}(?![a-z])", head):
            after = head[match.end():]
            following = re.findall(r"[a-z]+", after)
            # A past participle after it describes a part ("a tall backed chair"); a word in -ing does not ("a curved
            # hunting knife", "a round cooking pot").
            if after.startswith("-") or (following and following[0].endswith("ed") and
                                         PARTICIPLE.fullmatch(following[0])):
                continue
            if noun and (word in noun.split() or noun in QUALITY_OF_THE_KIND.get(quality, ())):
                continue
            if noun in QUALITY_NOT_JUDGED.get(quality, ()):
                continue
            if not _said(head, word):
                continue
            found.setdefault(quality, (match.start(), word))
    kept = [(quality, word) for quality, (_, word) in sorted(found.items(), key=lambda item: item[1][0])]
    return [(quality, word) for quality, word in kept
            if not any(_said(head, other) for other in QUALITY_CONTRADICTED_BY[quality])]


def _curved_said(text):
    """Whether the head phrase calls the whole thing curved, as `qualities_of` reads a quality, for a thing whose
    curve is not judged in its picture (QUALITY_NOT_JUDGED)."""
    return any(quality == "curved" for quality, _ in qualities_of(text, ""))


def quality_texts(subject):
    """The picture judge's contrast for each quality the words name: [(quality, wordings of it, wordings of its
    opposite)], paired by place."""
    noun = subject.noun or UNKNOWN_THING
    return [(quality, [asked.format(noun=noun) for asked, _ in QUALITY_JUDGE_TEXTS[quality]],
             [other.format(noun=noun) for _, other in QUALITY_JUDGE_TEXTS[quality]])
            for quality in subject.qualities]


def character_of(text):
    """The person, animal or creature the (lowered) text asks for, or "" when it asks for none: the creature's name
    ending last in the head phrase, not owned ("a king's crown") nor joined to the next word ("a dragon-headed staff"),
    and followed by nothing or by a participle ("a dragon sleeping"), not by another noun ("a wolf skull"); and no
    word that makes it a likeness ("a dragon statue")."""
    head = CHARACTER_PHRASE_END.split(text, maxsplit=1)[0]
    if any(_found(head, word) for word in NOT_A_CHARACTER_WORDS):
        return ""
    last = None
    for word in CHARACTER_WORDS:
        for match in re.finditer(rf"(?<![a-z]){re.escape(word)}s?(?![a-z])", head):
            if head[match.end():].startswith(("'", "\u2019", "-")):
                continue
            if last is None or (match.end(), len(word)) > last[:2]:
                last = (match.end(), len(word), word)
    if last is None:
        return ""
    after = re.findall(r"[a-z]+", head[last[0]:])
    things = {*STANDING, *NOT_STANDING, *(w for table in WEAPON_KINDS.values() for w in table)}
    if not after:
        return last[2]
    # A participle and then a thing: the creature's name and the participle say what the thing is like ("a wolf
    # headed staff", "a king sized bed", "a giant spiked mace"), and the thing is what is asked for.
    if PARTICIPLE.fullmatch(after[0]) and after[0] not in things and not any(_found(" ".join(after[1:]), thing)
                                                                             for thing in things):
        return last[2]
    return ""


# A prompt with nothing left to read is not made (a program in front of the pipeline can refuse it with the same
# words).
EMPTY_PROMPT = "Describe the 3-D object you want made."


def subject_of(prompt):
    """The subject a prompt asks for. An unknown thing is an object that is not assumed to stand or be symmetric."""
    text = cleaned(prompt).lower()
    why = {}
    if not text:
        why["kind"] = "nothing to read: the prompt is empty"
        return Subject(prompt="", kind="object", noun="", why=why, not_yet=EMPTY_PROMPT)
    character = character_of(text)
    if character:
        why["kind"] = f"a character: the word '{character}'; characters come in the second release"
        return Subject(prompt=cleaned(prompt), kind="character", noun=character, why=why, not_yet=CHARACTERS_LATER)
    several, one = several_of(text)
    if several:
        if one and (several_of(one)[0] or character_of(one)):
            one = ""
        named = _last_thing(one, known_things()) if one else None
        why["kind"] = f"more than one thing: {several}; model-maker makes one object at a time"
        return Subject(prompt=cleaned(prompt), kind="object", noun=named[1] if named else "", why=why, several=several,
                       not_yet=SEVERAL_THINGS.format(one=one) if one else SEVERAL_THINGS_NO_ONE)
    weapon_words = {w: (kind, size) for kind, table in WEAPON_KINDS.items() for w, size in table.items()}
    weapon = _named(text, weapon_words)
    if weapon == "bow" and any(w in text for w in NOT_WEAPON):
        weapon = ""
    things = {**STANDING, **NOT_STANDING}
    thing = _named(text, things)
    if thing == "plate" and any(_found(text, w) for w in NOT_PLATE):
        thing = ""
    # "a chest with a sword carved on it" is a chest; "a sword in a wooden scabbard" is a sword: the head noun wins.
    if weapon and thing:
        head = _head_phrase(text)
        if _found(head, thing) and not _found(head, weapon):
            weapon = ""
    if weapon:
        kind, size = weapon_words[weapon]
        why["kind"] = f"held in the hand: the word '{weapon}' ({kind})"
        qualities = qualities_of(text, weapon)
        if kind == "blade" and (any(quality == "curved" for quality, _ in qualities) or
                                ("curved" in QUALITY_NOT_JUDGED and weapon in QUALITY_NOT_JUDGED["curved"] and
                                 _curved_said(text))):
            # A curved dagger is checked as a sabre is: its curve without a kink, no straight line, no mirror.
            kind = "curved blade"
            why["kind"] = f"held in the hand: the word '{weapon}' (a blade), and 'curved': a curved blade"
        handled = kind in ("hafted", "spear")
        symmetric = ""
        if kind in ("blade", "spear"):
            symmetric = "across"
            why["symmetric"] = f"across the blade: a {kind} is mirror-symmetric across its width"
        elif kind == "hafted" and any(w in text for w in TWO_SIDED_WORDS):
            symmetric = "across"
            why["symmetric"] = "across the haft: the words name a two-sided head"
        elif kind == "bow":
            symmetric = "along"
            why["symmetric"] = "about its grip: a bow's two limbs mirror each other"
        elif kind == "shield" and any(_found(text, w) for w in ("round", "circular", "kite", "heater", "oval")):
            symmetric = "mirror"
            why["symmetric"] = "left to right: the words name a regular shield's outline"
        if kind == "hafted" and weapon in NOT_SYMMETRIC_THINGS and not symmetric:
            why["symmetric"] = f"not asked: a {weapon}'s head is one-sided unless the words say double"
        if any(_said(text, w) for w in NOT_SYMMETRIC_WORDS):
            symmetric = ""
            why["symmetric"] = "not asked: the words say it is broken, bent or in disorder"
        _say_qualities(why, qualities)
        return Subject(prompt=cleaned(prompt), kind="weapon", noun=weapon, weapon_kind=kind, handled=handled,
                       stands=False, symmetric=symmetric, size_m=size, why=why, not_yet=NOT_YET.get(kind, ""),
                       qualities=[quality for quality, _ in qualities])
    stands = thing in STANDING
    size = things.get(thing, UNKNOWN_SIZE_M)
    why["kind"] = f"object: the word '{thing}'" if thing else "object: no word known; nothing is assumed"
    why["stands"] = (f"stands: a {thing} stands on its own base" if stands else
                     f"does not stand: a {thing} leans, hangs or lies loose" if thing else
                     "not asked: no word says it stands")
    symmetric = "mirror" if thing and thing not in NOT_SYMMETRIC_THINGS else ""
    why["symmetric"] = (f"left to right: a {thing} is made to a plan" if symmetric else
                        f"not asked: a {thing} is not mirror-symmetric" if thing else "not asked: no word known")
    if symmetric and any(_said(text, w) for w in NOT_SYMMETRIC_WORDS):
        symmetric = ""
        why["symmetric"] = "not asked: the words say it is broken, open or in disorder"
    handled = thing in HANDLED_TOOLS
    if handled:
        why["handled"] = f"a long handle: a {thing}'s shaft and grip are checked as an axe's or a hammer's haft is"
    state = ""
    if thing in OPENING_PARTS:
        part = OPENING_PARTS[thing]
        opened = next((w for w in OPEN_WORDS if _said(text, w)), "")
        shut = next((w for w in SHUT_WORDS if _said(text, w)), "")
        state = "open" if opened else "closed"
        why["state"] = (f"open: the word '{opened}'" if opened else f"closed: the word '{shut}'" if shut else
                        f"closed: a {thing} is painted with its {part} shut unless the words open it")
    lies = thing in LYING
    if lies:
        upright = next((w for w in UPRIGHT_WORDS if _found(text, w)), "")
        lies = not upright
        why["lies"] = (f"left as made: the words '{upright}'" if upright else
                       f"lies: a {thing} rests on its broad face; laid on the face it most often comes to rest on")
    qualities = qualities_of(text, thing)
    _say_qualities(why, qualities)
    return Subject(prompt=cleaned(prompt), kind="object", noun=thing, handled=handled, stands=stands,
                   symmetric=symmetric, size_m=size, why=why, state=state, lies=lies,
                   qualities=[quality for quality, _ in qualities])


def _say_qualities(why, qualities):
    if qualities:
        why["qualities"] = "; ".join(f"{quality}: the word '{word}'; the picture must show it, not "
                                     f"{QUALITY_OPPOSITES[quality]}" for quality, word in qualities)


def state_texts(subject):
    """The picture judge's contrast: (wordings of the state the words ask for, wordings of the other state); both
    empty for a thing with no lid, cover or doors."""
    if not subject.state:
        return [], []
    pairs = [(shut.format(noun=subject.noun), opened.format(noun=subject.noun))
             for shut, opened in STATE_JUDGE_TEXTS[OPENING_PARTS[subject.noun]]]
    shut, opened = [s for s, _ in pairs], [o for _, o in pairs]
    return (shut, opened) if subject.state == "closed" else (opened, shut)


def picture_prompt(subject, lead=DEFAULT_LEAD):
    """The words the picture model is given: the light first, then the user's words, then the view and style."""
    if subject.weapon_kind == "shield":
        view = SHIELD_VIEW
    elif subject.weapon_kind == "bow":
        view = BOW_VIEW
    elif subject.weapon_kind == "hafted":
        view = HAFTED_VIEW
    elif subject.weapon_kind == "curved blade":
        view = CURVED_BLADE_VIEW
    elif subject.is_weapon:
        view = WEAPON_VIEW
    elif subject.handled:
        view = TOOL_VIEW
    else:
        view = OBJECT_VIEW
    state = f", {STATE_PICTURE_WORDS[OPENING_PARTS[subject.noun]][subject.state]}" if subject.state else ""
    head = f", {HEAD_PICTURE_WORDS[HEAD_KINDS[subject.noun]]}" if subject.noun in HEAD_KINDS else ""
    curve = f", {CURVED_BLADE_WORDS}" if subject.weapon_kind == "curved blade" else ""
    return f"{LEADS[lead]} {subject.prompt}{state}{head}{curve}, {view}, {STYLE}"


def paint_prompt(subject):
    """The words the view painter is given beside the picture (it sees the shape and the picture too)."""
    return f"{subject.prompt}, high quality, even light"
