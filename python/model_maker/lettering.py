"""The picture check "no painted lettering": letters, a logo or a signature painted on the object.

The picture model sometimes paints letters on a thing nobody asked to carry any: release 1's unseen run passed "an
iron anvil" with three lines of made-up words on its side, and they went on into its views and its paint. The picture
words say "no text, no signature", but the picture model's first text encoder (CLIP) reads only 77 tokens and cuts
them off (docs/setup.md, "Warnings you can ignore"). So each try's cut-out is read, and a try with letters is not
chosen (another try is, or another round is painted).

The reader is tesseract (Apache-2.0), version 5 or newer with its English data, an optical character recogniser, run as
its own program (`tesseract`, found on the PATH). It is probed once (`reader`): its version, and its English data. It
is optional: without it, or with one that cannot be used (version 4, which has no Sauvola threshold and would read
every region with Otsu's alone; no English data; a reading that fails or takes too long), the check says "not judged"
and why, and is shown, not counted; it never passes unread, and the job says it was not read (`lettering_judged`).
A run that must read every picture (the measured run, `--require-lettering`) does not make a job without the reader,
and counts a try it could not read as failed.

The region round the cut-out is read in grey, the backdrop made white, stretched to its full contrast, at twice its
size, as painted and inverted (letters cut light into dark iron, and dark on light wood), each with one threshold for
the whole region and with one set from the light round each pixel, in tesseract's sparse-text mode, which looks for
words anywhere rather than for a page.

Grain, stone joints, rivets and hoops are read as short scraps ("iit", "Wil", "SSS") at middling confidence. A word
counts only when it is read with a confidence of at least `LEAST_CONFIDENCE`, has at least `LEAST_LETTERS` letters
and `LEAST_WORD_CHARACTERS` letters and digits, is nearly all letters and digits once the punctuation round it is
trimmed, and is at least `LEAST_WORD_HEIGHT` of the picture's side tall. The picture has lettering when its counted
words hold more than `MOST_CHARACTERS` letters and digits between them. The limits were set on the trials' tries,
labelled by eye (docs/objects.md, "Painted lettering"; tools/trial/lettering-cases.toml), so that no try a trial
rightly chose is refused, and no try without lettering.

The words can ask for lettering ("a sign that says OPEN", "a book with a title on its cover": words.lettering_asked):
then the letters are read and listed, and the check passes.
"""

import io
import os
import re
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .picture_checks import Check

NAME = "no painted lettering"
TESSERACT = "tesseract"
NOT_JUDGED = "not judged"
NOT_INSTALLED = f"{NOT_JUDGED} (tesseract is not installed)"
# Sauvola's threshold (`thresholding_method`) came with tesseract 5: version 4 takes the option without it, says "Could
# not set option" on its errors, and reads with Otsu's alone.
LEAST_VERSION = 5
UNSET_OPTION = "Could not set option"
ENGLISH = "eng"
# The probe's two short runs (--version, --list-langs) are given at most this long.
PROBE_SECONDS = 20
# The region read: the cut-out's bounding box and this many pixels round it.
CROP_PAD = 6
# Tesseract reads letters best some 20 to 60 pixels tall, and a painted word is often 10 to 20: the region is read at
# twice its size.
READ_SCALE = 2.0
# Tesseract's page segmentation mode 11: sparse text, as many words as it can find, in no particular order.
SPARSE_TEXT = "11"
# Tesseract's ways of telling ink from ground: Otsu's one threshold for the whole region, and Sauvola's, set from the
# light round each pixel (grey letters on a black blade beside a pale handle are lost to one threshold).
OTSU, SAUVOLA = "0", "2"
# The readings of each region: (which way round, how ink is told from ground, at what scale), each way round with each
# threshold. Measured and left out: the same four at half size, for letters over 60 pixels tall (a sign's): on the
# labelled tries they found more only of signs whose words asked for their letters ("OPEN").
READINGS = tuple((way, thresholding, READ_SCALE) for thresholding in (OTSU, SAUVOLA)
                 for way in ("as painted", "inverted"))
# Each reading is given at most this long (a 1024-pixel picture's region reads in well under a second).
READ_SECONDS = 60
# A word counts (see the module's words): its confidence (0-100), letters, letters and digits, and height as a share
# of the picture's side (8 pixels of 1024).
LEAST_CONFIDENCE = 70
LEAST_LETTERS = 2
LEAST_WORD_CHARACTERS = 3
LEAST_SIGN_SHARE = 0.8
LEAST_WORD_HEIGHT = 8 / 1024
# The counted words of a picture hold at most this many letters and digits between them; more is lettering.
MOST_CHARACTERS = 4
# Two readings of one word (as painted and inverted) overlap at least this much (intersection over union).
SAME_WORD_OVERLAP = 0.5
TRIMMED = ".,;:'\"`‘’“”!?()[]{}<>|_-—–~*"


@dataclass
class Word:
    text: str
    confidence: float
    left: float
    top: float
    width: float
    height: float
    way: str  # the reading it came from (`reading_name`)

    @property
    def trimmed(self):
        return self.text.strip(TRIMMED)

    def as_dict(self):
        return {"text": self.text, "confidence": round(self.confidence, 1), "box": [round(self.left, 1),
                round(self.top, 1), round(self.width, 1), round(self.height, 1)], "way": self.way}


@dataclass(frozen=True)
class Reader:
    """What the probe found: whether tesseract can read for the check, its version ("5.5.0"; None when it gave none),
    and why it cannot ("" when it can)."""

    usable: bool
    version: str = None
    why: str = ""


class ReadingFailed(Exception):
    """A reading tesseract did not finish as asked: it exited with an error, or did not take an option."""


# The probe's answer per tesseract, PATH and TESSDATA_PREFIX: probed once, not for every try.
_READERS = {}


def reader():
    """The reader as probed (once for each tesseract, PATH and TESSDATA_PREFIX this process runs with)."""
    key = (TESSERACT, os.environ.get("PATH"), os.environ.get("TESSDATA_PREFIX"))
    if key not in _READERS:
        _READERS[key] = probed()
    return _READERS[key]


def forget_reader():
    """The probe's answers forgotten (a test, or a process that installed tesseract since)."""
    _READERS.clear()


def tool_run(arguments, seconds):
    """`tesseract ARGUMENTS`: its exit code and its output and errors as text."""
    done = subprocess.run([TESSERACT, *arguments], capture_output=True, timeout=seconds,
                          env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    return done.returncode, done.stdout.decode("utf-8", "replace"), done.stderr.decode("utf-8", "replace")


def last_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:200] if lines else "nothing said"


def probed():
    """Whether tesseract is here, runs, is version 5 or newer, and has its English data."""
    if shutil.which(TESSERACT) is None:
        return Reader(False, None, NOT_INSTALLED)
    try:
        code, out, errors = tool_run(["--version"], PROBE_SECONDS)
    except (OSError, subprocess.SubprocessError) as error:
        return Reader(False, None, f"{NOT_JUDGED} (tesseract does not run: {type(error).__name__}: {error})")
    # Version 3 and some builds of 4 print it on their errors, not their output.
    found = re.search(r"tesseract\s+v?(\d+)\.(\d+)(?:\.(\d+))?", out + "\n" + errors, re.IGNORECASE)
    if code != 0 or found is None:
        return Reader(False, None, f"{NOT_JUDGED} (tesseract --version gave no version: exit {code}, "
                                   f"{last_line(errors or out)})")
    version = ".".join(part for part in found.groups() if part is not None)
    if int(found.group(1)) < LEAST_VERSION:
        return Reader(False, version, f"{NOT_JUDGED} (tesseract {version}: version {LEAST_VERSION} or newer is needed)")
    try:
        code, out, errors = tool_run(["--list-langs"], PROBE_SECONDS)
    except (OSError, subprocess.SubprocessError) as error:
        return Reader(False, version, f"{NOT_JUDGED} (tesseract {version} does not list its languages: "
                                      f"{type(error).__name__}: {error})")
    # The first line names the folder ("List of available languages in ..."), one language a line after it.
    languages = {line.strip() for line in (out + "\n" + errors).splitlines()[1:]}
    if code != 0 or ENGLISH not in languages:
        return Reader(False, version, f"{NOT_JUDGED} (tesseract {version} has no English data, {ENGLISH}: "
                                      f"{last_line(errors or out)})")
    return Reader(True, version, "")


def was_read(check):
    """Whether a lettering check (a Check or its record) was read, rather than "not judged"."""
    words = check["words"] if isinstance(check, dict) else check.words
    return not words.startswith(NOT_JUDGED)


def regions(rgb, mask):
    """The region round the cut-out, in grey, as painted and inverted: each with the backdrop white and its contrast
    stretched over the object's own greys (the 1st to the 99th percentile); and the region's top-left corner in the
    picture."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    top, bottom = max(rows[0] - CROP_PAD, 0), min(rows[-1] + CROP_PAD + 1, mask.shape[0])
    left, right = max(cols[0] - CROP_PAD, 0), min(cols[-1] + CROP_PAD + 1, mask.shape[1])
    grey = rgb[top:bottom, left:right, :3].astype(np.float64) @ [0.299, 0.587, 0.114]
    inside = mask[top:bottom, left:right]
    found = []
    for way, seen in (("as painted", grey), ("inverted", 255 - grey)):
        low, high = np.percentile(seen[inside], [1, 99])
        stretched = np.clip((seen - low) / max(high - low, 1.0) * 255, 0, 255)
        stretched[~inside] = 255
        found.append((way, Image.fromarray(stretched.astype(np.uint8))))
    return found, (left, top)


def read(image, thresholding=OTSU):
    """The words tesseract reads in a grey picture: [(text, confidence, left, top, width, height)] in its pixels.
    Raises ReadingFailed when tesseract exits with an error or does not take an option, and
    subprocess.TimeoutExpired after `READ_SECONDS`."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    # One thread a reading: tesseract's own threads slow a short reading down.
    done = subprocess.run([TESSERACT, "stdin", "stdout", "--psm", SPARSE_TEXT, "-l", ENGLISH, "-c",
                           f"thresholding_method={thresholding}", "tsv"],
                          input=buffer.getvalue(), capture_output=True, timeout=READ_SECONDS,
                          env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    errors = done.stderr.decode("utf-8", "replace")
    if done.returncode != 0:
        raise ReadingFailed(f"tesseract exited {done.returncode}: {last_line(errors)}")
    if UNSET_OPTION in errors:
        raise ReadingFailed(f"tesseract did not take thresholding_method={thresholding}: {last_line(errors)}")
    return words_of_tsv(done.stdout.decode("utf-8", "replace"))


def words_of_tsv(tsv):
    """The words of tesseract's TSV: [(text, confidence, left, top, width, height)]. Its header, lines that are not a
    word (level 5), words with no text, and lines cut short or with a field that is not a number are passed over."""
    words = []
    for line in tsv.splitlines()[1:]:
        parts = line.split("\t")
        # Level 5 is a word; its text is the 12th column.
        if len(parts) < 12 or parts[0] != "5" or not parts[11].strip():
            continue
        try:
            left, top, width, height = (int(p) for p in parts[6:10])
            confidence = float(parts[10])
        except ValueError:
            continue
        words.append((parts[11].strip(), confidence, left, top, width, height))
    return words


def overlap(a, b):
    """Two words' boxes' intersection over their union."""
    width = min(a.left + a.width, b.left + b.width) - max(a.left, b.left)
    height = min(a.top + a.height, b.top + b.height) - max(a.top, b.top)
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    return shared / (a.width * a.height + b.width * b.height - shared)


def words_in(rgb, mask, readings=READINGS):
    """Every word read in the cut-out's region, in the picture's pixels; a word read more than once is kept once, as
    its likeliest reading (words that do not count are kept too: `lettering_of` keeps once only those that count)."""
    return once(every_reading(rgb, mask, readings))


def every_reading(rgb, mask, readings=READINGS):
    """Every word of every reading (`readings`) of the cut-out's region, in the picture's pixels."""
    found, (left, top) = regions(rgb, mask)
    words = []
    for way, image in found:
        for wanted_way, thresholding, scale in readings:
            if wanted_way != way:
                continue
            scaled = image.resize((max(round(image.width * scale), 1), max(round(image.height * scale), 1)),
                                  Image.LANCZOS)
            for text, confidence, x, y, width, height in read(scaled, thresholding):
                words.append(Word(text, confidence, x / scale + left, y / scale + top, width / scale, height / scale,
                                  reading_name(way, thresholding, scale)))
    return words


def reading_name(way, thresholding, scale):
    """A reading's name, as a word read in it records it: "as painted, Otsu, x2"."""
    return f"{way}, {'Sauvola' if thresholding == SAUVOLA else 'Otsu'}, x{scale:g}"


def same_word(a, b):
    """Whether two words read are one word read twice: their boxes overlap `SAME_WORD_OVERLAP`, or they touch and one's
    text is the other's or holds it (a looser box round the same word in another reading, or one reading's box round
    two words where another boxed each)."""
    if overlap(a, b) >= SAME_WORD_OVERLAP:
        return True
    first, second = a.trimmed.casefold(), b.trimmed.casefold()
    return overlap(a, b) > 0 and bool(first) and bool(second) and (first in second or second in first)


def once(words):
    """The words, one read more than once (`same_word`) kept as its likeliest reading."""
    kept = []
    for word in sorted(words, key=lambda w: -w.confidence):
        if not any(same_word(word, other) for other in kept):
            kept.append(word)
    return kept


def counts(word, side):
    """Whether a word read counts as lettering (see the module's words); `side` the picture's longer side."""
    text = word.trimmed
    letters = sum(c.isalpha() for c in text)
    characters = sum(c.isalnum() for c in text)
    return (word.confidence >= LEAST_CONFIDENCE and letters >= LEAST_LETTERS and characters >= LEAST_WORD_CHARACTERS
            and characters >= LEAST_SIGN_SHARE * len(text) and word.height >= LEAST_WORD_HEIGHT * side)


def lettering_of(words, side):
    """The counted words, each kept once, and the letters and digits they hold between them. The words that count are
    picked first, then kept once: a scrap read with more confidence over a word that counts must not hide it."""
    counted = once([w for w in words if counts(w, side)])
    return counted, sum(sum(c.isalnum() for c in w.trimmed) for w in counted)


def not_judged(why, required=False):
    """The check when the picture could not be read: shown, not counted, never a pass; counted as failed when the run
    requires the reader (`required`)."""
    return Check(NAME, 0.0, MOST_CHARACTERS, "characters", False,
                 f"{why}: letters painted on the object are not looked for (docs/setup.md)"
                 + ("; the run requires the reader, so the try is not chosen" if required else ""),
                 counted=required)


def lettering_check(rgb, mask, asked="", required=False):
    """The check of a picture (H x W x 3, uint8) and its cut-out (H x W, bool); `asked` the word that asks for
    lettering (words.lettering_asked), "" when none does; `required`: a try not read is counted as failed."""
    found = reader()
    if not found.usable:
        return not_judged(found.why, required)
    if not mask.any():
        return not_judged(f"{NOT_JUDGED} (no cut-out to read)", required)
    try:
        every = every_reading(rgb, mask)
    except (ReadingFailed, subprocess.SubprocessError, OSError) as error:
        return not_judged(f"{NOT_JUDGED} (tesseract {found.version} failed: {type(error).__name__}: {error})"[:400],
                          required)
    counted, characters = lettering_of(every, max(mask.shape))
    read_words = ", ".join(f'"{w.trimmed}"' for w in counted[:6]) + (" ..." if len(counted) > 6 else "")
    if asked:
        return Check(NAME, characters, MOST_CHARACTERS, "characters", True,
                     f"the words ask for lettering (the word '{asked}'): not refused; "
                     + (f"read {read_words}" if counted else "no lettering read"))
    if characters > MOST_CHARACTERS:
        return Check(NAME, characters, MOST_CHARACTERS, "characters", False,
                     f"lettering painted on the object: read {read_words} ({characters} characters; at most "
                     f"{MOST_CHARACTERS}), and the words do not ask for any")
    return Check(NAME, characters, MOST_CHARACTERS, "characters", True,
                 "no lettering read" + (f" (only {read_words}: {characters} characters, at most {MOST_CHARACTERS})"
                                        if counted else ""))


def add_lettering_check(result, rgb, mask, asked="", required=False):
    """The check added to a try's checks (`picture_checks.PictureResult`): in the pipeline's choice of try, so a try
    with letters the words do not ask for is not chosen."""
    result.checks.append(lettering_check(rgb, mask, asked, required))
    return result.checks[-1]
