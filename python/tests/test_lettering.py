import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from model_maker import lettering, picture_checks, words

# CI installs tesseract and fails when these skip (.github/workflows/ci.yml).
NEEDS_TESSERACT = unittest.skipUnless(lettering.reader().usable, f"no usable tesseract: {lettering.reader().why}")
TSV_HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
USABLE = lettering.Reader(True, "5.5.0", "")


def slab_with_text(text, size=30, ink=(170, 170, 170), slab=(30, 30, 32)):
    """A pale backdrop with a dark slab (the cut-out) and `text` written on it; ("" for none)."""
    from PIL import Image, ImageDraw, ImageFont

    picture = Image.new("RGB", (1024, 1024), (225, 225, 225))
    draw = ImageDraw.Draw(picture)
    draw.rectangle((300, 300, 750, 520), fill=slab)
    if text:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            raise unittest.SkipTest("Pillow's default font cannot be sized here")
        draw.text((340, 380), text, fill=ink, font=font)
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[300:521, 300:751] = True
    return np.asarray(picture), mask


def word(text, confidence=90.0, height=20.0):
    return lettering.Word(text, confidence, 400.0, 400.0, 10.0 * len(text), height, "as painted")


class WhatCounts(unittest.TestCase):
    def test_a_clear_word_counts(self):
        self.assertTrue(lettering.counts(word("EANTIES"), 1024))

    def test_grain_scraps_do_not(self):
        for scrap in (word("iit", confidence=60.0), word("SS"), word("|||"), word("a1"), word("—", confidence=95.0)):
            self.assertFalse(lettering.counts(scrap, 1024), scrap.text)

    def test_letters_too_small_to_be_painted_lettering_do_not(self):
        self.assertFalse(lettering.counts(word("WORD", height=5.0), 1024))

    def test_a_word_mostly_of_signs_does_not(self):
        self.assertFalse(lettering.counts(word("a#%&b(c"), 1024))

    def test_punctuation_round_a_word_is_trimmed(self):
        self.assertTrue(lettering.counts(word("“OPEN,”"), 1024))

    def test_several_characters_make_lettering(self):
        counted, characters = lettering.lettering_of([word("BLHOM"), word("ail", confidence=40.0)], 1024)
        self.assertEqual([w.text for w in counted], ["BLHOM"])
        self.assertEqual(characters, 5)

    def test_one_reading_of_a_word_both_ways_is_kept_once(self):
        a, b = word("OPEN"), word("OPEN", confidence=80.0)
        self.assertGreater(lettering.overlap(a, b), lettering.SAME_WORD_OVERLAP)
        kept = lettering.once([b, a])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].confidence, 90.0)


def boxed(text, left, top, width, height, confidence=90.0, way="as painted"):
    return lettering.Word(text, confidence, left, top, width, height, way)


class KeptOnce(unittest.TestCase):
    """`once()` and `lettering_of()`: a word read in several readings counts once (the 0.2.0 review, missing test 3)."""

    def test_a_short_word_read_twice_in_looser_boxes_counts_once(self):
        # "RIP" read twice would be 6 characters, over the limit of 4: refused for 3 letters.
        tight, loose = boxed("RIP", 400, 400, 30, 20), boxed("RIP", 390, 395, 60, 40, confidence=80.0, way="inverted")
        self.assertLess(lettering.overlap(tight, loose), lettering.SAME_WORD_OVERLAP)
        counted, characters = lettering.lettering_of([tight, loose], 1024)
        self.assertEqual(len(counted), 1)
        self.assertEqual(characters, 3)

    def test_two_words_boxed_as_one_in_another_reading_count_once(self):
        both = boxed("OPENTODAY", 400, 400, 200, 20, confidence=75.0)
        first, second = boxed("OPEN", 400, 400, 90, 20), boxed("TODAY", 500, 400, 100, 20, confidence=88.0)
        _, characters = lettering.lettering_of([both, first, second], 1024)
        self.assertEqual(characters, 9)

    def test_two_words_apart_are_both_kept(self):
        self.assertEqual(len(lettering.once([boxed("OPEN", 100, 100, 90, 20), boxed("OPEN", 600, 600, 90, 20)])), 2)

    def test_a_likelier_scrap_does_not_hide_a_word_that_counts(self):
        scrap, real = boxed("|||", 400, 400, 60, 20, confidence=95.0), boxed("WORDS", 400, 400, 60, 20, 75.0)
        # Kept once before counting, the scrap would win the box and the word be lost.
        self.assertEqual([w.text for w in lettering.once([scrap, real])], ["|||"])
        counted, characters = lettering.lettering_of([scrap, real], 1024)
        self.assertEqual([w.text for w in counted], ["WORDS"])
        self.assertEqual(characters, 5)


def fake_tesseract(version=(0, "tesseract 5.5.0\n leptonica-1.86.0\n", ""),
                   languages=(0, 'List of available languages in "/usr/share/tessdata/" (2):\neng\nosd\n', ""),
                   reading=(0, TSV_HEADER, "")):
    """subprocess.run as a tesseract would answer: each answer (exit code, output, errors) or an exception; the calls
    made are kept in `.calls`."""
    calls = []

    def run(arguments, **_):
        calls.append(arguments)
        answer = {"--version": version, "--list-langs": languages}.get(arguments[1], reading)
        if isinstance(answer, BaseException):
            raise answer
        code, out, errors = answer
        return subprocess.CompletedProcess(arguments, code, out.encode(), errors.encode())

    run.calls = calls
    return run


class TheReader(unittest.TestCase):
    """The probe (version 5 or newer, its English data), once; a reader that cannot be used, or a reading that fails,
    is "not judged" with its reason: never a crash, never a pass (the 0.2.0 review, defects 2 and 3, missing test 1)."""

    def setUp(self):
        lettering.forget_reader()
        self.addCleanup(lettering.forget_reader)

    def judged_with(self, run, required=False):
        with mock.patch.object(lettering.shutil, "which", return_value="/usr/bin/tesseract"), \
                mock.patch.object(lettering.subprocess, "run", run):
            return lettering.reader(), lettering.lettering_check(*slab_with_text(""), required=required)

    def assert_not_judged(self, check, *said):
        self.assertFalse(check.passed, check.words)
        self.assertFalse(check.counted, check.words)
        self.assertTrue(check.words.startswith(lettering.NOT_JUDGED), check.words)
        self.assertFalse(lettering.was_read(check))
        for words in said:
            self.assertIn(words, check.words)

    def test_a_usable_reader_reads_and_is_probed_once(self):
        run = fake_tesseract()
        found, check = self.judged_with(run)
        with mock.patch.object(lettering.shutil, "which", return_value="/usr/bin/tesseract"), \
                mock.patch.object(lettering.subprocess, "run", run):
            lettering.lettering_check(*slab_with_text(""))
        self.assertEqual(found, USABLE)
        self.assertTrue(check.passed and check.counted and lettering.was_read(check), check.words)
        self.assertEqual(sum(call[1] == "--version" for call in run.calls), 1)
        self.assertEqual(sum(call[1] == "stdin" for call in run.calls), 2 * len(lettering.READINGS))

    def test_tesseract_4_is_not_used(self):
        run = fake_tesseract(version=(0, "tesseract 4.1.1\n leptonica-1.82.0\n", ""))
        found, check = self.judged_with(run)
        self.assertEqual((found.usable, found.version), (False, "4.1.1"))
        self.assert_not_judged(check, "4.1.1", "version 5 or newer")
        self.assertFalse(any(call[1] == "stdin" for call in run.calls), "no reading is made")

    def test_a_version_printed_on_the_errors_is_found(self):
        found, check = self.judged_with(fake_tesseract(version=(1, "", "tesseract 3.05.01\n")))
        self.assertEqual(found.version, None)
        self.assert_not_judged(check, "no version")
        lettering.forget_reader()
        found, _ = self.judged_with(fake_tesseract(version=(0, "", "tesseract 3.05.01\n")))
        self.assertEqual((found.usable, found.version), (False, "3.05.01"))

    def test_a_version_that_says_nothing_is_not_used(self):
        found, check = self.judged_with(fake_tesseract(version=(0, "", "")))
        self.assertFalse(found.usable)
        self.assert_not_judged(check, "no version")

    def test_without_english_data_it_is_not_used(self):
        found, check = self.judged_with(fake_tesseract(languages=(0, "List of available languages (1):\nosd\n", "")))
        self.assertEqual((found.usable, found.version), (False, "5.5.0"))
        self.assert_not_judged(check, "no English data")

    def test_a_probe_that_fails_or_hangs_is_not_used(self):
        for broken in (subprocess.TimeoutExpired("tesseract", lettering.PROBE_SECONDS), OSError("Exec format error")):
            lettering.forget_reader()
            found, check = self.judged_with(fake_tesseract(version=broken))
            self.assertFalse(found.usable)
            self.assert_not_judged(check, "does not run")
            lettering.forget_reader()
            found, check = self.judged_with(fake_tesseract(languages=broken))
            self.assertFalse(found.usable)
            self.assert_not_judged(check, "does not list its languages")

    def test_a_reading_that_fails_is_not_judged(self):
        _, check = self.judged_with(fake_tesseract(reading=(1, "", "Error opening data file eng.traineddata\n")))
        self.assert_not_judged(check, "exited 1", "eng.traineddata")

    def test_a_reading_that_times_out_is_not_judged(self):
        _, check = self.judged_with(fake_tesseract(reading=subprocess.TimeoutExpired("tesseract",
                                                                                     lettering.READ_SECONDS)))
        self.assert_not_judged(check, "TimeoutExpired")

    def test_an_option_not_taken_is_not_judged(self):
        _, check = self.judged_with(fake_tesseract(reading=(0, TSV_HEADER, "Could not set option: "
                                                                           "thresholding_method=2\n")))
        self.assert_not_judged(check, "thresholding_method")

    def test_a_run_that_requires_the_reader_counts_a_try_not_read_as_failed(self):
        _, check = self.judged_with(fake_tesseract(reading=(1, "", "Error\n")), required=True)
        self.assertFalse(check.passed)
        self.assertTrue(check.counted)
        result = picture_checks.PictureResult(checks=[check])
        self.assertFalse(result.passed)

    def test_an_empty_cut_out_is_not_a_counted_pass(self):
        picture, mask = slab_with_text("")
        with mock.patch.object(lettering, "reader", return_value=USABLE):
            check = lettering.lettering_check(picture, np.zeros_like(mask))
        self.assert_not_judged(check, "no cut-out")


class TheTSV(unittest.TestCase):
    """`read()` on tesseract's TSV as it comes (the 0.2.0 review, missing test 2)."""

    def read(self, tsv, errors=""):
        run = fake_tesseract(reading=(0, tsv, errors))
        with mock.patch.object(lettering.subprocess, "run", run):
            return lettering.read(Image.new("L", (40, 20), 255))

    def test_a_header_alone_is_no_word(self):
        self.assertEqual(self.read(TSV_HEADER), [])
        self.assertEqual(self.read(""), [])

    def test_words_are_read_and_the_rest_passed_over(self):
        tsv = TSV_HEADER + "\n".join([
            "1\t1\t0\t0\t0\t0\t0\t0\t100\t50\t-1\t",
            "5\t1\t1\t1\t1\t1\t10\t12\t30\t14\t91.5\tOPEN",
            "5\t1\t1\t1\t1\t2\t50\t12\t30\t14\t40.0\t   ",
            "5\t1\t1\t1\t1\t3\t50\t12\t30",
            "5\t1\t1\t1\t1\t4\tx\t12\t30\t14\t60.0\tBAD",
            "5\t1\t1\t1\t1\t5\t90\t12\t30\t14\t-1\tLOW"])
        self.assertEqual(self.read(tsv), [("OPEN", 91.5, 10, 12, 30, 14), ("LOW", -1.0, 90, 12, 30, 14)])

    def test_a_word_read_at_confidence_minus_one_does_not_count(self):
        self.assertFalse(lettering.counts(word("LOWWORD", confidence=-1.0), 1024))

    def test_what_tesseract_says_on_its_errors_does_not_stop_a_reading(self):
        tsv = TSV_HEADER + "5\t1\t1\t1\t1\t1\t10\t12\t30\t14\t91.5\tOPEN\n"
        self.assertEqual(len(self.read(tsv, "Estimating resolution as 312\nWarning: Invalid resolution 0 dpi\n")), 1)


class TheCheck(unittest.TestCase):
    def setUp(self):
        lettering.forget_reader()
        self.addCleanup(lettering.forget_reader)

    def test_without_tesseract_it_is_not_judged_and_never_passes(self):
        picture, mask = slab_with_text("")
        with mock.patch.object(lettering, "TESSERACT", "no-such-tesseract-here"):
            check = lettering.lettering_check(picture, mask)
        self.assertFalse(check.passed)
        self.assertFalse(check.counted)
        self.assertIn(lettering.NOT_INSTALLED, check.words)

    def test_a_try_shown_unjudged_still_passes_its_counted_checks(self):
        result = picture_checks.PictureResult()
        with mock.patch.object(lettering, "TESSERACT", "no-such-tesseract-here"):
            lettering.add_lettering_check(result, *slab_with_text(""))
        self.assertTrue(result.passed)

    @NEEDS_TESSERACT
    def test_lettering_light_on_dark_iron_fails(self):
        check = lettering.lettering_check(*slab_with_text("NUISNHIFTS EANITIES"))
        self.assertFalse(check.passed, check.words)
        self.assertTrue(check.counted)

    @NEEDS_TESSERACT
    def test_lettering_dark_on_light_wood_fails(self):
        check = lettering.lettering_check(*slab_with_text("BLHOM BOORY", ink=(60, 40, 20), slab=(190, 150, 100)))
        self.assertFalse(check.passed, check.words)

    @NEEDS_TESSERACT
    def test_a_plain_slab_passes(self):
        check = lettering.lettering_check(*slab_with_text(""))
        self.assertTrue(check.passed, check.words)

    @NEEDS_TESSERACT
    def test_lettering_the_words_ask_for_passes_and_is_listed(self):
        check = lettering.lettering_check(*slab_with_text("OPEN TODAY"), asked="say")
        self.assertTrue(check.passed)
        self.assertIn("OPEN", check.words)

    @NEEDS_TESSERACT
    def test_a_try_with_lettering_is_not_chosen(self):
        lettered, plain = picture_checks.PictureResult(score=0.9), picture_checks.PictureResult(score=0.5)
        lettering.add_lettering_check(lettered, *slab_with_text("NUISNHIFTS EANITIES"))
        lettering.add_lettering_check(plain, *slab_with_text(""))
        self.assertEqual(picture_checks.best_try([lettered, plain]), 1)


class TheWordsAsk(unittest.TestCase):
    def test_words_that_ask_for_lettering(self):
        for prompt, asked in (("a wooden sign that says OPEN", "say"), ("a book with a title on its cover", "title"),
                              ('a sign that reads "inn"', "read"), ("a gravestone with an inscription", "inscription"),
                              ("a scroll with writing on it", "writing on"), ("a brass plaque engraved with a name",
                                                                            "engraved")):
            self.assertEqual(words.lettering_asked(prompt), asked, prompt)

    def test_common_ways_of_asking_are_read(self):
        # The 0.2.0 review, defect 6: each was painted with "no text" and every try with the letters asked refused.
        for prompt, asked in (("a sign with OPEN painted on it", "OPEN"), ("a mug printed with HELLO", "printed"),
                              ("a crate marked FRAGILE", "marked"), ("a wooden sign reading OPEN", "reading"),
                              ("a book called Dune", "called"), ('a sign "OPEN"', "OPEN"),
                              ("a sign with the letter A", "letter"), ("a sword with a rune", "rune"),
                              ("a rune stone", "rune"), ("a tombstone with RIP", "RIP"),
                              ("a tablet with hieroglyphs", "hieroglyph"), ("a sign 'inn'", "inn"),
                              ("a door with a number", "number"), ("a sword with letters along its blade", "letter")):
            self.assertEqual(words.lettering_asked(prompt), asked, prompt)
            # As the pipeline reads it (the prompt as read, `cleaned`), it still asks.
            self.assertTrue(words.lettering_asked(words.cleaned(prompt)), prompt)

    def test_a_denial_reaches_only_its_own_words(self):
        # The 0.2.0 review, defect 5: a negation of something else does not deny the lettering asked.
        for prompt, asked in (("a box with no lid, labelled FRAGILE", "labelled"), ("a no smoking label", "label"),
                              ("a wooden sign without paint that says inn", "say"),
                              ("a crate with no paint, labelled", "labelled")):
            self.assertEqual(words.lettering_asked(prompt), asked, prompt)
        for prompt in ("a barrel not yet labelled", "a sign that does not say anything", "a sign free of text",
                       "a sign with no visible text", "a sign with no text, logo or signature",
                       "a sign without words or logos", "a sword, no text"):
            self.assertEqual(words.lettering_asked(prompt), "", prompt)

    def test_words_that_do_not_ask_for_lettering_where_they_stand(self):
        # The 0.2.0 review, defect 7: each let any letters through.
        for prompt in ("an engraved silver cup", "a pile of letters", "a number of coins", "a text-free sign",
                       "a logo-free mug", "a sign with neither text nor logo", "a well-read book", "a title deed",
                       "a letter opener", "a comfortable reading chair", "a lamp for reading", "a TV stand",
                       "AN IRON ANVIL", "a smith's hammer"):
            self.assertEqual(words.lettering_asked(prompt), "", prompt)
            self.assertIn("no text", words.picture_prompt(words.subject_of(prompt)), prompt)

    def test_the_picture_words_follow_the_asking(self):
        for prompt in ("a box with no lid, labelled FRAGILE", "a tombstone with RIP", "a crate marked FRAGILE"):
            self.assertNotIn("no text", words.picture_prompt(words.subject_of(prompt)), prompt)

    def test_things_that_often_bear_letters_do_not_ask(self):
        for prompt in ("an iron anvil", "a brass plaque", "a wooden sign", "a gold coin", "a wooden crate",
                       "a writing desk", "a reading lamp", "a brand new sword", "a sword", "a breadboard",
                       "a well designed chair"):
            self.assertEqual(words.lettering_asked(prompt), "", prompt)

    def test_lettering_denied_is_not_asked(self):
        for prompt in ("a sign with no text", "a crate without any lettering", "a book with no title"):
            self.assertEqual(words.lettering_asked(prompt), "", prompt)

    def test_the_picture_words_drop_no_text_only_when_lettering_is_asked(self):
        self.assertIn("no text", words.picture_prompt(words.subject_of("an iron anvil")))
        asked = words.picture_prompt(words.subject_of("a wooden sign that says OPEN"))
        self.assertNotIn("no text", asked)
        self.assertNotIn("no signature", asked)


class InThePipeline(unittest.TestCase):
    """The picture stage with the lettering check, the reader faked (the 0.2.0 review, missing test 4): a lettered try is
    not chosen; all lettered, the job fails on it; the record and the ledger say whether the chosen picture was read."""

    class Painter:
        def paint(self, prompt, seed):
            picture = np.full((256, 256, 3), 214, np.uint8)
            picture[40:220, 110:146] = 90
            return Image.fromarray(picture)

    class Cutter:
        def mask(self, picture):
            return np.asarray(picture)[..., 0] < 150

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder)
        ledger_folder = os.path.join(self.folder, "ledger")
        patcher = mock.patch.dict(os.environ, {"MODEL_MAKER_LEDGER": ledger_folder})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ledger = os.path.join(ledger_folder, "jobs.jsonl")

    def job(self, prompt="an iron anvil"):
        from model_maker import pipeline

        return pipeline.Job(os.path.join(self.folder, "job"), prompt)

    def picked(self, job, lettered_seeds, reader=USABLE, tries=2, required=False):
        """The picture stage of `job` with `tries` tries a round, the tries of `lettered_seeds` read as lettered."""
        from model_maker import painter, pipeline

        settings = pipeline.Settings(verify=False, tries=tries, require_lettering=required)
        first = painter.seed_of(words.picture_prompt(job.subject, settings.lead))
        painted = iter(range(first, first + 100))

        def every_reading(rgb, mask, readings=lettering.READINGS):
            return [boxed("EANITIES", 100, 60, 40, 12), boxed("NUISNHIETS", 100, 80, 40, 12)] \
                if next(painted) - first in lettered_seeds else []

        found = {}
        with mock.patch.object(lettering, "reader", return_value=reader), \
                mock.patch.object(lettering, "every_reading", side_effect=every_reading):
            pipeline.picture_job(job, settings, self.Painter(), self.Cutter(), None, found, report=lambda _: None)
        return job.record["picture"], first, found, settings

    def test_a_lettered_try_is_not_chosen_and_the_record_says_it_was_read(self):
        record, first, found, _ = self.picked(self.job(), lettered_seeds={0})
        self.assertEqual(record["seed"], first + 1)
        self.assertTrue(record["passed"])
        self.assertTrue(record["lettering_judged"])
        self.assertEqual(record["lettering_asked"], "")
        refused = next(c for c in record["tries"][str(first)]["checks"] if c["name"] == lettering.NAME)
        self.assertFalse(refused["passed"])
        self.assertIn("EANITIES", refused["words"])
        self.assertEqual(found["faults"], [lettering.NAME])
        self.assertEqual(self.job().record["lettering"], {"judged": True, "tesseract": "5.5.0", "read_at": "pictures",
                                                          "why": ""})

    def test_every_try_lettered_ends_the_job_failed_on_it(self):
        from model_maker import pipeline

        job = self.job()
        record, _, found, settings = self.picked(job, lettered_seeds=set(range(100)))
        self.assertFalse(record["passed"])
        self.assertEqual(found["rounds"], settings.picture_rounds)
        pipeline.no_picture_passed(job, settings, report=lambda _: None)
        self.assertEqual(job.record["status"], f"fails: {lettering.NAME}")
        with open(self.ledger) as file:
            line = [json.loads(text) for text in file][-1]
        self.assertFalse(line["passed"])
        self.assertEqual(line["failed"], [lettering.NAME])
        self.assertIs(line["lettering_judged"], True)

    def test_without_a_reader_the_tries_pass_unread_and_the_record_and_ledger_say_so(self):
        from model_maker import pipeline

        job = self.job()
        missing = lettering.Reader(False, None, lettering.NOT_INSTALLED)
        record, first, found, settings = self.picked(job, lettered_seeds={0}, reader=missing)
        self.assertEqual(record["seed"], first)
        self.assertTrue(record["passed"])
        self.assertFalse(record["lettering_judged"])
        self.assertFalse(job.record["lettering"]["judged"])
        self.assertIn(lettering.NOT_INSTALLED, job.record["lettering"]["why"])
        # A check shown, not counted, is no fault (the 0.2.0 review, defect 8).
        self.assertEqual(found["faults"], [])
        pipeline.no_picture_passed(job, settings, report=lambda _: None)
        with open(self.ledger) as file:
            line = [json.loads(text) for text in file][-1]
        self.assertIs(line["lettering_judged"], False)

    def test_a_run_that_requires_the_reader_makes_no_job_without_it(self):
        from model_maker import pipeline

        job = self.job()
        settings = pipeline.Settings(models=os.path.join(self.folder, "no-models"), verify=False,
                                     require_lettering=True)
        missing = lettering.Reader(False, "4.1.1", "not judged (tesseract 4.1.1: version 5 or newer is needed)")
        with mock.patch.object(lettering, "reader", return_value=missing):
            pipeline.run([job], settings, report=lambda _: None, stages=("pictures",))
        self.assertTrue(job.record["status"].startswith(f"not made: {pipeline.NEEDS_LETTERING_READER}"),
                        job.record["status"])
        self.assertIn("4.1.1", job.record["status"])
        self.assertFalse(job.has("picture.png"))

    def test_a_run_that_requires_the_reader_does_not_choose_a_try_it_could_not_read(self):
        from model_maker import painter, pipeline

        job = self.job()
        settings = pipeline.Settings(verify=False, tries=2, require_lettering=True)
        unread = iter([True, False, False, False])

        def check(rgb, mask, asked="", required=False):
            if next(unread):
                return lettering.not_judged("not judged (tesseract 5.5.0 failed: ReadingFailed: exit 1)", required)
            return picture_checks.Check(lettering.NAME, 0.0, lettering.MOST_CHARACTERS, "characters", True,
                                        "no lettering read")

        first = painter.seed_of(words.picture_prompt(job.subject))
        with mock.patch.object(lettering, "reader", return_value=USABLE), \
                mock.patch.object(lettering, "lettering_check", side_effect=check):
            pipeline.picture_job(job, settings, self.Painter(), self.Cutter(), None, {}, report=lambda _: None)
        self.assertEqual(job.record["picture"]["seed"], first + 1)
        self.assertTrue(job.record["picture"]["lettering_judged"])

    def test_the_settings_keep_whether_the_reader_was_required(self):
        from model_maker import ledger, pipeline

        self.assertIs(ledger.settings_of(pipeline.Settings(verify=False, require_lettering=True))["require_lettering"],
                      True)
        self.assertIs(ledger.settings_of(pipeline.Settings(verify=False))["require_lettering"], False)


if __name__ == "__main__":
    unittest.main()
