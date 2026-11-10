import contextlib
import io
import os
import shutil
import tempfile
import tomllib
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from model_maker import __main__ as command, inspect_mesh, object_checks as O, painter, registry, render, subject_judge, words

from . import shapes

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_CASES = os.path.join(HERE, "..", "..", "tools", "trial", "prompt-cases.toml")
HEAD_CASES = os.path.join(HERE, "..", "..", "tools", "trial", "head-cases.toml")


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


class TheShare(unittest.TestCase):
    def test_the_thing_asked_for_is_judged_against_the_likeliest_other(self):
        picture = unit([1.0, 0.3, 0.0])
        asked = unit([1.0, 0.0, 0.0])
        others = np.array([unit([0.0, 0.0, 1.0]), unit([0.0, 1.0, 0.0])])
        share, likeliest = subject_judge.subject_share(picture, asked, others, scale=100.0)
        self.assertGreater(share, 0.99)
        self.assertEqual(likeliest, 1)
        # Asked for the second other thing, the picture's thing is the likeliest other: it fails.
        share, likeliest = subject_judge.subject_share(picture, others[1], np.array([asked, others[0]]), scale=100.0)
        self.assertLess(share, 0.01)
        self.assertEqual(likeliest, 0)

    def test_the_check_says_what_came_nearest(self):
        chair = words.subject_of("an old looking chair")
        check = subject_judge.subject_check(chair, 0.4, "a table")
        self.assertFalse(check.passed)
        self.assertIn("a table", check.words)
        self.assertTrue(subject_judge.subject_check(chair, 0.99, "a stool").passed)


class TheContrast(unittest.TestCase):
    def test_a_chair_is_judged_against_a_table_a_vase_a_person_text_and_an_empty_backdrop(self):
        asked, others = words.subject_contrast(words.subject_of("an old looking chair"))
        self.assertEqual(asked[0], "a photo of an old looking chair")
        self.assertIn("a seat with a back rest", asked[1])
        for other in ("a table", "a stool", "a jug", "a person", "an animal", "text", "an empty backdrop",
                      "several things"):
            self.assertIn(other, others)
        self.assertNotIn("a chair", others)

    def test_a_kind_the_prompt_names_is_never_its_contrast(self):
        # "a candle in a brass holder" names a candle; "a chest with a sword carved on it" a chest and a sword.
        _, others = words.subject_contrast(words.subject_of("a chest with a sword carved on it"))
        self.assertNotIn("a chest", others)
        self.assertNotIn("a sword", others)
        self.assertIn("a table", others)

    def test_a_likeness_of_a_creature_is_not_judged_against_a_person_or_an_animal(self):
        _, others = words.subject_contrast(words.subject_of("a stone statue of a knight"))
        self.assertNotIn("a person", others)
        self.assertNotIn("an animal", others)
        self.assertIn("text", others)

    def test_a_prompt_that_asks_for_letters_is_not_judged_against_text(self):
        _, others = words.subject_contrast(words.subject_of("a wooden signpost with carved letters"))
        self.assertNotIn("text", others)

    def test_a_thing_of_no_known_word_is_the_prompt_alone_against_every_kind(self):
        asked, others = words.subject_contrast(words.subject_of("a brass astrolabe"))
        self.assertEqual(asked, ["a photo of a brass astrolabe"])
        self.assertEqual(len(others), len(words.SUBJECT_KINDS) + len(words.SUBJECT_DISTRACTORS))

    def test_each_kind_is_one_row_and_its_description_names_it_first(self):
        seen = {}
        for nouns, description in words.SUBJECT_KINDS:
            for noun in nouns:
                self.assertNotIn(noun, seen, f"{noun} is in two rows")
                seen[noun] = description
            name, look = description.split(", ", 1)
            self.assertIn(name.split(" ", 1)[1], nouns, description)
            self.assertTrue(look)


class TheLabelledSet(unittest.TestCase):
    def test_the_labelled_pictures_are_well_formed(self):
        with open(PROMPT_CASES, "rb") as file:
            pictures = tomllib.load(file)["picture"]
        names = [p["name"] for p in pictures]
        self.assertEqual(len(names), len(set(names)))
        for picture in pictures:
            self.assertEqual(set(picture) - {"name", "source", "mask", "prompt", "shows", "why"}, set(), picture)
            self.assertFalse(words.subject_of(picture["prompt"]).not_yet, picture)
            if "shows" in picture:
                self.assertEqual(picture["shows"], "unclear")
                self.assertTrue(picture["why"])
        self.assertGreaterEqual(len({p["prompt"] for p in pictures}), 15)

    def test_the_labelled_heads_are_well_formed_and_cover_each_wording(self):
        with open(HEAD_CASES, "rb") as file:
            pictures = tomllib.load(file)["picture"]
        names = [p["name"] for p in pictures]
        self.assertEqual(len(names), len(set(names)))
        family = set(words.HEAD_KINDS)
        for picture in pictures:
            self.assertEqual(set(picture), {"name", "source", "prompt", "wording", "head", "why"}, picture)
            self.assertIn(picture["head"], ("hammer", "axe", "hybrid", "other"), picture)
            self.assertIn(picture["wording"], ("trial", "old", "new", "turn"), picture)
            self.assertTrue(picture["why"])
            noun = words.subject_of(picture["prompt"]).noun
            self.assertTrue(noun in family or noun in ("battle axe", "hatchet"), picture)
        # At least eight tries a prompt for each wording measured (docs/objects.md, "War hammers").
        counts = {}
        for picture in pictures:
            key = (picture["prompt"], picture["wording"])
            counts[key] = counts.get(key, 0) + 1
        for prompt in ("a war hammer", "a maul", "a mallet", "a sledgehammer", "a battle axe",
                       "a double-bitted battle axe", "a hatchet"):
            for wording in ("old", "new"):
                self.assertGreaterEqual(counts.get((prompt, wording), 0), 8, (prompt, wording))


def write_chair(folder, gap_filled=False):
    """A chair's mesh, a picture of it (its silhouette on a grey backdrop) and its true cut-out; with `gap_filled`, a
    forged cut-out whose gap under the seat is filled, as if the model had a web there."""
    import trimesh

    corners, triangles = shapes.chair()
    corners = O.set_down(corners, 0.95)
    trimesh.Trimesh(corners, triangles, process=False).export(os.path.join(folder, "model.glb"))
    mask = render.silhouette(corners, triangles, 0, 384)
    picture = np.where(mask[..., None], 110, 205).astype(np.uint8).repeat(3, -1)
    Image.fromarray(picture).save(os.path.join(folder, "picture.png"))
    sent = mask.copy()
    if gap_filled:
        rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        low = rows[0] + (rows[-1] - rows[0]) * 3 // 4
        sent[low:rows[-1], cols[0]:cols[-1]] = True
    Image.fromarray(sent.astype(np.uint8) * 255).save(os.path.join(folder, "mask.png"))
    return mask


class OurOwnCutOut(unittest.TestCase):
    def inspect(self, folder, mask):
        with mock.patch.object(inspect_mesh, "our_cut", return_value=mask) as cut:
            command.main(["inspect", os.path.join(folder, "model.glb"), "--picture", os.path.join(folder, "picture.png"),
                          "--mask", os.path.join(folder, "mask.png"), "--recut", "--prompt=an old looking chair",
                          "--out", os.path.join(folder, "out")])
            self.assertEqual(cut.call_count, 1)
        import json

        with open(os.path.join(folder, "out", "checks.json")) as file:
            found = json.load(file)
        return found, {c["name"]: c for c in found["checks"]["picture"]}

    def test_a_cut_out_that_is_the_pictures_agrees_with_ours(self):
        folder = tempfile.mkdtemp()
        try:
            mask = write_chair(folder)
            with contextlib.redirect_stdout(io.StringIO()):
                found, picture = self.inspect(folder, mask)
            agreement = picture[inspect_mesh.CUTOUT_AGREEMENT]
            self.assertTrue(agreement["passed"], agreement)
            self.assertEqual(agreement["value"], 100.0)
            self.assertNotIn(inspect_mesh.CUTOUT_AGREEMENT, found["status"])
        finally:
            shutil.rmtree(folder)

    def test_a_forged_cut_out_is_caught_and_the_checks_use_ours(self):
        folder = tempfile.mkdtemp()
        try:
            mask = write_chair(folder, gap_filled=True)
            with contextlib.redirect_stdout(io.StringIO()):
                found, picture = self.inspect(folder, mask)
            agreement = picture[inspect_mesh.CUTOUT_AGREEMENT]
            self.assertFalse(agreement["passed"], agreement)
            self.assertIn(inspect_mesh.CUTOUT_AGREEMENT, found["status"])
            # The picture's checks were judged on our cut-out: the size it covers is ours, not the forged one's.
            ours = picture["object's size in the picture"]["value"]
            self.assertAlmostEqual(ours, mask.mean() * 100, places=2)
        finally:
            shutil.rmtree(folder)

    def test_a_cut_out_of_another_size_is_caught(self):
        check = inspect_mesh.agreement_check(np.ones((10, 12), bool), np.ones((12, 10), bool))
        self.assertFalse(check.passed)


class ModelsNotHere(unittest.TestCase):
    def test_a_command_whose_model_is_not_downloaded_stops_with_its_own_code(self):
        folder = tempfile.mkdtemp()
        try:
            Image.new("RGB", (64, 64), (200, 200, 200)).save(os.path.join(folder, "picture.png"))
            with mock.patch.dict(os.environ, {"MODEL_MAKER_MODELS": folder}), \
                    contextlib.redirect_stderr(io.StringIO()) as said:
                code = command.main(["picture-check", os.path.join(folder, "picture.png"),
                                     "--prompt=an old looking chair", "--out", os.path.join(folder, "out")])
            self.assertEqual(code, registry.MODEL_MISSING_EXIT)
            self.assertIn("not here", said.getvalue())
        finally:
            shutil.rmtree(folder)


class TheModel(unittest.TestCase):
    def test_siglip_2_judges_a_picture_against_a_prompt_on_the_cpu(self):
        folder = os.path.join(os.environ.get("MODEL_MAKER_MODELS", "models"), "siglip2-so400m-patch14-384")
        if not os.path.exists(os.path.join(folder, "model.safetensors")):
            self.skipTest("SigLIP 2 is not downloaded")
        out = tempfile.mkdtemp()
        try:
            Image.new("RGB", (384, 384), (220, 220, 220)).save(os.path.join(out, "empty.png"))
            with contextlib.redirect_stdout(io.StringIO()):
                record = subject_judge.picture_check(os.path.join(out, "empty.png"), "an old looking chair",
                                                     os.path.join(out, "out"))
            self.assertTrue(record["status"].startswith("fails: "), record)
            self.assertEqual(record["likeliest_other"], "an empty backdrop")
        finally:
            shutil.rmtree(out)


if __name__ == "__main__":
    unittest.main()


class InTheChoiceOfTry(unittest.TestCase):
    """The pipeline's picture stage judges each try against its prompt, and a try the check refuses is not chosen
    (without it, the trials chose a war hammer with an axe-like head, 0.49, over a real one, 0.97)."""

    class Painter:
        def paint(self, prompt, seed):
            picture = np.full((256, 256, 3), 214, np.uint8)
            picture[40:220, 110:146] = 90
            return Image.fromarray(picture)

    class Cutter:
        def mask(self, picture):
            return np.asarray(picture)[..., 0] < 150

    def chosen(self, shares, judge=object()):
        from model_maker import pipeline

        folder = tempfile.mkdtemp()
        try:
            job = pipeline.Job(os.path.join(folder, "war-hammer"), "a war hammer")
            settings = pipeline.Settings(verify=False, tries=len(shares))
            first = painter.seed_of(words.picture_prompt(job.subject))
            given = iter(shares)

            def judged(judge, picture, mask, directions):
                return next(given), "an axe"

            with mock.patch.object(subject_judge, "contrast_directions", return_value="directions"), \
                    mock.patch.object(subject_judge, "judged", side_effect=judged):
                pipeline.picture_job(job, settings, self.Painter(), self.Cutter(), judge, {}, report=lambda _: None)
            return job.record["picture"], first
        finally:
            shutil.rmtree(folder)

    def test_an_axe_headed_try_is_not_chosen_and_the_record_keeps_the_verdict(self):
        record, first = self.chosen([0.58, 0.49, 0.97, 0.88])
        # Every try scores the same on the other checks (the same picture): without the check the first is chosen.
        self.assertEqual(record["seed"], first + 2)
        self.assertTrue(record["passed"])
        self.assertTrue(record["subject_judged"])
        refused = record["tries"][str(first)]
        self.assertFalse(refused["passed"])
        self.assertIn(subject_judge.NAME, [c["name"] for c in refused["checks"] if not c["passed"]])

    def test_without_siglip_2_the_tries_go_unjudged_and_the_record_says_so(self):
        from model_maker import pipeline

        folder = tempfile.mkdtemp()
        try:
            job = pipeline.Job(os.path.join(folder, "war-hammer"), "a war hammer")
            pipeline.picture_job(job, pipeline.Settings(verify=False, tries=2), self.Painter(), self.Cutter(), None,
                                 {}, report=lambda _: None)
            record = job.record["picture"]
            self.assertFalse(record["subject_judged"])
            self.assertNotIn(subject_judge.NAME, [c["name"] for t in record["tries"].values() for c in t["checks"]])
        finally:
            shutil.rmtree(folder)
