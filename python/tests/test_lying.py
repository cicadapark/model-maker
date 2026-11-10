import json
import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from model_maker import lying, object_checks as O, pipeline, render, words

from . import shapes


def book(tilt_deg=0.0, cell=0.006):
    """A closed book 25 x 18 cm and 4 cm thick, its cover facing the viewer (+Z), standing on its bottom edge, then
    leaned `tilt_deg` back about X, its cover turned up toward the viewer (as the fix-check's book came out of the
    shape model), and set on y = 0."""
    corners, triangles = shapes.surface_of([((-0.09, 0.0, -0.02), (0.09, 0.25, 0.02))], cell)
    t = np.radians(tilt_deg)
    turn = np.array([[1, 0, 0], [0, np.cos(t), np.sin(t)], [0, -np.sin(t), np.cos(t)]])
    corners = corners @ turn.T
    return O.set_down(corners, float(np.ptp(corners, axis=0).max())), triangles


def base_checks(corners, triangles):
    return {c.name: c.passed for c in O.base_checks(O.base_measures(corners, triangles))}


class StablePoses(unittest.TestCase):
    def test_a_book_rests_on_its_covers_far_more_often_than_on_its_edges(self):
        corners, triangles = book()
        poses = lying.stable_poses(corners, triangles)
        self.assertAlmostEqual(sum(p["share"] for p in poses), 1.0, places=6)
        covers = [p for p in poses if abs(p["down"][2]) > 0.95]
        self.assertEqual(len(covers), 2, poses)
        self.assertEqual({id(p) for p in poses[:2]}, {id(p) for p in covers})
        self.assertGreater(min(p["share"] for p in covers), 0.3)
        # Resting on a cover, its centre is half the thickness above the floor.
        self.assertAlmostEqual(covers[0]["height_m"], 0.02, delta=0.004)

    def test_the_turn_takes_the_resting_direction_straight_down(self):
        for down in ([0.0, 0.0, 1.0], [0.6, 0.0, -0.8], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]):
            rotation = lying.turning(np.array(down))
            self.assertTrue(np.allclose(rotation @ np.array(down), [0.0, -1.0, 0.0], atol=1e-9), down)
            self.assertTrue(np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9))
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=9)


class Laid(unittest.TestCase):
    def test_a_book_made_standing_tilted_is_laid_on_its_cover_and_then_stands(self):
        corners, triangles = book(tilt_deg=35.0)
        self.assertFalse(base_checks(corners, triangles)["base level"], "it stands on an edge as made")
        subject = words.subject_of("a leather-bound book, closed")
        laid_corners, laid_triangles, record = lying.laid(subject, corners, triangles)
        self.assertTrue(record["laid"])
        self.assertTrue(all(base_checks(laid_corners, laid_triangles).values()),
                        base_checks(laid_corners, laid_triangles))
        self.assertAlmostEqual(float(np.ptp(laid_corners[:, 1])), 0.04, delta=0.008)
        self.assertAlmostEqual(float(laid_corners[:, 1].min()), 0.0, places=9)
        # It falls back onto its back cover, so the front one, which faced the viewer, faces up: a turn of 90 - 35
        # degrees (the fix-check's book settled 54.8 degrees).
        self.assertAlmostEqual(record["turned_deg"], 90 - 35, delta=4)
        # The made frame and the laid one map onto each other.
        self.assertTrue(np.allclose(lying.to_made(laid_corners, record), corners, atol=1e-9))
        self.assertTrue(np.allclose(lying.from_made(corners, record), laid_corners, atol=1e-9))

    def test_a_book_made_lying_flat_is_left_flat(self):
        corners, triangles = book(tilt_deg=-90.0)
        _, _, record = lying.laid(words.subject_of("an old book"), corners, triangles)
        self.assertLess(min(record["turned_deg"], 180 - record["turned_deg"]), 1.0, record)

    def test_a_chair_that_tips_is_never_laid_down_and_still_fails(self):
        corners, triangles = shapes.chair(lean=0.3)
        subject = words.subject_of("an old looking chair")
        self.assertFalse(subject.lies)
        laid_corners, _, record = lying.laid(subject, corners, triangles)
        self.assertFalse(record["laid"])
        self.assertIs(laid_corners, corners)
        self.assertFalse(base_checks(corners, triangles)["centre over the base"])


class Words(unittest.TestCase):
    def test_which_things_lie(self):
        for prompt in ("a leather-bound book, closed", "an old gold coin", "a pewter plate", "a wooden tray"):
            subject = words.subject_of(prompt)
            self.assertTrue(subject.lies, prompt)
            self.assertTrue(subject.stands, prompt)
            self.assertIn("lies", subject.why)
        for prompt in ("an old looking chair", "a small round table", "a wooden treasure chest", "a clay jug",
                       "a candle in a brass holder", "an old iron sword"):
            self.assertFalse(words.subject_of(prompt).lies, prompt)

    def test_a_book_asked_upright_is_left_as_made(self):
        subject = words.subject_of("a leather-bound book standing upright on its spine")
        self.assertFalse(subject.lies)
        self.assertIn("left as made", subject.why["lies"])

    def test_plate_armour_is_no_dish(self):
        self.assertNotEqual(words.subject_of("a suit of plate armour").noun, "plate")


class Finish(unittest.TestCase):
    def test_a_book_made_on_its_edge_finishes_lying_with_its_paint_following(self):
        folder = tempfile.mkdtemp()
        try:
            job = pipeline.Job(os.path.join(folder, "book"), "a leather-bound book, closed")
            corners, triangles = book(tilt_deg=35.0)
            mask = render.silhouette(corners, triangles, 0, 512, 10)
            Image.fromarray(np.where(mask[..., None], 90, 210).astype(np.uint8).repeat(3, -1)).save(
                job.path("picture.png"))
            Image.fromarray(mask.astype(np.uint8) * 255).save(job.path("mask.png"))
            np.savez_compressed(job.path("shape.npz"), corners=corners, triangles=triangles)
            # The front cover (facing +Z as made) painted red, the rest blue: after laying, red must face up.
            from model_maker import mesh_tools as M

            points, normals, _ = M.sampled(corners, triangles, 300000, seed=4)
            front = normals @ np.array([0.0, np.sin(np.radians(35)), np.cos(np.radians(35))]) > 0.9
            colours = np.where(front[:, None], [0.9, 0.1, 0.1], [0.1, 0.1, 0.9])
            np.savez_compressed(job.path("painted.npz"), colours=np.full((len(corners), 3), 0.5), known=None,
                                cloud_points=points, cloud_normals=normals, cloud_colours=colours)
            pipeline.finish_stage([job], pipeline.Settings(texture=512, verify=False), report=lambda _: None)
            with open(job.path(job["slug"] + ".provenance.json")) as file:
                provenance = json.load(file)
            shape_checks = {c["name"]: c["passed"] for c in provenance["checks"]["shape"]}
            for name in ("base level", "centre over the base", "every foot on the ground",
                         "outline matches the picture"):
                self.assertTrue(shape_checks[name], (name, provenance["checks"]["shape"]))
            self.assertTrue(provenance["laid"]["laid"])
            self.assertGreater(provenance["checks"]["paint"][0]["value"], 99.0)
            self.assertIn("laid on its broad face", shape_checks)
            from model_maker import glb

            with open(job.path(job["slug"] + ".glb"), "rb") as file:
                document, binary = glb.read_document(file.read())
            positions = glb.accessor_values(document, binary,
                                            document["meshes"][0]["primitives"][0]["attributes"]["POSITION"])
            self.assertLess(float(np.ptp(positions[:, 1])), 0.05, "it lies flat in the .glb")
        finally:
            shutil.rmtree(folder)

    def test_a_job_recorded_before_lying_was_read_takes_it_from_its_words(self):
        folder = tempfile.mkdtemp()
        try:
            job = pipeline.Job(os.path.join(folder, "book"), "a leather-bound book, closed")
            del job.record["subject"]["lies"]
            job.save()
            self.assertTrue(pipeline.Job(job.folder).subject.lies)
        finally:
            shutil.rmtree(folder)


if __name__ == "__main__":
    unittest.main()
