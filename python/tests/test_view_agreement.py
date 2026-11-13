import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from model_maker import inspection, pipeline, view_agreement, words

WHITE, BLACK = (0.92, 0.92, 0.9), (0.04, 0.04, 0.05)
# Towards each of the six cameras, as paint.painted gives them: four round the object, then above and below.
DIRECTIONS = np.array([[0, 0, 1], [1, 0, 0], [0, 0, -1], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)


def painted_round(colours=(WHITE,) * 4, spacing=0.002, radius=0.1, height=0.1):
    """A cylinder's side as the four side views would paint it: each view's points where it sees the side with a
    facing of 0.55 or more (as paint.painted keeps them), so neighbours share the strip round 45 degrees, painted in
    that view's colour. Returns (points, normals, colours, view_of)."""
    turns = np.arange(0, 2 * np.pi, spacing / radius)
    heights = np.arange(0, height, spacing)
    angle, y = (a.ravel() for a in np.meshgrid(turns, heights))
    normals = np.stack([np.sin(angle), np.zeros_like(angle), np.cos(angle)], 1)
    points = normals * radius + np.stack([np.zeros_like(y), y, np.zeros_like(y)], 1)
    parts = []
    for view in range(4):
        seen = normals @ DIRECTIONS[view] >= 0.55
        parts.append((points[seen], normals[seen], np.tile(colours[view], (int(seen.sum()), 1)),
                      np.full(int(seen.sum()), view)))
    return tuple(np.concatenate([part[k] for part in parts]) for k in range(4))


class ViewAgreementTest(unittest.TestCase):
    def test_views_that_agree_are_not_unlike(self):
        found = view_agreement.measured(*painted_round())
        self.assertEqual(found["worst"], 0.0)
        self.assertTrue(all(v == 0.0 for v in found["views"]), found)

    def test_one_view_out_of_family_is_found_and_only_it(self):
        # The sealed run's teapot: the second view painted the white body black.
        found = view_agreement.measured(*painted_round((WHITE, BLACK, WHITE, WHITE)))
        self.assertGreaterEqual(found["worst"], view_agreement.REPAINT_FROM)
        self.assertEqual(int(np.argmax(found["views"])), 1)
        self.assertGreater(found["pairs"]["0-1"], 0.9)
        self.assertEqual(found["pairs"]["2-3"], 0.0)
        # Its neighbours each disagree with it on one side only: their lesser pair is 0.
        self.assertEqual(found["views"][0], 0.0)
        self.assertEqual(found["views"][2], 0.0)

    def test_a_view_a_little_lighter_is_not_unlike(self):
        lighter = tuple(min(1.0, c + 0.06) for c in WHITE)
        self.assertEqual(view_agreement.measured(*painted_round((WHITE, lighter, WHITE, WHITE)))["worst"], 0.0)

    def test_a_pair_too_small_to_judge_is_none(self):
        points, normals, colours, view_of = painted_round()
        lonely = view_of != 1
        found = view_agreement.measured(points[lonely], normals[lonely], colours[lonely], view_of[lonely])
        self.assertIsNone(found["pairs"]["0-1"])
        self.assertIsNone(found["views"][1])

    def test_the_try_that_agrees_best_is_kept_the_first_of_equals(self):
        self.assertEqual(view_agreement.kept_try([{"worst": 0.9}, {"worst": 0.03}, {"worst": 0.1}]), 1)
        self.assertEqual(view_agreement.kept_try([{"worst": 0.2}, {"worst": 0.2}]), 0)

    def test_one_view_per_place(self):
        tilted = np.radians(10)
        front = np.array([np.sin(tilted), 0, np.cos(tilted)])
        seam = np.array([np.sqrt(0.5), 0, np.sqrt(0.5)])
        # Patches of points 1 mm apart, each painted by the views named: (where, normal, views).
        patches = [((0, 0, 0), front, (0, 1)), ((0, 0.1, 0), seam, (0, 1)), ((0, 0.2, 0), front, (1,)),
                   ((0, 0.3, 0), front, (-1, 1)), ((0, 0.4, 0), front, (1,)), ((0, 0.4, 0), -front, (0,))]
        grid = np.stack(np.meshgrid(np.arange(10), np.arange(10)), -1).reshape(-1, 2) * 0.001
        points, normals, view_of, patch_of = [], [], [], []
        for k, (where, normal, views) in enumerate(patches):
            for view in views:
                points.append(np.asarray(where) + np.stack([grid[:, 0], grid[:, 1], np.zeros(len(grid))], 1))
                normals.append(np.tile(normal, (len(grid), 1)))
                view_of.append(np.full(len(grid), view))
                patch_of.append(np.full(len(grid), k))
        points, normals, view_of, patch_of = (np.concatenate(a) for a in (points, normals, view_of, patch_of))
        keep = view_agreement.one_view_per_place(points, normals, view_of, DIRECTIONS)

        def kept(patch, view):
            return bool(keep[(patch_of == patch) & (view_of == view)].all())

        def dropped(patch, view):
            return not keep[(patch_of == patch) & (view_of == view)].any()

        # Seen squarely from the front and from the side at a slant: the front's paint only.
        self.assertTrue(kept(0, 0) and dropped(0, 1))
        # On the 45-degree seam both views are as square: both kept.
        self.assertTrue(kept(1, 0) and kept(1, 1))
        # A place only the side painted (the front blocked there) keeps the side's paint.
        self.assertTrue(kept(2, 1))
        # The touch-up's fill is always kept, and does not push a view's paint out.
        self.assertTrue(kept(3, -1) and kept(3, 1))
        # A thin wall's other side, painted by the front, is another surface: the side's paint here stays.
        self.assertTrue(kept(4, 1) and kept(5, 0))


class FakeJob:
    def __init__(self, folder):
        self.folder = folder
        self.subject = words.subject_of("a ceramic teapot")

    def path(self, *parts):
        return os.path.join(self.folder, *parts)


class RepaintTest(unittest.TestCase):
    def painted_by_seed(self, colours_by_seed, calls):
        def painted(painter, corners, triangles, rgb, mask, text, seed=0, views_out=None, report=print, steps=20,
                    with_views=False, turn=None):
            calls.append(seed)
            os.makedirs(views_out, exist_ok=True)
            for k in range(6):
                Image.new("RGB", (8, 8), (seed, k, 0)).save(os.path.join(views_out, f"view-{k}.png"))
            points, normals, colours, view_of = painted_round(colours_by_seed[seed][:4], spacing=0.004)
            if len(colours_by_seed[seed]) > 4:
                # The second view black only above this height.
                colours[(view_of == 1) & (points[:, 1] > colours_by_seed[seed][4])] = BLACK
            return (np.full((3, 3), seed), np.ones(3, bool), (points, normals, colours),
                    {"view_of": view_of.astype(np.int8), "directions": DIRECTIONS.astype(np.float32)})
        return painted

    def run_views(self, colours_by_seed):
        calls = []
        settings = mock.Mock(paint_steps=20)
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch("model_maker.paint.painted", self.painted_by_seed(colours_by_seed, calls)):
            os.makedirs(os.path.join(folder, "views"))
            found = pipeline.painted_views(FakeJob(folder), None, {"corners": None, "triangles": None}, None, None,
                                           settings, report=lambda *_: None)
            kept_view = Image.open(os.path.join(folder, "views", "view-0.png")).getpixel((0, 0))[0]
        return found, calls, kept_view

    def test_views_that_agree_are_painted_once(self):
        found, calls, kept_view = self.run_views({0: (WHITE,) * 4})
        self.assertEqual(calls, [0])
        self.assertEqual(found[4]["kept_seed"], 0)
        self.assertEqual(kept_view, 0)

    def test_a_view_out_of_family_is_painted_again_and_the_best_kept(self):
        odd = (WHITE, BLACK, WHITE, WHITE)
        found, calls, kept_view = self.run_views({0: odd, 1: (WHITE,) * 4, 2: odd})
        self.assertEqual(calls, [0, 1])
        record = found[4]
        self.assertEqual(record["kept_seed"], 1)
        self.assertEqual([t["seed"] for t in record["tries"]], [0, 1])
        self.assertGreaterEqual(record["tries"][0]["worst"], view_agreement.REPAINT_FROM)
        # The kept try's colours, cloud and views.
        self.assertEqual(found[0][0, 0], 1)
        self.assertEqual(kept_view, 1)

    def test_at_most_two_more_seeds_and_the_least_unlike_kept(self):
        # Seed 1's odd view is black over its top 60 % only: still over the limit, but less unlike than the others.
        found, calls, _ = self.run_views({0: (WHITE, BLACK, WHITE, WHITE), 1: (WHITE, WHITE, WHITE, WHITE, 0.04),
                                          2: (WHITE, BLACK, WHITE, WHITE)})
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(found[4]["kept_seed"], 1)

    def test_the_check_is_shown_not_counted_and_names_the_seeds(self):
        record = {"tries": [{"seed": 0, "worst": 0.91}, {"seed": 1, "worst": 0.03}], "kept_seed": 1}
        check = inspection.views_agree_check(record)
        self.assertFalse(check.counted)
        self.assertTrue(check.passed)
        self.assertAlmostEqual(check.value, 3.0)
        self.assertIn("seed 0: 91%", check.words)
        self.assertIn("seed 1 kept", check.words)


if __name__ == "__main__":
    unittest.main()
