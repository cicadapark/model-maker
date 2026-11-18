import json
import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from model_maker import bake, glb, mesh_tools as M, object_checks as O, pipeline, render, simplify

from . import shapes

REPOSITORY = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def painted_half(corners, triangles, count=200000):
    """A colour cloud on the surface's upper half only (red above, the rest unpainted)."""
    points, normals, _ = M.sampled(corners, triangles, count, seed=5)
    upper = points[:, 1] > np.median(corners[:, 1])
    colours = np.tile([0.8, 0.1, 0.1], (int(upper.sum()), 1))
    return points[upper], normals[upper], colours


class Bake(unittest.TestCase):
    def test_the_atlas_holds_every_triangle_inside_the_square(self):
        corners, triangles = shapes.chair()
        split_corners, split_triangles, uv, original, density = bake.atlas(corners, triangles, 1024)
        self.assertEqual(len(split_triangles), len(triangles))
        self.assertTrue((uv >= 0).all() and (uv <= 1).all())
        self.assertTrue(np.allclose(split_corners, corners[original]))
        self.assertGreater(density, 100)

    def test_coverage_counts_the_painted_share(self):
        corners, triangles = shapes.chair()
        colours = np.full((len(corners), 3), 0.5)
        made = bake.baked(corners, triangles, colours, painted_half(corners, triangles), side=512)
        self.assertTrue(0.25 < made["coverage"] < 0.75, made["coverage"])
        self.assertLess(made["overlap"], 0.05)
        texture = made["texture"]
        self.assertTrue(((texture[..., 0] > 0.7) & (texture[..., 1] < 0.2)).any(), "no painted red")
        full = bake.baked(corners, triangles, colours, (*M.sampled(corners, triangles, 400000, seed=6)[:2],
                                                        np.full((400000, 3), 0.3)), side=512)
        self.assertGreater(full["coverage"], 0.97)


class BlackViews(unittest.TestCase):
    def test_a_painted_view_that_came_back_black_fails(self):
        # The coverage counts a texel painted whatever its colour.
        from model_maker import inspection

        good = np.full((64, 64, 3), 0.5)
        good[20:40, 30:34] = 0.3
        black = np.zeros((64, 64, 3))
        self.assertTrue(inspection.views_check([inspection.black_share(good)] * 6).passed)
        failed = inspection.views_check([inspection.black_share(good)] * 5 + [inspection.black_share(black)])
        self.assertFalse(failed.passed, failed.words)

    def test_a_black_thing_on_its_grey_backdrop_is_not_a_broken_view(self):
        # A smartphone's black screen filled 37 % of a view and was refused as broken (16 Nov 2026).
        from model_maker import inspection

        phone = np.full((64, 64, 3), 0.5)
        phone[8:56, 12:52] = 0.0
        self.assertTrue(inspection.views_check([inspection.black_share(phone)] * 6).passed)


class SeenCoverage(unittest.TestCase):
    def test_the_inside_of_a_deep_narrow_well_counts_little(self):
        corners, triangles = shapes.surface_of([((0.0, 0.0, 0.0), (0.3, 0.3, 0.3))], 0.005,
                                               cut=[((0.12, 0.05, 0.12), (0.18, 0.4, 0.18))])
        points, normals, _ = M.sampled(corners, triangles, 400000, seed=7)
        in_well = (np.abs(points[:, 0] - 0.15) < 0.035) & (np.abs(points[:, 2] - 0.15) < 0.035) & \
            (points[:, 1] < 0.295)
        cloud = (points[~in_well], normals[~in_well], np.full((int((~in_well).sum()), 3), 0.4))
        made = bake.baked(corners, triangles, np.full((len(corners), 3), 0.5), cloud, side=512)
        self.assertLess(made["coverage_all_texels"], 0.93)
        self.assertGreater(made["coverage"], made["coverage_all_texels"])
        self.assertGreater(made["coverage"], 0.97)


class Export(unittest.TestCase):
    def test_a_glb_reads_back_with_its_mesh_texture_and_marks(self):
        corners, triangles = shapes.sword()
        made = bake.baked(corners, triangles, np.full((len(corners), 3), 0.6), None, side=256)
        normals = M.corner_normals(corners, triangles)[made["original"]]
        marks = {"grip": ([0.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
                 "tip": ([0.0, 0.95, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])}
        data = glb.glb_bytes("sword", made["corners"], made["triangles"], normals, made["uvs"], made["texture"],
                             marks=marks, extras={"prompt": "an old iron sword"})
        document, binary = glb.read_document(data)
        self.assertEqual([n["name"] for n in document["nodes"]], ["sword", "grip", "tip"])
        primitive = document["meshes"][0]["primitives"][0]
        positions = glb.accessor_values(document, binary, primitive["attributes"]["POSITION"])
        indices = glb.accessor_values(document, binary, primitive["indices"])
        self.assertEqual(len(positions), len(made["corners"]))
        self.assertEqual(len(indices), 3 * len(triangles))
        self.assertEqual(document["asset"]["extras"]["prompt"], "an old iron sword")

    def test_a_pair_is_the_one_mesh_placed_twice_mirrored_or_copied_beside_it(self):
        # 16 Nov 2026: a pair (boots) is one object made and its mirror image, an exact match.
        corners, triangles = shapes.sword()
        made = bake.baked(corners, triangles, np.full((len(corners), 3), 0.6), None, side=128)
        normals = M.corner_normals(corners, triangles)[made["original"]]
        widths = np.ptp(made["corners"][:, [0, 2]], axis=0)
        axis = 0 if widths[0] <= widths[1] * 1.1 else 2  # across the narrower width: side by side, not end to end
        other = 2 - axis
        if widths[axis // 2] < glb.FLAT_SHARE * min(widths[other // 2], float(np.ptp(made["corners"][:, 1]))):
            axis = other  # a flat thing (this sword) stands beside its pair in its own plane
        low, high = float(made["corners"][:, axis].min()), float(made["corners"][:, axis].max())
        for pair in ("mirror", "copy"):
            data = glb.glb_bytes("boot", made["corners"], made["triangles"], normals, made["uvs"], made["texture"],
                                 pair=pair)
            document, _ = glb.read_document(data)
            self.assertEqual(len(document["meshes"]), 1)
            roots = document["scenes"][0]["nodes"]
            self.assertEqual(len(roots), 2)
            second = document["nodes"][roots[1]]
            self.assertEqual(second["mesh"], 0)
            scale = second.get("scale", [1.0, 1.0, 1.0])
            self.assertEqual(scale[axis], -1.0 if pair == "mirror" else 1.0)
            self.assertEqual([v for k, v in enumerate(scale) if k != axis], [1.0, 1.0])
            # The second's span across that width starts past the first's far side: they do not overlap.
            ends = sorted(scale[axis] * x + second["translation"][axis] for x in (low, high))
            self.assertGreater(ends[0], high)
        single, _ = glb.read_document(glb.glb_bytes("boot", made["corners"], made["triangles"], normals, made["uvs"],
                                                    made["texture"]))
        self.assertEqual(single["scenes"][0]["nodes"], [0])

    def test_a_shoe_s_mirror_stands_on_its_arch_side(self):
        # Wooden clogs were placed a left on the right (16 Nov 2026): the mirror goes to the side the arch faces.
        from model_maker import glb as G
        for arch in (1, -1):
            # A flat sole 30 cm long and 10 cm wide along z, its arch cut 2 cm into one side along the middle third.
            xs, zs = np.meshgrid(np.linspace(0, 0.30, 61), np.linspace(-0.05, 0.05, 21))
            sole = np.stack([xs.ravel(), np.zeros(xs.size), zs.ravel()], 1)
            middle = (sole[:, 0] > 0.08) & (sole[:, 0] < 0.22)  # the arch's cut reaches past the middle third
            sole = sole[~(middle & (arch * sole[:, 2] > 0.03))]
            top = sole + [0, 0.08, 0]
            corners = np.concatenate([sole, top])
            node = G.pair_node("clog", corners, "mirror", footwear=True)
            self.assertEqual(G.pair_node("glove", corners, "mirror")["translation"][2] > 0, True)  # not footwear
            self.assertEqual(node["scale"], [1.0, 1.0, -1.0])
            self.assertEqual(np.sign(node["translation"][2]), arch)

    def test_a_flat_thing_s_pair_stands_beside_it_in_its_own_plane(self):
        # Two daggers placed flat against flat 2 cm apart read as one from the front (review, 16 Nov 2026).
        from model_maker import glb as G
        blade = np.array([[x, y, z] for x in (-0.02, 0.02) for y in (0.0, 0.3) for z in (-0.003, 0.003)], dtype=float)
        node = G.pair_node("dagger", blade, "copy")
        self.assertGreater(node["translation"][0], 0.04)
        self.assertEqual(node["translation"][2], 0.0)

    def test_the_second_of_a_pair_has_the_first_s_marks(self):
        corners, triangles = shapes.sword()
        made = bake.baked(corners, triangles, np.full((len(corners), 3), 0.6), None, side=128)
        normals = M.corner_normals(corners, triangles)[made["original"]]
        marks = {"grip": ([0.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])}
        document, _ = glb.read_document(glb.glb_bytes("dagger", made["corners"], made["triangles"], normals,
                                                      made["uvs"], made["texture"], marks=marks, pair="copy"))
        second = document["nodes"][document["scenes"][0]["nodes"][1]]
        self.assertEqual([document["nodes"][c]["name"] for c in second["children"]], ["grip (pair)"])

    def test_a_part_our_code_adds_is_its_own_mesh_in_its_own_colour(self):
        corners, triangles = shapes.bow()
        made = bake.baked(corners, triangles, np.full((len(corners), 3), 0.6), None, side=256)
        normals = M.corner_normals(corners, triangles)[made["original"]]
        from model_maker.weapon_checks import STRING_COLOUR, bow_string
        string = bow_string(corners, triangles)
        data = glb.glb_bytes("bow", made["corners"], made["triangles"], normals, made["uvs"], made["texture"],
                             parts=[{"name": "string", "corners": string["corners"],
                                     "triangles": string["triangles"], "colour": STRING_COLOUR}])
        document, _ = glb.read_document(data)
        self.assertEqual([n["name"] for n in document["nodes"]], ["bow", "string"])
        self.assertEqual(document["nodes"][0]["children"], [1])
        self.assertEqual(len(document["meshes"]), 2)
        self.assertEqual(document["materials"][1]["pbrMetallicRoughness"]["baseColorFactor"][:3],
                         list(STRING_COLOUR))

    def test_a_marks_rotation_turns_its_y_to_the_given_up(self):
        q = np.array(glb.mark_rotation([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]))
        x, y, z, w = q
        # The quaternion's rotation of +Y.
        turned = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
        self.assertTrue(np.allclose(turned, [1.0, 0.0, 0.0], atol=1e-6))


class RepairWords(unittest.TestCase):
    def test_a_loose_piece_moved_to_touch_is_said_in_the_report(self):
        from model_maker import report
        words = report.repair_words({"holes_filled": 2, "pieces_joined": [{"moved_mm": 1.4, "at": [0, 0, 0]},
                                                                         {"moved_mm": 2.6, "at": [0, 1, 0]}]})
        self.assertIn("2 small holes filled", words)
        self.assertIn("2 loose pieces within 3 mm moved to touch the main piece (at most 2.6 mm)", words)
        self.assertNotIn("moved", report.repair_words({"holes_filled": 1, "pieces_joined": []}))


class Simplify(unittest.TestCase):
    def test_a_dense_surface_comes_to_its_budget(self):
        built = os.path.join(REPOSITORY, "target", "debug", "model-maker-simplify")
        if not os.environ.get("MODEL_MAKER_SIMPLIFY") and not os.path.exists(built):
            self.skipTest("model-maker-simplify is not built")
        os.environ.setdefault("MODEL_MAKER_SIMPLIFY", built)
        corners, triangles = shapes.chair(cell=0.006)
        small_corners, small_triangles, kept, method = simplify.simplified(corners, triangles, 4000)
        self.assertEqual(method, "edge collapse")
        self.assertLessEqual(len(small_triangles), 4000)
        self.assertTrue(np.allclose(small_corners, corners[kept]))
        self.assertEqual(O.wholeness(small_corners, small_triangles)["holes"], 0)


class Finish(unittest.TestCase):
    def test_a_job_with_its_shape_and_paint_finishes_with_a_glb_a_sidecar_and_a_report(self):
        folder = tempfile.mkdtemp()
        try:
            job = pipeline.Job(os.path.join(folder, "chair"), "an old looking chair")
            corners, triangles = shapes.chair()
            corners = O.set_down(corners, job["size_m"])
            mask = render.silhouette(corners, triangles, 20, 512)
            Image.fromarray(np.where(mask[..., None], 120, 210).astype(np.uint8).repeat(3, -1)).save(
                job.path("picture.png"))
            Image.fromarray(mask.astype(np.uint8) * 255).save(job.path("mask.png"))
            np.savez_compressed(job.path("shape.npz"), corners=corners, triangles=triangles)
            cloud = painted_half(corners, triangles)
            np.savez_compressed(job.path("painted.npz"), colours=np.full((len(corners), 3), 0.5), known=None,
                                cloud_points=cloud[0], cloud_normals=cloud[1], cloud_colours=cloud[2])
            pipeline.finish_stage([job], pipeline.Settings(texture=512, verify=False), report=lambda _: None)
            self.assertTrue(job.has("chair.glb"))
            self.assertTrue(job.has(os.path.join("report", "report.html")))
            with open(job.path("chair.provenance.json")) as file:
                provenance = json.load(file)
            self.assertEqual({m["id"] for m in provenance["models"]}, set(pipeline.MODELS_USED))
            self.assertTrue(all(m["licence"] for m in provenance["models"]))
            shape_checks = {c["name"]: c["passed"] for c in provenance["checks"]["shape"]}
            self.assertTrue(all(shape_checks.values()), shape_checks)
            # Half the chair was painted: the coverage check says so.
            self.assertFalse(provenance["checks"]["paint"][0]["passed"])
            self.assertTrue(job.record["status"].startswith("fails: paint covers the surface"))
        finally:
            shutil.rmtree(folder)

    def test_the_state_judges_verdict_on_the_chosen_picture_is_in_the_sidecar_with_its_model(self):
        from model_maker import picture_checks, state_judge

        folder = tempfile.mkdtemp()
        try:
            job = pipeline.Job(os.path.join(folder, "chest"), "a wooden treasure chest")
            corners, triangles = shapes.surface_of([((-0.4, 0.0, -0.27), (0.4, 0.5, 0.27))], 0.02)
            corners = O.set_down(corners, job["size_m"])
            mask = render.silhouette(corners, triangles, 20, 512)
            Image.fromarray(np.where(mask[..., None], 120, 210).astype(np.uint8).repeat(3, -1)).save(
                job.path("picture.png"))
            Image.fromarray(mask.astype(np.uint8) * 255).save(job.path("mask.png"))
            np.savez_compressed(job.path("shape.npz"), corners=corners, triangles=triangles)
            chosen = picture_checks.PictureResult()
            state_judge.add_state_check(chosen, job.subject, 0.29)
            job.record["picture"] = {"seed": 7, "state_judged": True, "tries": {"7": chosen.as_dict()}}
            job.save()
            pipeline.finish_stage([job], pipeline.Settings(texture=256, verify=False), report=lambda _: None)
            with open(job.path(f"{job['slug']}.provenance.json")) as file:
                provenance = json.load(file)
            picture_checks_said = {c["name"]: c["passed"] for c in provenance["checks"]["picture"]}
            self.assertFalse(picture_checks_said[state_judge.NAME])
            self.assertIn(state_judge.NAME, job.record["status"])
            self.assertIn(pipeline.STATE_MODEL, [m["id"] for m in provenance["models"]])
        finally:
            shutil.rmtree(folder)

    def test_the_subject_judges_verdict_on_the_chosen_picture_is_in_the_sidecar_with_its_model(self):
        from model_maker import picture_checks, subject_judge

        folder = tempfile.mkdtemp()
        try:
            job = pipeline.Job(os.path.join(folder, "jug"), "a clay jug")
            corners, triangles = shapes.surface_of([((-0.1, 0.0, -0.1), (0.1, 0.3, 0.1))], 0.01)
            corners = O.set_down(corners, job["size_m"])
            mask = render.silhouette(corners, triangles, 20, 512)
            Image.fromarray(np.where(mask[..., None], 120, 210).astype(np.uint8).repeat(3, -1)).save(
                job.path("picture.png"))
            Image.fromarray(mask.astype(np.uint8) * 255).save(job.path("mask.png"))
            np.savez_compressed(job.path("shape.npz"), corners=corners, triangles=triangles)
            chosen = picture_checks.PictureResult()
            subject_judge.add_subject_check(chosen, job.subject, 0.49, "a bottle")
            job.record["picture"] = {"seed": 7, "state_judged": False, "subject_judged": True,
                                     "tries": {"7": chosen.as_dict()}}
            job.save()
            pipeline.finish_stage([job], pipeline.Settings(texture=256, verify=False), report=lambda _: None)
            with open(job.path(f"{job['slug']}.provenance.json")) as file:
                provenance = json.load(file)
            picture_checks_said = {c["name"]: c["passed"] for c in provenance["checks"]["picture"]}
            self.assertFalse(picture_checks_said[subject_judge.NAME])
            self.assertIn(subject_judge.NAME, job.record["status"])
            self.assertIn(pipeline.STATE_MODEL, [m["id"] for m in provenance["models"]])
        finally:
            shutil.rmtree(folder)

    def anvil_with_its_try(self, folder, lettering_check=None, lettering_record=None):
        """An unpainted anvil's job ready for its finish, its chosen try's record holding `lettering_check`."""
        from model_maker import picture_checks

        job = pipeline.Job(os.path.join(folder, "anvil"), "an iron anvil")
        corners, triangles = shapes.surface_of([((-0.2, 0.0, -0.1), (0.2, 0.3, 0.1))], 0.01)
        corners = O.set_down(corners, job["size_m"])
        mask = render.silhouette(corners, triangles, 20, 512)
        Image.fromarray(np.where(mask[..., None], 120, 210).astype(np.uint8).repeat(3, -1)).save(
            job.path("picture.png"))
        Image.fromarray(mask.astype(np.uint8) * 255).save(job.path("mask.png"))
        np.savez_compressed(job.path("shape.npz"), corners=corners, triangles=triangles)
        chosen = picture_checks.PictureResult(checks=[lettering_check] if lettering_check is not None else [])
        job.record["picture"] = {"seed": 7, "tries": {"7": chosen.as_dict()}}
        if lettering_record is not None:
            job.record["lettering"] = lettering_record
        job.save()
        return job

    def finished_with(self, job, reader, readings=()):
        from unittest import mock

        from model_maker import lettering

        # The ledger beside the job, not the repository's metrics/.
        ledger_folder = os.path.join(os.path.dirname(job.folder), "ledger")
        with mock.patch.object(lettering, "reader", return_value=reader), \
                mock.patch.object(lettering, "every_reading", return_value=list(readings)), \
                mock.patch.dict(os.environ, {"MODEL_MAKER_LEDGER": ledger_folder}):
            pipeline.finish_stage([job], pipeline.Settings(texture=256, verify=False), report=lambda _: None)
        with open(job.path(f"{job['slug']}.provenance.json")) as file:
            return json.load(file)

    def test_a_refinish_reads_the_lettering_again(self):
        # The 0.2.0 review: the other picture checks are measured again at a finish; the lettering needs no card.
        from model_maker import lettering, picture_checks

        folder = tempfile.mkdtemp()
        try:
            passed_then = picture_checks.Check(lettering.NAME, 0.0, lettering.MOST_CHARACTERS, "characters", True,
                                               "no lettering read")
            job = self.anvil_with_its_try(folder, passed_then, {"judged": True, "tesseract": "5.3.0",
                                                                 "read_at": "pictures", "why": ""})
            read_now = [lettering.Word("EANITIES", 90.0, 200.0, 200.0, 60.0, 12.0, "as painted")]
            provenance = self.finished_with(job, lettering.Reader(True, "5.5.0", ""), read_now)
            said = {c["name"]: c for c in provenance["checks"]["picture"]}
            self.assertFalse(said[lettering.NAME]["passed"])
            self.assertIn(lettering.NAME, job.record["status"])
            self.assertIs(provenance["lettering_judged"], True)
            self.assertEqual(provenance["tesseract_version"], "5.5.0")
            self.assertEqual(job.record["lettering"]["read_at"], "finish")
        finally:
            shutil.rmtree(folder)

    def test_without_a_reader_a_refinish_keeps_the_reading_made_when_the_try_was_chosen(self):
        from model_maker import lettering, picture_checks

        folder = tempfile.mkdtemp()
        try:
            refused_then = picture_checks.Check(lettering.NAME, 12.0, lettering.MOST_CHARACTERS, "characters", False,
                                                "lettering painted on the object: read \"EANITIES\"")
            job = self.anvil_with_its_try(folder, refused_then, {"judged": True, "tesseract": "5.3.0",
                                                                  "read_at": "pictures", "why": ""})
            provenance = self.finished_with(job, lettering.Reader(False, None, lettering.NOT_INSTALLED))
            said = {c["name"]: c for c in provenance["checks"]["picture"]}
            self.assertFalse(said[lettering.NAME]["passed"])
            self.assertIs(provenance["lettering_judged"], True)
            self.assertEqual(provenance["tesseract_version"], "5.3.0")
        finally:
            shutil.rmtree(folder)

    def test_a_job_never_read_says_so_in_its_sidecar_report_and_ledger_line(self):
        from model_maker import lettering

        folder = tempfile.mkdtemp()
        try:
            job = self.anvil_with_its_try(folder, lettering.not_judged(lettering.NOT_INSTALLED))
            provenance = self.finished_with(job, lettering.Reader(False, None, lettering.NOT_INSTALLED))
            self.assertIs(provenance["lettering_judged"], False)
            self.assertIsNone(provenance["tesseract_version"])
            # Not counted: the status is what the counted checks say; the sidecar, report and ledger say it was unread.
            self.assertNotIn(lettering.NAME, job.record["status"])
            with open(job.path("report", "report.html")) as file:
                self.assertIn("not read for painted lettering", file.read().lower())
            with open(os.path.join(folder, "ledger", "jobs.jsonl")) as file:
                line = [json.loads(text) for text in file][-1]
            self.assertIs(line["lettering_judged"], False)
        finally:
            shutil.rmtree(folder)

    def finish_judged(self, job, found, require=False, keep_going=True):
        """`finish_stage` with the question judge `found` (lettering read as clean); what it reported."""
        from unittest import mock

        from model_maker import lettering, question_judge

        said = []
        ledger_folder = os.path.join(os.path.dirname(job.folder), "ledger")
        with mock.patch.object(lettering, "reader", return_value=lettering.Reader(True, "5.3.0", "")), \
                mock.patch.object(lettering, "every_reading", return_value=[]), \
                mock.patch.object(question_judge, "found", return_value=found), \
                mock.patch.dict(os.environ, {"MODEL_MAKER_LEDGER": ledger_folder}):
            pipeline.finish_stage([job], pipeline.Settings(texture=256, verify=False,
                                                           require_question_judge=require),
                                  report=said.append, keep_going=keep_going)
        return said

    def test_without_the_question_judge_a_job_is_not_asked_and_its_ledger_line_says_so(self):
        from model_maker import question_judge

        folder = tempfile.mkdtemp()
        try:
            job = self.anvil_with_its_try(folder)
            said = self.finish_judged(job, question_judge.Found(False, why="not judged (not installed)"))
            self.assertTrue(any("not installed" in line for line in said), said)
            self.assertIs(job.record["question_judge"]["judged"], False)
            self.assertNotIn(question_judge.NAME, job.record["status"], "not counted when not required")
            with open(os.path.join(folder, "ledger", "jobs.jsonl")) as file:
                line = [json.loads(text) for text in file][-1]
            self.assertIs(line["question_judged"], False)
            # Required (a measured run's job finished again): it fails, counted.
            job.record.setdefault("settings", {})["require_question_judge"] = True
            self.finish_judged(job, question_judge.Found(False, why="not judged (not installed)"))
            self.assertIn(question_judge.NAME, job.record["status"])
        finally:
            shutil.rmtree(folder)

    def test_a_judge_that_cannot_start_is_said_once_and_the_jobs_still_finish(self):
        from model_maker import question_judge

        folder = tempfile.mkdtemp()
        try:
            job = self.anvil_with_its_try(folder)
            broken = os.path.join(folder, "llama-server")
            with open(broken, "w") as file:
                file.write("#!/bin/sh\necho cannot load the model\nexit 1\n")
            os.chmod(broken, 0o755)
            said = self.finish_judged(job, question_judge.Found(True, broken, broken, broken))
            self.assertEqual(sum("cannot load the model" in line for line in said), 1, said)
            self.assertIn("did not start", job.record["question_judge"]["why"])
            self.assertFalse(any("aborted" in line for line in said), said)
            self.assertTrue(job.has(f"{job['slug']}.glb"))
            self.assertIs(job.record["question_judge"]["judged"], False)
        finally:
            shutil.rmtree(folder)

    def test_a_job_that_raises_under_the_judge_is_its_own_error_and_the_judge_is_stopped(self):
        from unittest import mock

        from model_maker import question_judge

        folder = tempfile.mkdtemp()
        started = []

        class StandIn:
            threads, restarted, process = 1, False, None

            def __init__(self, where):
                started.append(self)
                self.stopped = False

            def __enter__(self):
                return self

            def __exit__(self, *_):
                self.stopped = True

        try:
            job = self.anvil_with_its_try(folder)
            with mock.patch.object(question_judge, "Judge", StandIn), \
                    mock.patch.object(question_judge, "object_check", side_effect=ValueError("a broken turntable")):
                said = self.finish_judged(job, question_judge.Found(True, "s", "m", "v"))
            self.assertEqual(len(started), 1)
            self.assertTrue(started[0].stopped)
            self.assertTrue(any("aborted at finish: ValueError: a broken turntable" in line for line in said), said)
            # Without keep_going (make): the error goes up, and the judge is stopped all the same.
            with mock.patch.object(question_judge, "Judge", StandIn), \
                    mock.patch.object(question_judge, "object_check", side_effect=ValueError("a broken turntable")):
                with self.assertRaises(ValueError):
                    self.finish_judged(job, question_judge.Found(True, "s", "m", "v"), keep_going=False)
            self.assertTrue(started[1].stopped)
        finally:
            shutil.rmtree(folder)

    def test_a_web_in_a_mug_s_handle_is_carved_and_the_sidecar_says_so(self):
        folder = tempfile.mkdtemp()
        try:
            job = pipeline.Job(os.path.join(folder, "mug"), "a clay jug")
            open_corners, open_triangles = shapes.mug()
            open_corners = O.set_down(open_corners, job["size_m"])
            # Turned 10 degrees: further round, the body's corner hides the loop.
            mask = render.silhouette(open_corners, open_triangles, 10, 512)
            Image.fromarray(np.where(mask[..., None], 120, 210).astype(np.uint8).repeat(3, -1)).save(
                job.path("picture.png"))
            Image.fromarray(mask.astype(np.uint8) * 255).save(job.path("mask.png"))
            corners, triangles = shapes.mug("lower")
            corners = O.set_down(corners, job["size_m"])
            np.savez_compressed(job.path("shape.npz"), corners=corners, triangles=triangles)
            pipeline.finish_stage([job], pipeline.Settings(texture=256, verify=False), report=lambda _: None)
            with open(job.path(f"{job['slug']}.provenance.json")) as file:
                provenance = json.load(file)
            self.assertTrue(provenance["carved"]["carved"], provenance["carved"])
            shape_checks = {c["name"]: c for c in provenance["checks"]["shape"]}
            for name in ("outline matches the picture", "no holes", "one main piece"):
                self.assertTrue(shape_checks[name]["passed"], (name, shape_checks[name]["words"]))
            self.assertFalse(shape_checks["webs in the picture's holes cut away"]["counted"])
            self.assertEqual(len(np.load(job.path("shape.npz"))["triangles"]), len(triangles), "kept as made")
        finally:
            shutil.rmtree(folder)

    def test_a_table_with_a_leg_a_little_short_is_trimmed_level_and_the_sidecar_says_so(self):
        folder = tempfile.mkdtemp()
        try:
            job = pipeline.Job(os.path.join(folder, "table"), "a small round table")
            corners, triangles = shapes.three_legged_table(short_leg=0.025)
            corners = O.set_down(corners, job["size_m"])
            mask = render.silhouette(corners, triangles, 20, 512)
            Image.fromarray(np.where(mask[..., None], 120, 210).astype(np.uint8).repeat(3, -1)).save(
                job.path("picture.png"))
            Image.fromarray(mask.astype(np.uint8) * 255).save(job.path("mask.png"))
            np.savez_compressed(job.path("shape.npz"), corners=corners, triangles=triangles)
            pipeline.finish_stage([job], pipeline.Settings(texture=256, verify=False), report=lambda _: None)
            with open(job.path(f"{job['slug']}.provenance.json")) as file:
                provenance = json.load(file)
            levelled = provenance["levelled"]
            self.assertTrue(levelled["trimmed"], levelled)
            self.assertAlmostEqual(levelled["trimmed_m"], 0.025, delta=0.011)
            self.assertLessEqual(levelled["trimmed_m"], levelled["limit_m"])
            shape_checks = {c["name"]: c["passed"] for c in provenance["checks"]["shape"] if c["counted"]}
            for name in ("every foot on the ground", "base level", "centre over the base", "no holes"):
                self.assertTrue(shape_checks[name], (name, shape_checks))
            with open(job.path(f"{job['slug']}.glb"), "rb") as file:
                document, binary = glb.read_document(file.read())
            positions = glb.accessor_values(document, binary,
                                            document["meshes"][0]["primitives"][0]["attributes"]["POSITION"])
            self.assertAlmostEqual(float(np.asarray(positions)[:, 1].min()), 0.0, places=5)
        finally:
            shutil.rmtree(folder)


if __name__ == "__main__":
    unittest.main()
