"""The bug report file (`model-maker bug-report`) and the remake that checks it (`model-maker reproduce`), on the CPU:
the remake's pipeline is a fake one here, so none of these tests uses a card."""

import contextlib
import io
import json
import os
import random
import shutil
import struct
import tarfile
import tempfile
import unittest
import zipfile
import zlib

from model_maker import bug_report as B

# Built here, not written out, so a search for secrets does not find them in this file.
TOKEN = "hf_" + "Q7x" * 12
ADDRESS = ".".join(["10", "20", "30", "40"])
EMAIL = "alice.smith" + "@" + "example.org"
USER, MACHINE = "alice", "alices-laptop"
HOME = "/home/" + USER
PROMPT = "an old iron chair"


def png_bytes(width, height, pixels, colour=2, filters=None, text=None):
    """A PNG written by hand: 8 bits a channel, each row with the given filter (0..4, cycled), and a tEXt chunk."""
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[colour]
    stride = width * channels
    filters = filters or [0]
    raw, previous = bytearray(), bytes(stride)
    for y in range(height):
        row = pixels[y * stride:(y + 1) * stride]
        kind = filters[y % len(filters)]
        raw.append(kind)
        raw += filtered(kind, row, previous, channels)
        previous = row

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    data = B.PNG_SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, colour, 0, 0, 0))
    if text:
        data += chunk(b"tEXt", b"Comment\0" + text.encode())
    return data + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b"")


def filtered(kind, row, previous, step):
    out = bytearray(len(row))
    for k in range(len(row)):
        a = row[k - step] if k >= step else 0
        b = previous[k]
        c = previous[k - step] if k >= step else 0
        if kind == 0:
            guess = 0
        elif kind == 1:
            guess = a
        elif kind == 2:
            guess = b
        elif kind == 3:
            guess = (a + b) >> 1
        else:
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            guess = a if pa <= pb and pa <= pc else b if pb <= pc else c
        out[k] = (row[k] - guess) & 255
    return bytes(out)


def glb_bytes(document):
    body = json.dumps(document).encode()
    body += b" " * (-len(body) % 4)
    binary = bytes(16)
    return (struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(body) + 8 + len(binary))
            + struct.pack("<I4s", len(body), b"JSON") + body + struct.pack("<I4s", len(binary), b"BIN\0") + binary)


def checks(one_piece=100.0, outline=0.34, passed=True):
    return {"picture": [{"name": "one piece", "value": one_piece, "limit": 97.0, "unit": "%", "passed": True,
                         "counted": True},
                        {"name": "shadow on the backdrop", "value": 2.9, "limit": 6.0, "unit": "%", "passed": True,
                         "counted": False}],
            "shape": [{"name": "outline matches the picture", "value": outline, "limit": 1.9, "unit": "% of height",
                       "passed": passed, "counted": True}]}


def fake_job(out, personal=True, prompt=PROMPT, settings=None):
    """A finished job's folder as the pipeline writes it; with `personal`, the user's home, name, machine, a token, an
    e-mail and an address written into every text, as a user's own runs and edits could leave them."""
    folder = os.path.join(out, "old-iron-chair-1a2b3c4d")
    os.makedirs(os.path.join(folder, "report"))
    os.makedirs(os.path.join(folder, "pictures"))
    where = f"{HOME}/Projects/made/old-iron-chair-1a2b3c4d" if personal else folder
    record = {"id": "f" * 32, "prompt": prompt, "slug": "old-iron-chair", "size_m": 0.9, "status": "passes every check",
              "made_at": "2026-11-10T14:26:17Z", "options": {"size_m": None, "stands": None, "symmetric": None},
              "settings": settings or {"tries": 4, "shape_tries": 2, "triangles": 60000, "texture": 2048,
                                       "lead": "floating", "paint_steps": 20, "picture_rounds": 2,
                                       "touch_up_views": 4, "touch_up_strength": 0.6},
              "picture": {"prompt": f"on a pale grey backdrop: {prompt}", "seed": 141221},
              "shape": {"seed": 43}}
    if personal:
        record.update({"note": f"made on {MACHINE} by {USER} ({EMAIL}) from {ADDRESS}, token {TOKEN}",
                       "kept_at": where, "windows": "C:\\Users\\" + USER + "\\Documents\\made",
                       "mac": f"/Users/{USER}/made/chair", "other": f"/mnt/{USER}-disk/jobs/chair/picture.png",
                       "root": "/root/made/chair"})
    with open(os.path.join(folder, "job.json"), "w") as file:
        json.dump(record, file, indent=1)
    sidecar = {"made_by": "model-maker", "version": "471ab9e", "prompt": prompt,
               "seeds": {"picture": 141221, "shape": 43, "paint": 0},
               "models": [{"id": "flux1-schnell", "version": "1 (hub revision 741f7c3ce8b3)"}],
               "status": "passes every check", "checks": checks()}
    if personal:
        sidecar["where"] = f"{where}/old-iron-chair.glb"
    with open(os.path.join(folder, "old-iron-chair.provenance.json"), "w") as file:
        json.dump(sidecar, file, indent=1)
    page = "<html><body><h1>an old iron chair</h1><img src=\"picture.png\"></body></html>"
    if personal:
        page = page.replace("</body>", f"<p>{where}/report</p><p>{MACHINE}</p><p>file://{HOME}/x/y.png</p></body>")
    with open(os.path.join(folder, "report", "report.html"), "w") as file:
        file.write(page)
    pixels = bytes(random.Random(1).randrange(256) for _ in range(8 * 8 * 3))
    picture = png_bytes(8, 8, pixels, text=f"made in {where}" if personal else None)
    for name in ("picture.png", os.path.join("report", "picture.png"), "mask.png",
                 os.path.join("pictures", "141221.png")):
        with open(os.path.join(folder, name), "wb") as file:
            file.write(picture)
    for name in ("shape.npz", "shape-43.npz", "painted.npz"):
        with open(os.path.join(folder, name), "wb") as file:
            file.write(bytes(1000))
    with open(os.path.join(folder, "old-iron-chair.glb"), "wb") as file:
        file.write(glb_bytes({"asset": {"version": "2.0"}, "extras": {"prompt": prompt,
                                                                       "made": where if personal else "here"}}))
    metrics = os.path.join(out, "metrics")
    os.makedirs(metrics)
    lines = [{"event": "load", "stage": "pictures", "seconds": 20.1},
             {"event": "stage", "stage": "pictures", "job": record["id"], "folder": where, "prompt": prompt,
              "machine": {"card": "a test card", "card_mb": 12288, "driver": "1.2.3", "cpu_threads": 8}},
             {"event": "stage", "stage": "pictures", "job": "e" * 32, "folder": f"{HOME}/Projects/made/other",
              "prompt": "another thing"},
             {"event": "job", "job": record["id"], "folder": where, "status": "passes every check",
              "machine": {"card": "a test card", "card_mb": 12288, "driver": "1.2.3", "cpu_threads": 8}},
             {"event": "run", "jobs": ["e" * 32, record["id"]], "whole_seconds": 600.0}]
    with open(os.path.join(metrics, "jobs.jsonl"), "w") as file:
        file.write("".join(json.dumps(line) + "\n" for line in lines))
    return folder


ABOUT = {"code_version": "33c9a1d", "models": [], "registry_sha256": None, "python": "3.12.0", "torch": "2.5.1",
         "packages": {}, "system": "Linux", "card": {"name": "a test card", "memory_mb": 12288, "driver": "1.2.3"},
         "cpu_threads": 8}


def members_of(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            return {name: archive.read(name) for name in archive.namelist()}
    with tarfile.open(path, "r:gz") as archive:
        return {info.name: archive.extractfile(info).read() for info in archive if info.isfile()}


def texts_of(members):
    """Every member's text: text files whole, a .glb's JSON, a PNG's text chunks."""
    found = {}
    for name, data in members.items():
        if name.endswith(".glb"):
            found[name] = B.glb_json(data)
        elif name.endswith(".png"):
            found[name] = data.decode("latin-1")
        else:
            found[name] = data.decode("utf-8")
    return found


class Folder(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder)
        # No real ledger is read: the job's own (beside it) is given.
        ledger_before = os.environ.get("MODEL_MAKER_LEDGER")
        os.environ["MODEL_MAKER_LEDGER"] = os.path.join(self.folder, "no-ledger")
        self.addCleanup(lambda: os.environ.__setitem__("MODEL_MAKER_LEDGER", ledger_before) if ledger_before
                        else os.environ.pop("MODEL_MAKER_LEDGER", None))

    def made(self, job, out_name="report.zip", **options):
        options.setdefault("about", ABOUT)
        options.setdefault("users", {USER})
        options.setdefault("machines", {MACHINE})
        options.setdefault("environment", {"HF_TOKEN": TOKEN, "HOME": HOME})
        return B.bug_report(job, os.path.join(self.folder, out_name), **options)


class BugReport(Folder):
    def test_no_home_user_machine_token_address_or_email_is_left_in_any_file(self):
        job = fake_job(os.path.join(self.folder, "out"))
        made = self.made(job, with_model=True)
        texts = texts_of(members_of(made.path))
        self.assertIn("bug-report/job/job.json", texts)
        for name, text in texts.items():
            for personal in (HOME, "/home/", "/Users/", "Users\\", USER, MACHINE, TOKEN, ADDRESS, EMAIL, "/mnt/",
                             "/root/", self.folder):
                self.assertNotIn(personal, text, f"{personal!r} left in {name}")
            self.assertNotIn(USER, text.lower(), name)
        record = json.loads(texts["bug-report/job/job.json"])
        self.assertEqual(record["prompt"], PROMPT)
        self.assertEqual(record["kept_at"], "<job>")
        self.assertEqual(record["windows"], "<home>\\Documents\\made")
        self.assertEqual(record["other"], "<path>/picture.png")
        self.assertIn("<token removed>", record["note"])
        self.assertIn("<machine>", record["note"])
        self.assertEqual(json.loads(texts["bug-report/job/old-iron-chair.provenance.json"])["where"],
                         "<job>/old-iron-chair.glb")
        self.assertIn("<job>/report", texts["bug-report/job/report/report.html"])
        self.assertIn("the job's folder", made.replaced)

    def test_it_holds_the_record_sidecar_report_pictures_ledger_and_versions_not_the_in_between_files(self):
        job = fake_job(os.path.join(self.folder, "out"))
        made = self.made(job)
        members = members_of(made.path)
        names = {name[len("bug-report/"):] for name in members}
        self.assertEqual(names, {"README.txt", "about.json", "job/job.json", "job/old-iron-chair.provenance.json",
                                 "job/picture.png", "job/mask.png", "job/report/report.html",
                                 "job/report/picture.png", "job/ledger.jsonl"})
        left = "\n".join(made.left_out)
        for name in ("shape.npz", "shape-43.npz", "painted.npz", "old-iron-chair.glb", "pictures/"):
            self.assertIn(name, left)
        lines = [json.loads(line) for line in members["bug-report/job/ledger.jsonl"].decode().splitlines()]
        self.assertEqual([line["event"] for line in lines], ["stage", "job", "run"])
        about = json.loads(members["bug-report/about.json"])
        self.assertEqual(about["code_version"], "33c9a1d")
        self.assertEqual(about["torch"], "2.5.1")
        self.assertEqual(about["card"]["name"], "a test card")
        self.assertEqual(about["job"]["version"], "471ab9e")
        self.assertEqual(about["job"]["made_on"]["card"], "a test card")
        readme = members["bug-report/README.txt"].decode()
        self.assertIn("same prompt with the same seeds", readme)
        self.assertIn("painted.npz", readme)

    def test_the_pictures_keep_their_pixels_but_lose_their_text(self):
        job = fake_job(os.path.join(self.folder, "out"))
        members = members_of(self.made(job).path)
        with open(os.path.join(job, "picture.png"), "rb") as file:
            original = file.read()
        sent = members["bug-report/job/picture.png"]
        self.assertIn(b"tEXt", original)
        self.assertNotIn(b"tEXt", sent)
        self.assertEqual(B.png_pixels(sent), B.png_pixels(original))

    def test_the_model_only_when_asked_and_its_json_cleaned(self):
        job = fake_job(os.path.join(self.folder, "out"))
        without = members_of(self.made(job).path)
        self.assertNotIn("bug-report/job/old-iron-chair.glb", without)
        with_model = members_of(self.made(job, "with.zip", with_model=True).path)
        glb = with_model["bug-report/job/old-iron-chair.glb"]
        document = json.loads(B.glb_json(glb))
        self.assertEqual(document["extras"]["made"], "<job>")
        self.assertEqual(struct.unpack("<I", glb[8:12])[0], len(glb))
        self.assertEqual(len(glb) % 4, 0)

    def test_a_tar_gz_names_no_user_or_group(self):
        job = fake_job(os.path.join(self.folder, "out"))
        made = self.made(job, "report.tar.gz")
        with tarfile.open(made.path, "r:gz") as archive:
            for info in archive:
                self.assertEqual((info.uname, info.gname, info.uid, info.gid), ("", "", 0, 0))

    def test_a_name_that_is_a_word_of_the_prompt_is_kept_in_the_prompt(self):
        job = fake_job(os.path.join(self.folder, "out"))
        with open(os.path.join(job, "job.json")) as file:
            record = json.load(file)
        record["elsewhere"] = "/srv/iron"
        with open(os.path.join(job, "job.json"), "w") as file:
            json.dump(record, file)
        made = self.made(job, users={USER, "iron"})
        sent = json.loads(members_of(made.path)["bug-report/job/job.json"])
        self.assertEqual(sent["prompt"], PROMPT)
        self.assertEqual(sent["elsewhere"], "/srv/<user>")

    def test_the_users_of_home_folders_in_the_texts_are_replaced_everywhere(self):
        # Made on another machine (its ledger says /home/bob/...): bob is no user here, but is replaced all the same.
        job = fake_job(os.path.join(self.folder, "out"), personal=False)
        with open(os.path.join(job, "job.json")) as file:
            record = json.load(file)
        record["made_in"] = "/home/bob/bob-files/chair"
        record["by"] = "bob"
        with open(os.path.join(job, "job.json"), "w") as file:
            json.dump(record, file)
        sent = members_of(self.made(job, users=set(), machines=set()).path)["bug-report/job/job.json"].decode()
        self.assertNotIn("bob", sent)
        self.assertIn('"by": "<user>"', sent)

    def test_the_scrubbers_check_finds_what_cleaning_removes(self):
        scrubber = B.Scrubber([(HOME, "<home>")], {USER}, {MACHINE}, secrets={TOKEN})
        raw = f"{HOME}/x {MACHINE} {TOKEN} {ADDRESS} {EMAIL} /Users/zed/x /opt/a/b/c"
        found = {what for what, _ in scrubber.leaks(raw)}
        self.assertTrue({"a folder of this machine", "a home folder", "a token", "a network address",
                         "an e-mail address", "an absolute path", "the machine's name", "a user name"} <= found)
        self.assertEqual(scrubber.leaks(scrubber.clean(raw)), [])
        # Web addresses, loopback, units and relative paths are not paths of this machine.
        kept = "https://huggingface.co/a/b/c 127.0.0.1 1.9 % of height report/picture.png </p></div>"
        self.assertEqual(scrubber.clean(kept), kept)

    def test_ipv6_addresses_are_replaced_and_times_of_day_kept(self):
        scrubber = B.Scrubber([], set(), set())
        raw = "from fe80::1ff:fe23:4567:890a and 2001:db8::7 and 2001:0db8:85a3:0000:0000:8a2e:0370:7334"
        self.assertEqual({what for what, _ in scrubber.leaks(raw)}, {"a network address"})
        cleaned = scrubber.clean(raw)
        self.assertEqual(cleaned, "from <address> and <address> and <address>")
        self.assertEqual(scrubber.leaks(cleaned), [])
        kept = "made at 2026-11-10T14:26:17Z, 14:26:17, on ::1; a::before {}"
        self.assertEqual(scrubber.clean(kept), kept)

    def test_a_picture_other_than_a_png_is_left_out(self):
        job = fake_job(os.path.join(self.folder, "out"))
        for name in ("view.jpg", "view.webp"):
            with open(os.path.join(job, "report", name), "wb") as file:
                file.write(b"\xff\xd8\xff\xe1 Exif made by someone")
        made = self.made(job)
        names = set(members_of(made.path))
        self.assertNotIn("bug-report/job/report/view.jpg", names)
        self.assertNotIn("bug-report/job/report/view.webp", names)
        left = "\n".join(made.left_out)
        self.assertIn("report/view.jpg", left)
        self.assertIn("report/view.webp", left)

    def test_the_command_writes_it_and_says_what_it_left_out(self):
        from model_maker import __main__ as command

        job = fake_job(os.path.join(self.folder, "out"))
        said = io.StringIO()
        out = os.path.join(self.folder, "via-command.zip")
        with contextlib.redirect_stdout(said):
            code = command.main(["bug-report", job, "--out", out, "--ledger", os.path.join(self.folder, "out",
                                                                                            "metrics")])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.isfile(out))
        self.assertIn("left out:", said.getvalue())
        self.assertIn("painted.npz", said.getvalue())
        self.assertRegex(said.getvalue(), r"bug report: .*via-command\.zip \(\d+ kB")

    def test_not_a_job_folder_is_refused(self):
        with self.assertRaises(B.Refused):
            B.bug_report(self.folder, os.path.join(self.folder, "x.zip"), about=ABOUT)
        job = fake_job(os.path.join(self.folder, "out"), personal=False)
        with self.assertRaises(B.Refused):
            self.made(job, "report.rar")


class Pixels(unittest.TestCase):
    def test_every_filter_and_colour_type_is_read_back(self):
        rng = random.Random(7)
        for colour, channels in ((0, 1), (2, 3), (4, 2), (6, 4)):
            pixels = bytes(rng.randrange(256) for _ in range(5 * 6 * channels))
            data = png_bytes(5, 6, pixels, colour, filters=[0, 1, 2, 3, 4])
            width, height, rgb = B.png_pixels(data)
            self.assertEqual((width, height, len(rgb)), (5, 6, 5 * 6 * 3))
            if colour == 2:
                self.assertEqual(rgb, pixels)
            if colour == 6:
                self.assertEqual(rgb, b"".join(pixels[k:k + 3] for k in range(0, len(pixels), 4)))
            if colour == 0:
                self.assertEqual(rgb[:3], bytes([pixels[0]] * 3))

    def test_it_reads_what_the_pipeline_writes(self):
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("numpy and PIL are not here")
        array = np.random.default_rng(3).integers(0, 256, (40, 30, 3), dtype=np.uint8)
        array[10:20] = 128
        saved = io.BytesIO()
        Image.fromarray(array).save(saved, format="PNG")
        width, height, rgb = B.png_pixels(saved.getvalue())
        self.assertEqual((width, height), (30, 40))
        self.assertEqual(rgb, array.tobytes())

    def test_mean_difference(self):
        a = png_bytes(2, 1, bytes([0, 0, 0, 10, 10, 10]))
        b = png_bytes(2, 1, bytes([3, 3, 3, 10, 10, 10]))
        self.assertAlmostEqual(B.mean_pixel_difference(a, b)[0], 1.5)
        self.assertEqual(B.mean_pixel_difference(a, a)[0], 0.0)
        self.assertIsNone(B.mean_pixel_difference(a, png_bytes(1, 1, bytes(3)))[0])


class Reproduce(Folder):
    def report_file(self, **job_options):
        job = fake_job(os.path.join(self.folder, "out"), **job_options)
        return self.made(job).path

    def fake_pipeline(self, seen, check_value=0.5, picture_shift=0, status="passes every check", seeds=None):
        """`make(order, out)` as the real pipeline's adapter is called, writing a job as the pipeline would."""

        def make(order, out):
            seen.append((order, os.environ.get("MODEL_MAKER_LEDGER")))
            folder = os.path.join(out, "old-iron-chair-1a2b3c4d")
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "job.json"), "w") as file:
                json.dump({"prompt": order.prompt, "slug": "old-iron-chair"}, file)
            with open(os.path.join(folder, "old-iron-chair.provenance.json"), "w") as file:
                json.dump({"version": "33c9a1d", "seeds": seeds or {"picture": 141221, "shape": 43, "paint": 0},
                           "models": [{"id": "flux1-schnell", "version": "1 (hub revision 741f7c3ce8b3)"}],
                           "status": status, "checks": checks(outline=check_value, passed=check_value <= 1.9)}, file)
            pixels = bytes(random.Random(1).randrange(256) for _ in range(8 * 8 * 3))
            pixels = bytes(min(255, v + picture_shift) for v in pixels)
            with open(os.path.join(folder, "picture.png"), "wb") as file:
                file.write(png_bytes(8, 8, pixels))
            return folder

        return make

    def test_it_remakes_the_reports_order_and_compares_side_by_side(self):
        report = self.report_file()
        seen, said = [], []
        out = os.path.join(self.folder, "remade")
        comparison = B.reproduce(report, out, self.fake_pipeline(seen), our_version="471ab9e",
                                 machine={"card": None, "driver": None}, say=said.append)
        order, ledger_during = seen[0]
        self.assertEqual(order.prompt, PROMPT)
        self.assertEqual((order.size_m, order.stands, order.symmetric), (None, None, None))
        self.assertEqual(order.settings, {"tries": 4, "shape_tries": 2, "triangles": 60000, "texture": 2048,
                                          "lead": "floating", "paint_steps": 20, "picture_rounds": 2,
                                          "touch_up_views": 4, "touch_up_strength": 0.6})
        self.assertEqual(order.seeds, {"picture": 141221, "shape": 43, "paint": 0})
        # The remake's ledger is its own, and ours is as it was after.
        self.assertEqual(ledger_during, os.path.join(os.path.abspath(out), "metrics"))
        self.assertEqual(os.environ["MODEL_MAKER_LEDGER"], os.path.join(self.folder, "no-ledger"))
        self.assertEqual(comparison["picture_mean_difference"], 0.0)
        self.assertEqual(comparison["verdict"], {"reported": "passes every check", "remade": "passes every check"})
        rows = {(r["stage"], r["check"]): r for r in comparison["checks"]}
        self.assertNotIn(("picture", "shadow on the backdrop"), rows)
        self.assertEqual((rows[("shape", "outline matches the picture")]["reported"],
                          rows[("shape", "outline matches the picture")]["remade"]), (0.34, 0.5))
        self.assertEqual(comparison["differs"], [])
        text = "\n".join(said)
        self.assertIn("no card found", text)
        self.assertIn("mean pixel difference 0.00 of 255", text)
        self.assertIn("the remake agrees with the report", text)
        with open(os.path.join(out, "comparison.json")) as file:
            self.assertEqual(json.load(file)["checks"], comparison["checks"])
        self.assertTrue(os.path.isfile(os.path.join(out, "reported", "picture.png")))

    def test_what_differs_is_said(self):
        report = self.report_file()
        said = []
        comparison = B.reproduce(report, os.path.join(self.folder, "remade"),
                                 self.fake_pipeline([], check_value=2.5, picture_shift=4,
                                                    status="fails: outline matches the picture",
                                                    seeds={"picture": 141299, "shape": 43, "paint": 0}),
                                 our_version="33c9a1d", machine={"card": "a test card", "driver": "1.2.3"},
                                 say=said.append)
        self.assertGreater(comparison["picture_mean_difference"], 0)
        self.assertEqual(comparison["differs"], ["the code's version", "the seeds", "the picture", "the verdict",
                                                 "a check's pass or fail"])
        text = "\n".join(said)
        self.assertIn("remade on: a test card (driver 1.2.3)", text)
        self.assertIn("<- passes / fails", text)
        self.assertIn("(they differ)", text)

    def test_a_report_with_settings_out_of_range_or_no_prompt_is_refused(self):
        for settings, words in (({"tries": 500}, "tries"), ({"texture": "big"}, "texture"),
                                ({"lead": "something else"}, "lead")):
            with self.subTest(settings=settings):
                shutil.rmtree(os.path.join(self.folder, "out"), ignore_errors=True)
                report = self.report_file(personal=False, settings=settings)
                with self.assertRaisesRegex(B.Refused, words):
                    B.reproduce(report, os.path.join(self.folder, "remade"), self.fake_pipeline([]), say=lambda _: 0)
        broken = os.path.join(self.folder, "broken.zip")
        with zipfile.ZipFile(broken, "w") as archive:
            archive.writestr("bug-report/job/job.json", json.dumps({"prompt": ""}))
        with self.assertRaisesRegex(B.Refused, "no prompt"):
            B.order_of(B.read_report(broken))

    def test_nothing_in_a_report_is_written_outside_its_folder(self):
        report = os.path.join(self.folder, "crafted.zip")
        with zipfile.ZipFile(report, "w") as archive:
            archive.writestr("bug-report/job/job.json", json.dumps({"prompt": PROMPT, "slug": "x"}))
            archive.writestr("bug-report/../../escaped.txt", "x")
            archive.writestr("../escaped.txt", "x")
        out = os.path.join(self.folder, "deep", "remade")
        B.reproduce(report, out, self.fake_pipeline([]), our_version="x", machine={}, say=lambda _: 0)
        for root, _, names in os.walk(self.folder):
            self.assertNotIn("escaped.txt", names, root)

    def test_the_command_builds_the_pipelines_settings_from_the_report(self):
        from model_maker import __main__ as command

        report = self.report_file()
        calls = []

        class FakeJob:
            folder = None

        def made(prompt, out, size, stands, symmetric, settings, stages):
            calls.append((prompt, settings, stages))
            FakeJob.folder = self.fake_pipeline([])(B.Order(prompt), out)
            return 0, FakeJob

        real = command.made
        command.made = made
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                code = command.main(["reproduce", report, "--out", os.path.join(self.folder, "remade"),
                                     "--no-verify"])
        finally:
            command.made = real
        self.assertEqual(code, 0)
        prompt, settings, stages = calls[0]
        self.assertEqual(prompt, PROMPT)
        self.assertEqual((settings.tries, settings.texture, settings.paint_steps, settings.touch_up_strength,
                          settings.verify), (4, 2048, 20, 0.6, False))
        self.assertEqual(stages, ["pictures", "shape", "paint", "finish"])


if __name__ == "__main__":
    unittest.main()
