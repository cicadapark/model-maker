import base64
import hashlib
import io
import json
import os
import shutil
import signal
import stat
import subprocess
import time
import sys
import tempfile
import textwrap
import unittest

from PIL import Image

from unittest import mock

from model_maker import question_judge as Q

from . import stand_ins

# A stand-in for llama-server: /health, and every question answered "yes" with the share in its environment.
FAKE_SERVER = textwrap.dedent('''\
    import json, math, os, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    # Its own id, for the test's clean-up: on Windows it runs under a launcher (stand_ins.program).
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stand-in.pids"), "a") as pids:
        pids.write(f"{{os.getpid()}}\\n")
    if "--version" in sys.argv:
        print("version: 0 (a stand-in)"); sys.exit(0)
    if os.environ.get("FAKE_ARGV"):
        json.dump(sys.argv[1:], open(os.environ["FAKE_ARGV"], "w"))
    if os.environ.get("FAKE_NO_RESTART") and os.path.exists(os.environ.get("FAKE_DIE_ONCE", "")):
        print("cannot start again"); sys.exit(4)
    port = int(sys.argv[sys.argv.index("--port") + 1])
    share = float(os.environ.get("FAKE_YES", "0.8"))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{{"status":"ok"}}')
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if os.environ.get("FAKE_BODIES"):
                with open(os.environ["FAKE_BODIES"], "a") as said:
                    said.write(json.dumps(body) + "\\n")
            assert body["chat_template_kwargs"] == {{"enable_thinking": False}}
            assert body["cache_prompt"] is False, body
            assert (body["max_tokens"], body["temperature"], body["logprobs"]) == (1, 0, True), body
            content = body["messages"][0]["content"]
            assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
            asked = content[1]["text"]
            assert asked.endswith("Answer yes or no."), asked
            expected = os.environ.get("FAKE_EXPECT")
            if expected:
                assert expected in asked, asked
            if os.environ.get("FAKE_DIE_ONCE") and not os.path.exists(os.environ["FAKE_DIE_ONCE"]):
                open(os.environ["FAKE_DIE_ONCE"], "w").close()
                os._exit(3)
            top = [{{"token": "yes", "logprob": math.log(share)}}, {{"token": "no", "logprob": math.log(1 - share)}}]
            if os.environ.get("FAKE_NEITHER"):
                top = [{{"token": "maybe", "logprob": 0.0}}]
            out = {{"choices": [{{"logprobs": {{"content": [{{"token": top[0]["token"], "top_logprobs": top}}]}}}}]}}
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(out).encode())
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
''').replace("{{", "{").replace("}}", "}")


def strip(width=2880, height=360):
    return Image.new("RGB", (width, height), (200, 180, 160))


class Installed(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.models = tempfile.mkdtemp()

    def install(self):
        folder = os.path.join(self.root, Q.SERVER_FOLDER)
        os.makedirs(folder)
        stand_ins.program(os.path.join(folder, "llama-server"), FAKE_SERVER)
        self.addCleanup(self.end_stand_ins, folder)
        os.makedirs(os.path.join(self.models, Q.MODEL_FOLDER))
        for name in (Q.MODEL_FILE, Q.VISION_FILE):
            open(os.path.join(self.models, Q.MODEL_FOLDER, name), "wb").close()
        empty = hashlib.sha256(b"").hexdigest()
        patcher = mock.patch.object(Q, "registry_sums", return_value={Q.MODEL_FILE: empty, Q.VISION_FILE: empty})
        patcher.start()
        self.addCleanup(patcher.stop)
        return Q.found(self.root, self.models)

    @staticmethod
    def end_stand_ins(folder):
        try:
            with open(os.path.join(folder, "stand-in.pids")) as file:
                stand_ins.end(int(line) for line in file if line.strip())
        except OSError:
            pass

    def test_missing_it_is_not_judged_and_counts_for_nothing(self):
        where = Q.found(self.root, self.models)
        self.assertFalse(where.usable)
        self.assertIn("./setup.sh --with-question-judge", where.why)
        check, record = Q.object_check(None, strip(), "a garden hoe", required=False)
        self.assertTrue(check.passed)
        self.assertFalse(check.counted)
        self.assertFalse(record["judged"])

    def test_missing_in_a_run_that_requires_it_fails(self):
        check, _ = Q.object_check(None, strip(), "a garden hoe", required=True)
        self.assertFalse(check.passed)
        self.assertTrue(check.counted)

    def test_a_judged_object_passes_from_the_limit_and_fails_under_it(self):
        where = self.install()
        self.assertTrue(where.usable, where.why)
        for share, passes in ((0.8, True), (0.3, False)):
            os.environ["FAKE_YES"] = str(share)
            try:
                with Q.Judge(where, threads=1) as judge:
                    check, record = Q.object_check(judge, strip(), "a garden hoe", required=False)
            finally:
                del os.environ["FAKE_YES"]
            self.assertEqual(check.passed, passes, check.words)
            # Counted (14 Nov 2026), asked about the thing the words name.
            self.assertTrue(check.counted)
            self.assertTrue(record["counted"])
            self.assertAlmostEqual(record["yes"], share, places=6)
            self.assertIn("Is it a hoe?", record["question"])
            if not passes:
                self.assertIn("does not look like a hoe", check.words)
            self.assertEqual(record["llama_tag"], Q.LLAMA_TAG)

    def test_an_answer_without_yes_or_no_is_not_judged(self):
        where = self.install()
        os.environ["FAKE_NEITHER"] = "1"
        try:
            with Q.Judge(where, threads=1) as judge:
                check, record = Q.object_check(judge, strip(), "a garden hoe", required=False)
        finally:
            del os.environ["FAKE_NEITHER"]
        self.assertFalse(record["judged"])
        self.assertFalse(check.counted)

    def test_the_pictures_are_a_note_the_others_asked_only_under_the_limit(self):
        where = self.install()
        pictures = {seed: Image.new("RGB", (64, 64)) for seed in ("1", "2", "3", "4")}
        for share, asked in ((0.9, 1), (0.3, 4)):
            os.environ["FAKE_YES"] = str(share)
            try:
                with Q.Judge(where, threads=1) as judge:
                    note = Q.picture_note(judge, pictures, "2", "a sickle")
            finally:
                del os.environ["FAKE_YES"]
            self.assertEqual(len(note["tries"]), asked)
            self.assertEqual(note["none_reaches"], share < Q.LIMIT)

    def test_the_record_says_what_answered_and_how(self):
        where = self.install()
        with mock.patch.dict(os.environ, {"FAKE_YES": "0.8", "FAKE_EXPECT": "Is it a hoe?"}):
            with Q.Judge(where, threads=1) as judge:
                _, record = Q.object_check(judge, strip(), "a garden hoe", required=False)
        self.assertAlmostEqual(record["p_yes"], 0.8, places=6)
        self.assertAlmostEqual(record["p_no"], 0.2, places=6)
        self.assertEqual(record["first_token"], "yes")
        empty = hashlib.sha256(b"").hexdigest()
        self.assertEqual((record["model_sha256"], record["vision_sha256"]), (empty, empty))
        self.assertFalse(record["restarted"])
        self.assertTrue(record["sums_are_the_registry_s"])
        self.assertGreater(record["server_memory_mb"]["peak"], 0)
        self.assertEqual(set(record["server_memory_mb"]), {"peak", "now", "own", "model_files"})

    def test_model_files_not_the_registry_s_are_not_asked(self):
        where = self.install()
        with mock.patch.object(Q, "registry_sums", return_value={Q.MODEL_FILE: "0" * 64, Q.VISION_FILE: "0" * 64}):
            with Q.Judge(where, threads=1) as judge:
                check, record = Q.object_check(judge, strip(), "a garden hoe", required=True)
        self.assertFalse(record["judged"])
        self.assertIn("not the registry's", record["why"])
        self.assertFalse(check.passed)
        self.assertTrue(check.counted, "a run that requires the judge fails a job it could not ask")

    def test_it_runs_on_the_cpu_and_listens_on_this_computer_only(self):
        where = self.install()
        said = os.path.join(self.root, "argv.json")
        with mock.patch.dict(os.environ, {"FAKE_ARGV": said}):
            with Q.Judge(where, threads=1) as judge:
                log = judge.log
                self.assertTrue(os.path.exists(log))
        with open(said) as file:
            argv = json.load(file)
        self.assertEqual(argv[argv.index("-ngl") + 1], "0")
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
        self.assertIn("--no-mmproj-offload", argv)
        self.assertFalse(os.path.exists(log), "its log is removed when it stops")
        self.assertEqual(Q.server_runs(where), (True, "version: 0 (a stand-in)"))

    def test_it_keeps_no_earlier_prompt(self):
        # llama.cpp keeps earlier prompts (8 GiB by default): the 9B's server grew from about 10 GB to 17.5 GB over a
        # long batch. Neither the server's cache nor any request keeps one.
        where = self.install()
        said, bodies = os.path.join(self.root, "argv.json"), os.path.join(self.root, "bodies.jsonl")
        with mock.patch.dict(os.environ, {"FAKE_ARGV": said, "FAKE_BODIES": bodies}):
            with Q.Judge(where, threads=1) as judge:
                data = Q.as_data(strip())
                for question in (Q.OBJECT_QUESTION.format(prompt="a hoe"), Q.PICTURE_QUESTION.format(prompt="a hoe")):
                    judge.ask(data, question)
        with open(said) as file:
            argv = json.load(file)
        self.assertEqual(argv[argv.index("--cache-ram") + 1], "0")
        self.assertNotIn("--no-repack", argv, "repacking is kept: without it the answers move")
        with open(bodies) as file:
            asked = [json.loads(line) for line in file]
        self.assertEqual(len(asked), 2)
        self.assertTrue(all(body["cache_prompt"] is False for body in asked), asked)

    def test_a_server_that_dies_and_cannot_start_again_is_a_job_not_judged(self):
        where = self.install()
        marker = os.path.join(self.root, "died")
        with mock.patch.dict(os.environ, {"FAKE_DIE_ONCE": marker, "FAKE_NO_RESTART": "1"}):
            with Q.Judge(where, threads=1) as judge:
                check, record = Q.object_check(judge, strip(), "a garden hoe", required=False)
                self.assertFalse(record["judged"])
                self.assertIn("could not start again", record["why"])
                self.assertFalse(check.counted)
                # The next job is not restarted again (once a stage), and says it was not the one restarted.
                _, second = Q.object_check(judge, strip(), "a garden hoe", required=False)
        self.assertTrue(record["restarted"])
        self.assertFalse(second["judged"])
        self.assertFalse(second["restarted"])

    def test_a_server_that_dies_is_started_once_more_and_asked_again(self):
        where = self.install()
        marker = os.path.join(self.root, "died")
        with mock.patch.dict(os.environ, {"FAKE_YES": "0.8", "FAKE_DIE_ONCE": marker}):
            with Q.Judge(where, threads=1) as judge:
                check, record = Q.object_check(judge, strip(), "a garden hoe", required=False)
        self.assertTrue(os.path.exists(marker))
        self.assertTrue(record["judged"], record)
        self.assertTrue(record["restarted"])
        self.assertTrue(check.passed)

    def test_a_server_that_cannot_start_leaves_nothing_running(self):
        where = self.install()
        stand_ins.program(os.path.join(self.root, Q.SERVER_FOLDER, "llama-server"),
                          "import sys\nprint('no model here', flush=True)\nsys.exit(7)\n")
        judge = Q.Judge(where, threads=1)
        with self.assertRaises(RuntimeError) as said:
            judge.start()
        self.assertIn("no model here", str(said.exception))
        self.assertIsNone(judge.process)

    def test_the_server_goes_when_model_maker_is_killed(self):
        # A parent killed outright (no `finally`) takes its server with it (PR_SET_PDEATHSIG).
        where = self.install()
        pid_file = os.path.join(self.root, "server.pid")
        parent = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""\
            import sys, time
            sys.path.insert(0, {os.path.dirname(os.path.dirname(Q.__file__))!r})
            from model_maker import question_judge as Q
            judge = Q.Judge(Q.found({self.root!r}, {self.models!r}), threads=1)
            judge.start()
            open({pid_file!r}, "w").write(str(judge.process.pid))
            time.sleep(600)
        """)])
        try:
            deadline = time.time() + 60
            while not os.path.exists(pid_file) and time.time() < deadline:
                time.sleep(0.2)
            time.sleep(0.2)
            with open(pid_file) as file:
                server = int(file.read())
            parent.kill()
            parent.wait()
            deadline = time.time() + 10
            while time.time() < deadline and stand_ins.alive(server):
                time.sleep(0.2)
            self.assertFalse(stand_ins.alive(server), "the server outlived model-maker")
        finally:
            parent.kill()

    def test_the_server_is_stopped_when_the_stage_ends(self):
        where = self.install()
        with Q.Judge(where, threads=1) as judge:
            process = judge.process
        self.assertIsNotNone(process.poll())


class AskedWords(unittest.TestCase):
    def test_the_thing_alone_or_the_whole_prompt(self):
        self.assertEqual(Q.asked_words("an old iron sword"), "a sword")
        self.assertEqual(Q.asked_words("a garden hoe"), "a hoe")
        self.assertEqual(Q.asked_words("a wooden chess board"), "a chess board")
        self.assertEqual(Q.asked_words("an iron birdcage"), "an iron birdcage")

    def test_a_name_of_two_things_is_asked_whole(self):
        self.assertEqual(Q.asked_words("a bow saw"), "a bow saw")
        self.assertEqual(Q.asked_words("an old wooden bow saw"), "a bow saw")
        self.assertEqual(Q.asked_words("a stone bench"), "a bench")
        for prompt, asked in {"a bone knife": "a knife", "a rock chair": "a chair", "a horn bow": "a bow",
                              "a rope chair": "a chair"}.items():
            self.assertEqual(Q.asked_words(prompt), asked, prompt)
        self.assertEqual(Q.asked_words("a wooden treasure chest"), "a chest")
        self.assertEqual(Q.asked_words("a pair of wireless headphones"), "a pair of headphones")


class AsTheTrialSentIt(unittest.TestCase):
    def test_eight_sides_become_four_by_two_at_most_1024_wide(self):
        data = Q.as_data(strip())
        image = Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1])))
        self.assertEqual(image.size, (1024, 512))

    def test_a_picture_is_at_most_512_a_side(self):
        data = Q.as_data(Image.new("RGB", (1024, 1024)))
        image = Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1])))
        self.assertEqual(image.size, (512, 512))


if __name__ == "__main__":
    unittest.main()
