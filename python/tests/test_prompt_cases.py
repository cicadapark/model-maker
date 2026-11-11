"""How prompts are read: words.py against the shared cases (prompt_cases.json), which a program that checks prompts
before the pipeline runs can test its own reading against, so the two cannot drift apart."""

import json
import os
import unittest

from model_maker import words

CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_cases.json")


def cases():
    with open(CASES) as file:
        return json.load(file)["cases"]


def verdict(prompt):
    subject = words.subject_of(prompt)
    if subject.kind == "character":
        return "character"
    if subject.several:
        return "several"
    return "not yet" if subject.not_yet else "make"


class ThePipelineReadsTheSharedCases(unittest.TestCase):
    def test_every_case_but_the_refusals_of_a_program_in_front(self):
        checked = 0
        for case in cases():
            if case["verdict"] == "refused":
                continue
            self.assertEqual(verdict(case["prompt"]), case["verdict"], case["prompt"])
            checked += 1
        self.assertGreater(checked, 50)

    def test_every_verdict_is_one_of_five(self):
        self.assertEqual({case["verdict"] for case in cases()}, {"make", "not yet", "character", "several", "refused"})

    def test_several_things_are_answered_with_one_to_ask_for(self):
        for prompt, one in {"a pair of leather boots": "a leather boot", "an old pair of boots": "an old boot",
                            "two wooden chairs": "a wooden chair", "boots": "a boot", "knives": "a knife",
                            "three iron swords with gold hilts": "an iron sword with gold hilts",
                            "a set of three knives": "a knife", "a sword and a shield": "a sword",
                            "leather boots, brown": "a leather boot, brown", "two boxes": "a box",
                            "a pair of earrings": "an earring"}.items():
            subject = words.subject_of(prompt)
            self.assertEqual(subject.not_yet, words.SEVERAL_THINGS.format(one=one), prompt)
            self.assertIn("one object at a time", subject.not_yet)
            self.assertIn(subject.several, subject.why["kind"])
            # What it suggests is made.
            self.assertEqual(verdict(one), "make", one)

    def test_several_things_with_no_one_to_suggest_say_so(self):
        subject = words.subject_of("a set of armour")
        self.assertEqual(subject.not_yet, words.SEVERAL_THINGS_NO_ONE)

    def test_why_several_names_the_words(self):
        expected = {"a pair of boots": "'pair of'", "two wooden chairs": "the count 'two'",
                    "leather boots": "the plural 'boots'", "a sword and a shield": "'sword' and 'shield'"}
        for prompt, why in expected.items():
            self.assertEqual(words.subject_of(prompt).several, why, prompt)

    def test_a_character_is_told_it_comes_in_the_second_release(self):
        subject = words.subject_of("a dwarf warrior")
        self.assertEqual(subject.not_yet, words.CHARACTERS_LATER)
        self.assertIn("second release", subject.not_yet)
        self.assertEqual(subject.noun, "warrior")
        self.assertIn("warrior", subject.why["kind"])

    def test_a_bow_keeps_its_own_answer(self):
        self.assertEqual(words.subject_of("a wooden longbow").not_yet, words.NOT_YET["bow"])

    def test_the_head_phrase_still_ends_where_it_did(self):
        for text, head in {"a candle in a brass holder": "a candle ", "a jug, clay": "a jug",
                           "a pot filled with coins": "a pot ", "a sword": "a sword"}.items():
            self.assertEqual(words._head_phrase(text), head)


if __name__ == "__main__":
    unittest.main()
