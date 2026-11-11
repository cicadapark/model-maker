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
    if subject.kind in ("animal", "character"):
        return subject.kind
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

    def test_every_verdict_is_one_of_six(self):
        self.assertEqual({case["verdict"] for case in cases()},
                         {"make", "not yet", "animal", "character", "several", "refused"})

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

    def test_a_character_is_told_it_comes_after_the_animals(self):
        subject = words.subject_of("a dwarf warrior")
        self.assertEqual(subject.kind, "character")
        self.assertEqual(subject.not_yet, words.CHARACTERS_LATER)
        self.assertIn("after the animals", subject.not_yet)
        self.assertEqual(subject.noun, "warrior")
        self.assertIn("warrior", subject.why["kind"])
        self.assertIn("after the animals", subject.why["kind"])

    def test_an_animal_is_told_it_comes_in_a_later_release(self):
        subject = words.subject_of("a grey wolf")
        self.assertEqual(subject.kind, "animal")
        self.assertEqual(subject.not_yet, words.ANIMALS_LATER)
        self.assertIn("a later release", subject.not_yet)
        self.assertEqual(subject.noun, "wolf")
        self.assertIn("wolf", subject.why["kind"])
        self.assertIn("a later release", subject.why["kind"])

    def test_each_animal_and_each_character_gets_its_own_answer(self):
        for word in words.ANIMAL_WORDS:
            subject = words.subject_of(f"a {word}")
            self.assertEqual((subject.kind, subject.noun, subject.not_yet), ("animal", word, words.ANIMALS_LATER),
                             word)
        for word in words.CHARACTER_WORDS:
            subject = words.subject_of(f"a {word}")
            self.assertEqual((subject.kind, subject.noun, subject.not_yet),
                             ("character", word, words.CHARACTERS_LATER), word)
        self.assertEqual(set(words.ANIMAL_WORDS) & set(words.CHARACTER_WORDS), set())
        for word in ("wolf", "dog", "horse", "bird", "fish", "snake", "serpent", "insect", "animal"):
            self.assertIn(word, words.ANIMAL_WORDS)
        for word in ("dragon", "wyvern", "griffin", "unicorn", "demon", "angel", "ghost", "zombie", "skeleton",
                     "vampire", "werewolf", "mermaid", "fairy", "golem", "robot", "monster", "creature", "beast",
                     "person", "knight", "woman"):
            self.assertIn(word, words.CHARACTER_WORDS)

    def test_the_answers_keep_the_order_and_say_what_is_made_now(self):
        self.assertTrue(words.ANIMALS_LATER.startswith("Animals come in a later release;"))
        self.assertTrue(words.CHARACTERS_LATER.startswith("People and fantasy creatures come after the animals;"))
        for answer in (words.ANIMALS_LATER, words.CHARACTERS_LATER):
            self.assertIn("this one makes 3-D objects", answer)
            self.assertIn("a stone statue of", answer)
            # What each answer suggests is made.
            suggestion = answer.split('"')[1]
            self.assertEqual(verdict(suggestion), "make", suggestion)

    def test_a_likeness_of_either_is_an_object(self):
        for prompt in ("a stone statue of a horse", "a horse statue", "a rocking horse", "a toy dog", "a plush rabbit",
                       "a carved wooden owl", "a chess knight", "a toy knight", "a stone statue of a dragon",
                       "a dragon statue", "a dragon figurine", "a stone statue of a knight on a horse"):
            subject = words.subject_of(prompt)
            self.assertEqual(subject.kind, "object", prompt)
            self.assertEqual(subject.not_yet, "", prompt)

    def test_a_person_and_an_animal_together_are_the_later_release(self):
        for prompt, person in {"a knight on a horse": "knight", "a horse with a knight": "knight",
                               "a knight riding a horse": "knight", "a horse and rider": "rider",
                               "a knight and his horse": "knight", "a wolf and a hunter": "hunter",
                               "a dragon and a horse": "dragon"}.items():
            subject = words.subject_of(prompt)
            self.assertEqual((subject.kind, subject.noun, subject.not_yet),
                             ("character", person, words.CHARACTERS_LATER), prompt)
            self.assertIn("after the animals", subject.why["kind"])

    def test_a_persons_word_that_only_describes_the_animal_leaves_it_an_animal(self):
        for prompt, animal in {"a giant spider": "spider", "a guard dog": "dog", "a king's horse": "horse",
                               "a horse with a giant saddle": "horse", "a horse with a hand guard": "horse",
                               "a cat and a chair": "cat"}.items():
            subject = words.subject_of(prompt)
            self.assertEqual((subject.kind, subject.noun, subject.not_yet), ("animal", animal, words.ANIMALS_LATER),
                             prompt)

    def answers(self, expected, kind, later):
        for prompt, word in expected.items():
            subject = words.subject_of(prompt)
            self.assertEqual((subject.kind, subject.noun, subject.not_yet), (kind, word, later), prompt)

    def test_an_irregular_participle_leaves_the_animal_the_subject(self):
        # The 0.2.0 review, defect 10: "ridden" hid the horse and the knight; the prompt was made as an object.
        self.answers({"a horse ridden by a knight": "knight", "a dog led by a boy": "boy",
                      "a horse held by a knight": "knight"}, "character", words.CHARACTERS_LATER)
        self.answers({"a horse ridden": "horse", "a pony led on a rope": "pony"}, "animal", words.ANIMALS_LATER)
        for word in ("ridden", "held", "drawn", "worn"):
            self.assertIn(word, words.IRREGULAR_PARTICIPLES)
        # Only when it ends the phrase, and never one that says what a likeness is made of.
        for prompt in ("a horse drawn cart", "a horse made of stone", "a cloak worn by a knight"):
            self.assertEqual(words.subject_of(prompt).kind, "object", prompt)
        self.assertNotIn("made", words.IRREGULAR_PARTICIPLES)

    def test_an_animal_after_an_unknown_head_is_not_hidden(self):
        # The 0.2.0 review, defect 10: "a cowboy on a horse" was made as an object.
        self.answers({"a cowboy on a horse": "cowboy", "a jockey on a horse": "jockey",
                      "a jockey riding a horse": "jockey", "a cowboy": "cowboy"}, "character",
                     words.CHARACTERS_LATER)
        # A head of no known word: the creature it is on is asked for.
        self.answers({"a sheriff on a horse": "horse", "a saddle on a horse": "horse", "on a horse": "horse"},
                     "animal", words.ANIMALS_LATER)
        # Not after a known thing ("the head noun wins"), nor after "with" (a picture of one on a thing).
        for prompt in ("a cowboy hat on a horse", "a cowboy hat", "a jockey cap", "a pendant with an owl",
                       "a vase with a horse painted on it", "a hat with a cat", "a cowboy statue on a horse"):
            subject = words.subject_of(prompt)
            self.assertEqual((subject.kind, subject.not_yet), ("object", ""), prompt)

    def test_an_animal_made_a_fantasy_creature_is_a_character(self):
        # The 0.2.0 review, defect 10: these were answered as animals.
        self.answers({"a zombie dog": "zombie", "a skeleton horse": "skeleton", "a robot dog": "robot",
                      "a ghost cat": "ghost", "a demon dog": "demon", "an undead wolf": "undead",
                      "a skeleton horse ridden by a knight": "skeleton"}, "character", words.CHARACTERS_LATER)
        # A size, a task, an owner or a colour leaves it an animal.
        self.answers({"a giant spider": "spider", "a guard dog": "dog", "a king's horse": "horse",
                      "a ghost-white horse": "horse"}, "animal", words.ANIMALS_LATER)

    def test_a_person_only_described_leaves_the_animal_an_animal(self):
        # The 0.2.0 review, defect 10: these were answered as characters; only an animal is asked for.
        self.answers({"a horse and a statue of a knight": "horse", "a horse with armour for a knight": "horse",
                      "a horse for a knight": "horse", "a cat sitting on a throne like a king": "cat",
                      "a horse and a picture of a king": "horse", "a cat like a king": "cat"}, "animal",
                     words.ANIMALS_LATER)
        # A person asked for with it is still the later release.
        self.answers({"a horse with a knight for a king": "knight", "a knight for a horse": "knight"}, "character",
                     words.CHARACTERS_LATER)

    def test_what_is_left_of_an_animal_is_an_object(self):
        # "a fish skeleton" was a character; it is an object, as "a wolf skull" is.
        for prompt in ("a fish skeleton", "a horse skeleton", "the skeleton of a horse", "a wolf skull",
                       "a horse skull"):
            subject = words.subject_of(prompt)
            self.assertEqual((subject.kind, subject.not_yet), ("object", ""), prompt)
        self.answers({"a skeleton": "skeleton", "a skeleton of a knight": "skeleton"}, "character",
                     words.CHARACTERS_LATER)
        self.answers({"a horse with a fish skeleton": "horse"}, "animal", words.ANIMALS_LATER)

    def test_the_answers_decided_before_the_review_hold(self):
        expected = {"a rocking horse": "object", "a dragon statue": "object", "a giant spider": "animal",
                    "a guard dog": "animal", "a king's horse": "animal", "a knight on a horse": "character",
                    "a horse and rider": "character", "a hand guard": "object", "a chess knight": "object"}
        for prompt, kind in expected.items():
            subject = words.subject_of(prompt)
            self.assertEqual(subject.kind, kind, prompt)
            self.assertEqual(bool(subject.not_yet), kind != "object", prompt)

    def test_a_bow_keeps_its_own_answer(self):
        self.assertEqual(words.subject_of("a wooden longbow").not_yet, words.NOT_YET["bow"])

    def test_the_head_phrase_still_ends_where_it_did(self):
        for text, head in {"a candle in a brass holder": "a candle ", "a jug, clay": "a jug",
                           "a pot filled with coins": "a pot ", "a sword": "a sword"}.items():
            self.assertEqual(words._head_phrase(text), head)


if __name__ == "__main__":
    unittest.main()
