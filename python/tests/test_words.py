import os
import unittest

from model_maker import words

TEN_OBJECTS = [
    "an old looking chair", "a rusty iron lantern", "a wooden barrel with iron hoops", "a three-legged stool",
    "a leather-bound book, closed", "a clay jug", "a wooden treasure chest", "a candle in a brass holder",
    "a small round table", "a broom",
]
SIX_WEAPONS = [
    "an old iron sword", "a double-bitted battle axe", "a wooden longbow", "a round wooden shield with an iron boss",
    "a spear with a leaf-shaped head", "a war hammer",
]


class TheTrialPromptsAreReadAsMeant(unittest.TestCase):
    def test_every_object_is_an_object_and_every_weapon_a_weapon(self):
        for prompt in TEN_OBJECTS:
            self.assertEqual(words.subject_of(prompt).kind, "object", prompt)
        for prompt in SIX_WEAPONS:
            self.assertEqual(words.subject_of(prompt).kind, "weapon", prompt)

    def test_the_head_noun_decides(self):
        cases = {"a candle in a brass holder": "candle", "a wooden barrel with iron hoops": "barrel",
                 "a double-bitted battle axe": "battle axe", "a war hammer": "war hammer",
                 "a wooden longbow": "longbow", "a wooden treasure chest": "chest",
                 "a chest with a sword carved on its lid": "chest", "a sword in a leather scabbard": "sword"}
        for prompt, noun in cases.items():
            self.assertEqual(words.subject_of(prompt).noun, noun, prompt)

    def test_things_that_stand_are_checked_standing_and_a_broom_is_not(self):
        for prompt in TEN_OBJECTS[:-1]:
            self.assertTrue(words.subject_of(prompt).stands, prompt)
        broom = words.subject_of("a broom")
        self.assertFalse(broom.stands)
        self.assertTrue(broom.handled, "a broom's handle is checked as a haft")
        for prompt in SIX_WEAPONS:
            self.assertFalse(words.subject_of(prompt).stands, prompt)

    def test_symmetry_is_asked_only_where_the_words_imply_it(self):
        expected = {"an old iron sword": "across", "a double-bitted battle axe": "across", "a wooden longbow": "along",
                    "a round wooden shield with an iron boss": "mirror", "a spear with a leaf-shaped head": "across",
                    "a war hammer": "", "a broom": "", "a wooden treasure chest": "mirror",
                    "a broken wooden chair": "", "a battle axe": ""}
        for prompt, symmetric in expected.items():
            self.assertEqual(words.subject_of(prompt).symmetric, symmetric, prompt)

    def test_weapon_kinds_and_lengths(self):
        sword = words.subject_of("an old iron sword")
        self.assertEqual((sword.weapon_kind, sword.size_m), ("blade", 0.95))
        self.assertEqual(words.subject_of("a wooden longbow").weapon_kind, "bow")
        self.assertEqual(words.subject_of("a spear with a leaf-shaped head").weapon_kind, "spear")
        self.assertEqual(words.subject_of("a curved sabre").weapon_kind, "curved blade")
        self.assertEqual(words.subject_of("a red bow tie").kind, "object")

    def test_every_choice_says_why(self):
        for prompt in TEN_OBJECTS + SIX_WEAPONS:
            subject = words.subject_of(prompt)
            self.assertIn("kind", subject.why, prompt)
            self.assertIn("symmetric", subject.why, prompt)

    def test_picture_words_lead_with_the_light_and_keep_the_prompt(self):
        said = words.picture_prompt(words.subject_of("  a clay   jug. "))
        self.assertTrue(said.startswith(words.LEAD))
        self.assertIn("a clay jug,", said)
        self.assertIn("upright", words.picture_prompt(words.subject_of("an old iron sword")))

    def test_a_head_on_a_haft_is_painted_without_a_point_and_a_blade_or_spear_keeps_it(self):
        # Asked for "its point or head at the top", every war hammer the trials painted had a spike and an axe-like
        # blade (docs/objects.md, "War hammers"). A sword's and a spear's tip is a point: their words are unchanged.
        for prompt in ("a war hammer", "a battle axe", "a double-bitted battle axe", "a hatchet", "a spiked mace"):
            said = words.picture_prompt(words.subject_of(prompt))
            self.assertIn(words.HAFTED_VIEW, said, prompt)
            self.assertNotIn("point", said, prompt)
        for prompt in ("an old iron sword", "a spear with a leaf-shaped head"):
            self.assertIn(words.WEAPON_VIEW, words.picture_prompt(words.subject_of(prompt)), prompt)

    def test_the_war_hammer_family_is_told_what_its_head_looks_like(self):
        for prompt, noun in (("a war hammer", "war hammer"), ("a heavy maul", "maul"), ("a wooden mallet", "mallet"),
                             ("a sledgehammer", "sledgehammer"), ("a rusty sledge hammer", "sledge hammer")):
            subject = words.subject_of(prompt)
            self.assertEqual((subject.noun, subject.weapon_kind), (noun, "hafted"), prompt)
            said = words.picture_prompt(subject)
            self.assertIn(f"{subject.prompt}, {words.HEAD_PICTURE_WORDS['hammer']}, {words.HAFTED_VIEW}", said)
        # No material is named, so a wooden mallet stays wooden; an axe is not told about a hammer's face.
        self.assertNotIn("steel", words.HEAD_PICTURE_WORDS["hammer"])
        self.assertNotIn("striking face", words.picture_prompt(words.subject_of("a battle axe")))

    def test_a_mace_and_a_club_are_told_they_are_short_with_a_heavy_head(self):
        # With the view alone every try was a long rod or a cane (docs/trials/2026-11-11-030-diagnosis.md).
        for prompt, kind in (("a short iron mace", "mace"), ("a spiked mace", "mace"), ("a wooden club", "club"),
                             ("a knotted oak club", "club")):
            subject = words.subject_of(prompt)
            self.assertEqual(subject.weapon_kind, "hafted", prompt)
            said = words.picture_prompt(subject)
            self.assertIn(f"{subject.prompt}, {words.HEAD_PICTURE_WORDS[kind]}, {words.HAFTED_VIEW}", said)
            self.assertIn("short", words.HEAD_PICTURE_WORDS[kind])
        # No material named: an iron mace and an oak club keep their own.
        for kind in ("mace", "club"):
            for material in ("iron", "steel", "wood", "oak"):
                self.assertNotIn(material, words.HEAD_PICTURE_WORDS[kind])
        self.assertNotIn("cudgel", words.picture_prompt(words.subject_of("a spiked mace")))

    def test_a_thing_known_only_loosely_by_its_name_is_described(self):
        for prompt, name in (("a wooden spinning wheel", "spinning wheel"), ("an old spinning wheel", "spinning wheel"),
                             ("a wooden water wheel", "water wheel"), ("a mill water wheel", "water wheel")):
            subject = words.subject_of(prompt)
            said = words.picture_prompt(subject)
            self.assertIn(f"{subject.prompt}, {words.THING_PICTURE_WORDS[name]}, {words.OBJECT_VIEW}", said)
        for prompt in ("a wooden cart wheel", "a wheelbarrow", "a wooden chair", "an iron cauldron",
                       "a stone water wheelhouse"):
            said = words.picture_prompt(words.subject_of(prompt))
            self.assertFalse(any(described in said for described in words.THING_PICTURE_WORDS.values()), prompt)

    def test_a_lid_or_cover_is_shut_unless_the_words_open_it(self):
        expected = {"a leather-bound book, closed": "closed", "a wooden treasure chest": "closed",
                    "an open wooden chest": "open", "a book lying open": "open", "an oak wardrobe": "closed",
                    "a small box, its lid ajar": "open", "a clay jug": "", "an old iron sword": "",
                    "a wooden crate": "closed"}
        for prompt, state in expected.items():
            subject = words.subject_of(prompt)
            self.assertEqual(subject.state, state, prompt)
            if state:
                self.assertIn("state", subject.why, prompt)

    def test_the_picture_words_say_the_state_and_the_judge_has_its_contrast(self):
        chest = words.subject_of("a wooden treasure chest")
        self.assertIn("its lid shut", words.picture_prompt(chest))
        wanted, unwanted = words.state_texts(chest)
        self.assertTrue(wanted and unwanted)
        self.assertTrue(all("open" not in text for text in wanted))
        self.assertTrue(all("open" in text for text in unwanted))
        opened = words.subject_of("an open wooden chest")
        self.assertIn("its lid open", words.picture_prompt(opened))
        self.assertEqual(words.state_texts(opened), (unwanted, wanted))
        jug = words.subject_of("a clay jug")
        self.assertEqual(words.state_texts(jug), ([], []))
        self.assertNotIn("shut", words.picture_prompt(jug))

    def test_the_judge_describes_what_each_state_looks_like_in_pairs(self):
        for prompt in ("a leather-bound book, closed", "a wooden treasure chest", "an oak wardrobe"):
            subject = words.subject_of(prompt)
            shut, opened = words.state_texts(subject)
            self.assertEqual(len(shut), len(opened), prompt)
            for closed_words, open_words in zip(shut, opened):
                self.assertIn(subject.noun, closed_words)
                self.assertIn(subject.noun, open_words)
                self.assertNotIn("open", closed_words)
                self.assertIn("open", open_words)
                # a description, not the state's name alone: the bare names scored an open book 0.88 shut
                self.assertIn(",", closed_words)
                self.assertIn(",", open_words)

    def test_the_state_judges_labelled_pictures_are_well_formed(self):
        import tomllib

        path = os.path.join(os.path.dirname(__file__), "..", "..", "tools", "trial", "state-cases.toml")
        with open(path, "rb") as file:
            pictures = tomllib.load(file)["picture"]
        self.assertEqual(len({p["name"] for p in pictures}), len(pictures))
        self.assertEqual(len({p["source"] for p in pictures}), len(pictures))
        for picture in pictures:
            self.assertIn(picture["shows"], ("open", "closed"))
            self.assertTrue(words.subject_of(picture["prompt"]).state, picture["name"])
        caught = [p for p in pictures if p["name"] == "book-3060-trial-4-935320"]
        self.assertEqual([(p["shows"], p.get("known_miss", False)) for p in caught], [("open", False)])

    def test_slugs_are_letters_and_dashes(self):
        self.assertEqual(words.slug_of("an old looking chair"), "chair")
        self.assertEqual(words.slug_of("a leather-bound book, closed"), "leather-bound-book-closed")
        self.assertRegex(words.slug_of("a candle in a brass holder!"), r"^[a-z0-9]+(-[a-z0-9]+)*$")

    def test_a_long_prompt_is_cut(self):
        self.assertLessEqual(len(words.cleaned("a chair " * 200)), words.MOST_PROMPT_CHARACTERS)

    def test_a_long_word_makes_a_short_slug(self):
        # A word of 256 letters or more gave "File name too long" at the job's folder.
        for prompt in ("a " + "x" * 300, "a " + "verylongword" * 25 + " chair", "a chair " + "ornate " * 40):
            slug = words.slug_of(prompt)
            self.assertLessEqual(len(slug), words.MOST_SLUG_CHARACTERS, prompt)
            self.assertRegex(slug, r"^[a-z0-9]+(-[a-z0-9]+)*$")

    def test_an_empty_prompt_is_not_made(self):
        # An empty prompt was accepted and made "something" in the folder "object".
        for prompt in ("", "   ", " . , ", "<b></b>"):
            subject = words.subject_of(prompt)
            self.assertEqual(subject.not_yet, words.EMPTY_PROMPT, repr(prompt))


class TrickyPromptsAreReadRight(unittest.TestCase):
    """A creature's word before a participle, and a negation before "open" (the verdicts, make or character, are in
    the shared cases)."""

    def test_a_creature_word_before_a_participle_and_a_thing_names_the_thing(self):
        cases = {"a giant spiked mace": "mace", "a king sized bed": "bed", "a cat shaped lamp": "lamp",
                 "a wolf headed staff": "staff", "a dragon scaled shield": "shield", "an eagle crested helmet": "helmet",
                 "a giant rusted iron key": "key", "a giant red mushroom": "mushroom"}
        for prompt, noun in cases.items():
            subject = words.subject_of(prompt)
            self.assertNotIn(subject.kind, ("animal", "character"), prompt)
            self.assertEqual(subject.noun, noun, prompt)

    def test_an_ending_in_ed_or_ing_on_no_stem_is_no_participle(self):
        for word in ("red", "sled", "shed", "bed", "king", "wing", "ring", "string", "sling"):
            self.assertIsNone(words.PARTICIPLE.fullmatch(word), word)
        for word in ("sleeping", "standing", "spiked", "headed", "flying", "rusted"):
            self.assertIsNotNone(words.PARTICIPLE.fullmatch(word), word)

    def test_a_creature_doing_something_is_still_a_character_or_an_animal(self):
        for prompt in ("a dragon sleeping", "a dragon sleeping on a chest", "a giant", "a knight kneeling"):
            self.assertEqual(words.subject_of(prompt).kind, "character", prompt)
        for prompt in ("a big black bear standing", "a horse grazing"):
            self.assertEqual(words.subject_of(prompt).kind, "animal", prompt)

    def test_a_denied_opening_leaves_it_shut_and_symmetric(self):
        for prompt in ("a book, never opened", "a chest that cannot be opened", "a chest, not open"):
            subject = words.subject_of(prompt)
            self.assertEqual(subject.state, "closed", prompt)
            self.assertEqual(subject.symmetric, "mirror", prompt)
            self.assertNotIn("open", words.picture_prompt(subject).split(subject.prompt, 1)[1].split(",")[1], prompt)
        self.assertEqual(words.subject_of("a book, opened").state, "open")
        self.assertEqual(words.subject_of("an open chest, never locked").state, "open")

    def test_upright_words_are_whole_words(self):
        self.assertTrue(words.subject_of("a ribbon edged plate").lies)
        self.assertFalse(words.subject_of("a plate standing on edge").lies)
        self.assertFalse(words.subject_of("a book standing upright").lies)



class QualitiesTheWordsName(unittest.TestCase):
    """A quality the prompt gives the whole thing is judged in its picture (subject_judge); one that describes a part,
    is denied or contradicts another is not."""

    def test_a_quality_in_the_head_phrase_is_read(self):
        self.assertEqual(words.subject_of("a short curved dagger").qualities, ["curved"])
        self.assertEqual(words.subject_of("a small round table").qualities, ["round"])
        self.assertEqual(words.subject_of("a circular wooden tray").qualities, ["round"])
        # A word in -ing after it names the thing's use, not a part: the thing is still curved.
        self.assertEqual(words.subject_of("a curved steel dagger").qualities, ["curved"])
        self.assertEqual(words.subject_of("a curved hunting dagger").qualities, ["curved"])

    def test_a_curved_knife_is_a_curved_blade_whose_curve_is_not_judged(self):
        knife = words.subject_of("a curved hunting knife")
        self.assertEqual(knife.weapon_kind, "curved blade")
        self.assertEqual(knife.qualities, [])
        self.assertIn(words.CURVED_BLADE_WORDS, words.picture_prompt(knife))
        self.assertEqual(words.subject_of("a hunting knife").weapon_kind, "blade")
        self.assertEqual(words.subject_of("a knife, not curved").weapon_kind, "blade")
        self.assertIn("curved", words.subject_of("a short curved dagger").why["qualities"])

    def test_a_quality_of_a_part_denied_or_contradicted_is_not(self):
        for prompt in ("a chest with a round lid", "a round-bottomed flask", "a sword, not curved",
                       "a round square table", "a curved-handled cane", "an old looking chair", "a clay jug"):
            self.assertEqual(words.subject_of(prompt).qualities, [], prompt)

    def test_a_curved_blade_is_checked_as_a_sabre_is(self):
        dagger = words.subject_of("a short curved dagger")
        self.assertEqual(dagger.weapon_kind, "curved blade")
        self.assertEqual(dagger.symmetric, "")
        self.assertEqual(dagger.size_m, 0.4)
        self.assertIn("curved blade", dagger.why["kind"])
        self.assertEqual(words.subject_of("a short dagger").weapon_kind, "blade")

    def test_the_judge_has_a_described_pair_for_each_quality(self):
        for quality, pairs in words.QUALITY_JUDGE_TEXTS.items():
            self.assertIn(quality, words.QUALITY_OPPOSITES)
            self.assertGreaterEqual(len(pairs), 2, quality)
            for asked, other in pairs:
                self.assertIn("{noun}", asked)
                self.assertIn("{noun}", other)
        for word, quality in words.QUALITY_WORDS.items():
            self.assertIn(quality, words.QUALITY_JUDGE_TEXTS, word)
        [(quality, asked, other)] = words.quality_texts(words.subject_of("a curved bronze dagger"))
        self.assertEqual(quality, "curved")
        self.assertTrue(all(text.startswith("a curved dagger") or "dagger with a curved" in text for text in asked))
        self.assertTrue(all("straight" in text for text in other))
        self.assertEqual(len(asked), len(other))

    def test_an_unknown_thing_is_named_object_in_the_judges_words(self):
        [(_, asked, _)] = words.quality_texts(words.subject_of("a round gizmo"))
        self.assertIn("object", asked[0])

    def test_a_curved_blade_is_painted_with_its_curve_in_view(self):
        for prompt in ("a curved bronze dagger", "a curved steel sabre"):
            said = words.picture_prompt(words.subject_of(prompt))
            self.assertIn(words.CURVED_BLADE_WORDS, said)
            self.assertIn(words.CURVED_BLADE_VIEW, said)
            self.assertNotIn(words.WEAPON_VIEW, said)
        # A straight blade and a thing with no quality are painted with the words they had.
        sword = words.picture_prompt(words.subject_of("an old iron sword"))
        self.assertIn(words.WEAPON_VIEW, sword)
        self.assertNotIn("arc", sword)

    def test_a_quality_the_kind_already_has_is_not_judged_again(self):
        for prompt in ("a curved steel sabre", "a curved iron sickle", "a curved scimitar"):
            self.assertEqual(words.subject_of(prompt).qualities, [], prompt)

    def test_a_curved_blade_is_judged_as_a_curved_blade_not_against_one(self):
        asked, others = words.subject_contrast(words.subject_of("a curved bronze dagger"))
        self.assertIn("a curved blade with a hilt", asked[1])
        self.assertNotIn("a sabre", others)
        self.assertNotIn("a sword", others)
        # A straight blade is still judged against a sabre.
        _, others = words.subject_contrast(words.subject_of("an old iron sword"))
        self.assertIn("a sabre", others)

    def test_only_the_measured_qualities_are_judged(self):
        self.assertEqual(set(words.QUALITY_JUDGE_TEXTS), {"curved", "round"})
        for prompt in ("a tall clay vase", "a square wooden box", "a broken wooden chair", "a straight iron sword",
                       "a squat iron lantern", "a rusty iron lantern", "a giant rusted iron key"):
            self.assertEqual(words.subject_of(prompt).qualities, [], prompt)

if __name__ == "__main__":
    unittest.main()
