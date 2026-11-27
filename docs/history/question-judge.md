# Is it the thing asked for?

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## Is it the thing asked for? (14 November 2026)

The checks measure how an object is made, not what it is: on 50 prompts sealed before the work, 8 objects were
another thing made well (a digging fork for "a garden hoe", a pick on a pole for "a scythe"), and most passed every
check. The question judge (`question_judge.py`, optional) shows the finished object's eight sides, laid out four by
two, to a vision-language model, Qwen 3.5 9B with its vision part, run on the processor by llama.cpp's server, and
asks "These are eight views of one object, seen from all round. Is it <prompt>? Answer yes or no." The check "it is
the thing asked for" is the probability of "yes" against "no" in the first word of its answer (VQAScore's way: Lin et
al. 2024); it passes from 0.5.

**What it is asked.** Asked the whole prompt ("Is it an old iron sword?"), it judged every word: it kept 16 of the
regression set's 17 good objects and refused a good longsword (0.44; "an iron sword" 0.59, "a sword" 0.78), its paint
not looking old. So it is asked about the thing the words name, with its article ("a sword", "a hoe", "a chess
board"; the whole prompt when the words name no thing model-maker knows), and it counts. Since 0.3.0 a name of two
words is asked whole ("Is it a bow saw?", not "a saw", which refused a right bow saw); a word that names a material as
well as a thing (bone, rock, horn, rope, chain, log, feather, as stone was before) is read as the material when another
noun follows, so "a bone knife" is asked as "a knife"; and of a pair, the one made is asked about ("a pair of
boots": "Is it a boot?"). That question, its limit 0.5
and the bar it had to meet were written down before it was measured (on the next sealed 50 of 0.2.1's successor,
judged blind, both cards): among the objects that pass every other check it caught 7 of 16 bad and refused none of
15 good on the 12 GB card, and caught 5 of 14 bad and refused 2 of 15 good on the 16 GB card (a narrow coffee table
asked "a table", a gargoyle); it refused none of the regression set's 17 good objects. A refused object's report says
what was asked and how to say more in the words. A run that requires it (`--require-question-judge`) fails a job it
could not ask: missing, not starting, or with model files that are not the registry's.

Measured on the sealed 50 of 0.2.1 on the 12 GB card, the limit fixed before the run: the objects that pass every check
while bad went from 11 to 5 (the hoe 0.09, the scythe 0.33, a bronze sword painted iron 0.37, the telescope 0.40, the
bellows 0.42 to 0.48, the hand mirror 0.44); 3 good ones were refused (a pitchfork, a halberd, a barrel). Asking about
its eight sides is what works: about the first picture alone, the same model is only a little better than the picture
check. Asking whether the object is "whole and well made" sees nothing (a split head, an open base, blotches pass).

The pictures are asked the same question, as a note: the chosen try first, the others only when it is under 0.5.
Nothing is chosen or refused by it: over 400 tries of 50 names, the try either judge ranks first was right as often as
the first try (38 of 50), and refusing a name when none of its 4 tries reaches 0.5 lost two right names.

Builds of llama.cpp answer differently (its own release archive of the same tag moved scores by up to 0.14 against a
local build; 8 of 96 verdicts flip at 0.5, both ways, all within 0.06 of it), and the archive picks a library for the
processor it runs on. model-maker uses only the archive setup puts in `.tools/llama-b11600/`, never one on the PATH;
an object near 0.5 may still be judged differently on another processor.

Known limits: long thin things (a pitchfork, a halberd) are tiny in the views and can be refused while right; names
the model does not know pass whatever is made (a sickle painted as a scythe scored 0.90); shape and paint faults are
not its question.

**Its memory (0.3.1).** llama.cpp's server keeps the prompts it has answered (up to 8 GiB by default), to answer
faster a prompt that begins the same way. The judge never asks the same picture twice, so nothing kept was used, and
the server grew over a batch: on 160 pictures from 9.4 to 17.5 GB (its peak resident memory in GiB, on its processor
alone, on the 12 GB card's PC of docs/setup.md: 24 threads, 45 GB), and up to 15.4 GB in a batch of 20 objects. Since
0.3.1 it keeps none (`--cache-ram 0` when it starts, and `cache_prompt` false in every question). On the same 160
pictures its peak was 9.6 GB and did not grow (its own memory 4.1 GB, the rest the model's files mapped in); asking
again 29 finished objects of two batches and 44 of their picture tries, 9.7 GB. Every answer was the same as before,
to the last digit: the 160 against the run before the change, the 73 against the answers their batches recorded. Not
repacking the weights for the processor (`--no-repack`) would save about 2.5 GB more, but it moves the answers (by up
to 0.09 on the 160), so it stays.
