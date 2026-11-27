# The regression set's cases

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

The cases as they were added, trial by trial; how the set is used, and why its data is not published, is in
[`docs/objects.md`, "The regression set"](../objects.md#the-regression-set).

Every model a check caught, and every case where a check once misread a good model, is kept with
the verdicts it must keep (`tools/trial/regression.toml`): the chair whose back legs came out short,
the table with a web between its legs, the waving and the plate-like longbows, the barrel's seams and
folds, the lantern's hollow and its truly unpainted bars, the jug's seldom-seen neck, the chair and
table that match their pictures with thin legs, the chest and book whose shadows lie outside their
cut-outs, the sword that passes, the third trial's table with a leg 24.6 mm short (it must still fail
"base level" as made and pass once levelled), its chest painted with the lid up and its book painted open
(both must fail "shows what the words ask"), the 12 GB card's book painted open with both halves in the cover's leather (it
must fail it too), and the first trial's chest and book, shut, which must pass it; every book and chest picture the
trials painted, labelled open or shut by eye (`tools/trial/state-cases.toml`: each must pass or fail the state check as
its label says, judged as one case, "state-pictures"); every distinct picture the trials painted, labelled by its prompt (`tools/trial/prompt-cases.toml`:
each clear one must pass "shows the thing asked for" against its own prompt and fail it against every other, judged as
one case, "prompt-pictures"; see [The picture against the prompt](picture-check.md)); the pictures painted with a quality and without it, labelled
by eye (`tools/trial/quality-cases.toml`: judged against the prompt that names the quality, each showing it must pass
"shows the thing asked for" and each showing its opposite must fail it, judged as one case, "quality-pictures"; see
[Several things, and qualities the words name](several-and-qualities.md)); the picture tries labelled by eye for painted lettering
(`tools/trial/lettering-cases.toml`: each showing none, and each whose prompt asks for lettering, must pass "no painted
lettering", each showing letters its prompt does not ask for must fail it, its known misses shown; judged as one case,
"lettering-pictures", and it needs tesseract 5 or newer; see [Painted lettering](lettering.md)); the fix-check's and the fifth trial's jugs, a web in the handle's loop (as made they must fail the outline, carved
they must match it, closed and in one piece), and the chairs, stool, tables and lantern whose holes must not be carved;
the fix-check's book made standing on its edge (laid, it must stand), its chest whose picture is a photograph (it
must match through a lens) and its candle as the six views painted it (it must keep failing the paint's coverage); the touch-up's first candle, whose fill painted holes and grey patches (it must fail "fill matches the paint round it" while its coverage passes), and its chair, filled in its own wood (it must pass both); the fifth trial's candle, whose foot left out shows a smear of wax and brass (it must fail the fill check), and a clay jug's and a shield's handle's places left out, the model showing one colour there (they must pass it). The state cases need SigLIP 2 in `MODEL_MAKER_MODELS`; it runs on the CPU too, a few seconds a picture. `tools/trial/regression.py run DATA` judges them all on the CPU in a
few minutes; it runs before any change to the pipeline or its checks, and every case must hold.
