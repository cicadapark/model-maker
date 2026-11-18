# How it works

One line of text goes in; a checked 3-D object comes out. This page shows the whole process in one picture, then
says each step in words. The details of every check, with its limit and what it was measured on, are in
[`objects.md`](objects.md).

![The model-maker pipeline as a diagram: the text is read; FLUX.1 [schnell] paints 4 picture tries, BiRefNet cuts them out, and our checks, SigLIP 2 and tesseract judge them, with 4 more tries if none passes and "not made" after 8; TripoSG builds the shape from the best picture, our code repairs and checks it, with a second shape if a check fails; MV-Adapter on SDXL paints six views, which must agree or are painted again; the views are baked into one texture and SDXL inpainting touches up places no view reached; the shape is levelled and checked again, the paint is checked, the question judge (Qwen 3.5 9B) asks of the eight sides whether it is the thing the words name, and the .glb, its report and its sidecar are written, every picture signed. Below, the loop between releases: look by eye, keep the case, fix and review, measure on sealed prompts.](images/pipeline.webp)

Colours: orange boxes are AI models that make something, green ones AI models that judge, blue ones our own code.
A dashed line means "try again"; a red box means nothing is made. The numbers are the settings of version 0.2.2, which 0.3.0 keeps.
The diagram's source is [`images/pipeline.html`](images/pipeline.html).

## The steps in words

1. **Read the words** (our code). What kind of thing it is, whether it is held in the hand, its size, the qualities
   the words name ("curved", "round"), open or shut, and lettering. An animal, a character, a bow or more than one
   thing is answered at once and nothing is made; two of one thing ("a pair of boots") is made once and placed twice
   in step 13, the second its mirror image or its copy.
2. **Paint 4 picture tries** (FLUX.1 [schnell]): one object on a plain backdrop, from 4 seeds.
3. **Cut each try out** (BiRefNet).
4. **Judge every try**: our code (one whole object, no shadow or floor, not cut off), SigLIP 2 (it shows the thing
   asked for, with its qualities, open or shut) and tesseract (no painted lettering unless the words ask for it). If
   no try passes, 4 more are painted; if none of the 8 passes, nothing is made and the report says why.
5. **Build the shape** (TripoSG) from the best picture.
6. **Repair and prepare the shape** (our code): holes the picture shows opened, 60,000 triangles, real size in
   metres, set down on its base; a hafted weapon turned upright, a book laid flat.
7. **Check the bare shape** (our code): its outline against the picture, holes, one piece, its feet, level and
   balance, symmetry where the words imply it; for a thing held in the hand, a straight blade, no kink, the grip and
   the tip. If a check fails, a second shape is built and the better one is kept.
8. **Paint six sides** (MV-Adapter on SDXL), fitted to the shape.
9. **Check that the six views agree** (our code): each side view against its neighbours where both saw the same place.
   A view unlike both is painted again from the next seed, at most twice; the try that agrees best is kept.
10. **Bake and touch up**: the views baked into one 2048 × 2048 texture (our code); places no view reached painted from
    up to 4 more views (SDXL inpainting), from above, level or below. A fill with colours found nowhere round it is
    painted again from another seed, and left out if it is still foreign.
11. **Finish and check** (our code): a leg a little short levelled, the shape checked again (a thing that stands also
    for being closed underneath), the paint's coverage checked; how the fill matches the paint round it is shown as a
    note, not counted.
12. **Ask whether it is the thing asked for** (Qwen 3.5 9B, optional: setup installs it unless told not to). The
    object's eight sides are shown together and the model is asked "Is it a sword?": the thing the words name, not
    every word of them. Below 50 % yes, the object is marked failing, and its report says what was asked and how to say
    more in the words.
13. **Write the result** (our code): the `.glb` with its grip and tip marks, the report, and the sidecar naming every
    model, licence and seed; every picture signed with content credentials. An object that fails a check is still
    written, marked failing, with the checks it failed.

**Between releases**, every object is looked at by eye from all round and from below; each failure caught becomes a
case in the regression set; a fix must keep every case and is read by an AI reviewer; and each release is measured
once on prompts sealed before the work began.
