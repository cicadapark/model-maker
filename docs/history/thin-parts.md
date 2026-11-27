# Thin parts

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

The shape model sees the object through a picture 224 pixels square ([`docs/objects.md`, "Thin parts"](../objects.md#thin-parts) gives each
thin part's width in it). On the first trial the spear's shaft, the axe's and hammer's hafts and the broom's
handle all came out straight and whole. The longbow did not, but for its view, not its thinness:
painted edge-on, its bend was out of sight and the shape model made it wave. Asked to face the
viewer, the picture model still painted it in profile with its string, and the shape model filled
the space between bow and string into a flat plate.

Bows are not made yet, so a bow prompt is answered before anything is made ([`docs/objects.md`, "How a prompt is
read"](../objects.md#how-a-prompt-is-read)). What is ready for them: a bowstring is below what the shape model can see. When a bow comes out as one curve to one side
of the line between its tips, standing at least 5 % of its length from it, the finish draws a
string (a cord 3 mm thick) between the tips with our own code: its own mesh named `string` in the
`.glb`, in one plain colour, marked in the sidecar as made by code, and not part of the checks,
which judge the shape model's work. A bow that is straight, or waves, gets none, and the sidecar
says why.
