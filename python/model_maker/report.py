"""An object's report: one HTML page with its pictures beside it. The picture it was made from and the tries beside
it, the outline laid over the picture's, the turntable painted and bare, where the paint came from, a held
object's grip and tip drawn on it, and every check with its measure, its limit and why it ran.
"""

import html
import os

import numpy as np
from PIL import Image, ImageDraw

from . import render

# How the report names a subject's kind (the record's own words, "object" and "weapon", stay as they are).
KIND_WORDS = {"object": "3-D object", "weapon": "3-D object held in the hand"}
# Under the models used: who made it, and the licence terms that come with it.
MADE_WITH_NOTE = ("Made by AI models from a text prompt. We claim no rights in what you make; the models' terms apply "
                  "(NOTICE). The use restrictions of the CreativeML Open RAIL++-M licence (Stable Diffusion XL, its "
                  "Attachment A) apply to this model.")

STYLE = """
:root { --bg: #f7f7f4; --fg: #1d1f1c; --muted: #5d625a; --line: #d8dbd2; --pass: #2f6b4f; --fail: #a8322a;
        --card: #ffffff; }
@media (prefers-color-scheme: dark) { :root { --bg: #161816; --fg: #e8eae5; --muted: #a2a99e; --line: #343a35;
        --pass: #7cc39c; --fail: #f08a80; --card: #1e211e; } }
body { background: var(--bg); color: var(--fg); font: 15px/1.55 system-ui, sans-serif; margin: 0;
       padding: 24px max(16px, 3vw) 64px; }
main { max-width: 1100px; margin: 0 auto; display: flex; flex-direction: column; gap: 22px; }
h1 { font-size: 1.6rem; margin: 0; } h2 { font-size: 1.15rem; margin: 0 0 8px; }
.muted { color: var(--muted); } .pass { color: var(--pass); font-weight: 650; } .fail { color: var(--fail);
font-weight: 650; }
img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 4px; background: #eeeeea; }
.row { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; }
.row figure { margin: 0; flex: 1 1 220px; max-width: 360px; } figcaption { font-size: .85rem; color: var(--muted); }
.tries figure { flex: 0 1 150px; }
table { border-collapse: collapse; width: 100%; } .wide { overflow-x: auto; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
section { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; }
"""


def drawn_marks(corners, triangles, colour_at, measures, side=520):
    """The held object seen from the front with its grip (a line from start to end, a ring where the hand goes) and
    its tip drawn on it."""
    picture = render.shaded(corners, triangles, 0, side, colour_at, tilt_deg=0.0)
    weapon = measures.get("weapon") or {}
    framing = render.Framing(corners, 0, side, 0.0)
    draw = ImageDraw.Draw(picture)

    def at(point):
        xy, _ = framing.pixels(np.array([point], dtype=np.float64))
        return tuple(xy[0])

    string = measures.get("string") or {}
    if string.get("from"):
        draw.line([at(string["from"]), at(string["to"])], fill=(150, 120, 60), width=2)
    grip = weapon.get("grip")
    if grip:
        draw.line([at(grip["from"]), at(grip["to"])], fill=(30, 120, 220), width=5)
        x, y = at(grip["hold"])
        draw.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(30, 120, 220), width=3)
    for name in ("tip", "front"):
        if weapon.get(name):
            x, y = at(weapon[name])
            draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(220, 60, 40))
    return picture


def _figure(path, caption):
    return (f'<figure><img src="{html.escape(path)}" alt="{html.escape(caption)}" loading="lazy">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def _check_rows(checks):
    rows = []
    for c in checks:
        if not c.get("counted", True):
            verdict = '<span class="muted">shown</span>'
        else:
            verdict = '<span class="pass">pass</span>' if c["passed"] else '<span class="fail">fail</span>'
        value = f"{c['value']:.2f}" if isinstance(c["value"], float) else str(c["value"])
        rows.append(f"<tr><td>{html.escape(c['name'])}</td><td>{verdict}</td><td>{value} {html.escape(c['unit'])}"
                    f"</td><td>{c['limit']} {html.escape(c['unit'])}</td><td>{html.escape(c['words'])}</td></tr>")
    return "\n".join(rows)


def written(folder, job, checks, measures, pictures, models_used):
    """Writes `folder/report.html` and its pictures (`pictures`: {file name: (PIL image, caption)}); returns its
    path. `checks` are dicts grouped by stage: {"picture": [...], "shape": [...], "paint": [...]}."""
    os.makedirs(folder, exist_ok=True)
    for name, (image, _) in pictures.items():
        image.save(os.path.join(folder, name))
    every = [c for group in checks.values() for c in group if c.get("counted", True)]
    failed = [c for c in every if not c["passed"]]
    verdict = (f'<span class="pass">passes all {len(every)} checks</span>' if not failed else
               f'<span class="fail">fails {len(failed)} of {len(every)} checks</span>: '
               + html.escape("; ".join(c["name"] for c in failed)))
    size = measures.get("size_m", [0, 0, 0])
    size_words = f"{size[0] * 100:.0f} × {size[1] * 100:.0f} × {size[2] * 100:.0f} cm (wide × high × deep)"
    weapon = measures.get("weapon") or {}
    if weapon.get("length_m"):
        size_words += f"; its length {weapon['length_m'] * 100:.0f} cm"
    subject = job["subject"]
    why = "".join(f"<li>{html.escape(k)}: {html.escape(v)}</li>" for k, v in subject.get("why", {}).items())
    figures = {kind: "".join(_figure(n, c) for n, (_, c) in pictures.items() if n.startswith(kind))
               for kind in ("picture", "try", "outline", "turntable", "bare", "coverage", "marks", "views")}
    sections = [
        f"<section><h1>{html.escape(job['prompt'])}</h1><p>{verdict}</p>"
        f"<p class='muted'>{html.escape(KIND_WORDS.get(subject['kind'], subject['kind']))}"
        f"{' (' + html.escape(subject['weapon_kind']) + ')' if subject.get('weapon_kind') else ''}; "
        f"{html.escape(size_words)}; size set to {job['size_m']:.2f} m (its largest side; "
        "<code>--size</code> sets it); "
        f"{measures.get('triangles', 0)} triangles.</p><ul class='muted'>{why}</ul></section>",
        f"<section><h2>Picture</h2><div class='row'>{figures['picture']}</div>"
        f"<h2>The tries</h2><div class='row tries'>{figures['try']}</div></section>",
        f"<section><h2>Shape</h2><p class='muted'>{html.escape(repair_words(measures.get('repair')))}</p>"
        f"<div class='row'>{figures['outline']}{figures['marks']}</div>"
        f"<div>{figures['turntable']}</div><div>{figures['bare']}</div></section>",
        f"<section><h2>Paint</h2><div class='row'>{figures['coverage']}{figures['views']}</div></section>",
    ]
    for group, group_checks in checks.items():
        if group_checks:
            sections.append(f"<section><h2>Checks: {html.escape(group)}</h2><div class='wide'><table><tr><th>check</th>"
                            "<th>result</th><th>measured</th><th>limit</th><th>what it means</th></tr>"
                            f"{_check_rows(group_checks)}</table></div></section>")
    model_rows = "".join(f"<tr><td>{html.escape(m['name'])}</td><td>{html.escape(m['role'])}</td>"
                         f"<td>{html.escape(m['version'])}</td><td><a href='{html.escape(m['licence_url'])}'>"
                         f"{html.escape(m['licence'])}</a></td></tr>" for m in models_used)
    sections.append("<section><h2>Made with</h2><div class='wide'><table><tr><th>model</th><th>role</th>"
                    f"<th>version</th><th>licence</th></tr>{model_rows}</table></div>"
                    f"<p>{html.escape(MADE_WITH_NOTE)}</p>"
                    f"<p class='muted'>Times: {html.escape(timing_words(job.get('seconds', {})))}</p></section>")
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width, initial-scale=1'><title>{html.escape(job['slug'])} report</title>"
            f"<style>{STYLE}</style></head><body><main>{''.join(sections)}</main></body></html>")
    path = os.path.join(folder, "report.html")
    with open(path, "w") as file:
        file.write(page)
    return path


REPAIR_WORDS = [("corners_joined", "corners at the same place joined"),
                ("degenerate_dropped", "triangles with no area dropped"),
                ("folds_dropped", "folds of no thickness dropped (in triangles)"),
                ("doubles_dropped", "doubled triangles dropped"), ("corners_split", "corners split where sheets touched"),
                ("holes_filled", "small holes filled"), ("holes_left", "holes too large to fill"),
                ("hollows_removed", "hollows inside it removed")]


def repair_words(repair):
    """What the clean-up before the checks did, in words."""
    if not repair:
        return "Not cleaned up before the checks."
    done = [f"{repair[key]} {words}" for key, words in REPAIR_WORDS if repair.get(key)]
    return "Cleaned up before the checks: " + ("; ".join(done) if done else "nothing needed") + "."


def timing_words(seconds):
    if not seconds:
        return "not timed"
    parts = [f"{stage} {value:.0f} s" for stage, value in seconds.items()]
    return ", ".join(parts) + f"; in all {sum(seconds.values()):.0f} s"


def thumbnail(image, side=300):
    small = image.copy()
    small.thumbnail((side, side))
    return small


def mask_picture(mask):
    return Image.fromarray(np.where(mask, 40, 235).astype(np.uint8))
