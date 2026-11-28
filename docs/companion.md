# The companion: a small window for model-maker

`model-maker-companion` is a small window that comes in the same archive as `model-maker`, beside it. It is for
people who would rather not type commands. It does five things and nothing more:

1. **This PC**: checks this PC the way `model-maker check` does, and shows each row with its verdict. **Set up**
   runs `model-maker setup` with one bar for each step. The models' bar shows bytes downloaded out of the total,
   the speed and the time left. Every model's licence is shown before you accept it (the same as
   `model-maker setup` asks). FLUX.1 [schnell] also needs a Hugging Face access token
   ([A gated model](setup.md#a-gated-model)). The token is passed to setup in a new file in the window's own
   folder (`.tools/companion` in model-maker's folder, yours only), and that file is deleted when setup ends, when it
   cannot start, and when the window closes. On Windows, the window asks first before Microsoft's C++ runtime is
   fetched. "This PC is ready" is the launcher's own word: its check passes, and setup's own record
   (`.tools/setup-done.json`, written when a setup ends well with the models) is there.
2. **Studio**: connecting this PC to the studio comes in the next version. Until then, the tab says so.
3. **The status line** (always at the top): what is running, and a fold with the card's load, temperature and
   memory (from `nvidia-smi`; "not known" when it does not say).
4. **Make**: one object from a prompt, as `model-maker make "<prompt>" --out <folder>` with the size, stands and
   symmetric choices you make, nothing else. It shows one bar for the whole job and one for each stage (pictures,
   shape, paint, finish). Each stage shows its time so far and, when this PC has made objects before, about how
   long it has left. **Cancel** stops the make and everything it started; its folder is kept as it is. At the end
   you get one line: "Made: passes N of N checks", or "Made, with N failed checks" (from the job's own record),
   **Open folder**, and **Make another**. The window shows no picture or 3-D view and keeps no list of past makes:
   to look at an object or edit it, open the studio.
5. **Open studio**: opens the studio page (https://cicadapark.com/model-maker-studio/) in your browser. The
   window opens only https addresses: that page and the models' licence pages.

What it does not do: it adds nothing the commands cannot do, and it never changes a check's verdict. It never sets
up or makes anything when a check has failed. It writes nothing into a job's folder or into the job ledger.

## How it works

- Each step is the `model-maker` program beside it: `model-maker setup` and `model-maker make`. Setup's steps come
  from a progress file the launcher writes when `MODEL_MAKER_PROGRESS` is set. A make's stages come from the job
  ledger's lines (`metrics/jobs.jsonl`). The program's printed output is shown under "Details" exactly as it is
  printed; the window never reads its meaning from that text.
- The window is drawn by the processor, not the graphics card, so the card's memory stays free for the make.
- It uses the system's own font. No font is shipped with it.
- One companion runs at a time for a model-maker folder, and it runs one make at a time. Closing the window while something runs
  asks first, then ends what it started. On Windows the launcher's job object ends what a make started (Cancel says
  only "Cancelled" there: the window counts nothing; a test on every pull request starts a make through the real
  `model-maker.exe` and checks that nothing of it is left). On Linux and
  macOS, Cancel and closing end the make's whole process group.
- Like `model-maker` itself, the program is **unsigned** for now; check the archive's sha256 against the published
  one.

## On a PC set up before this version

"Ready" needs setup's own record of a good end (`.tools/setup-done.json`), which earlier versions did not write.
A PC set up with an earlier version therefore shows "No record of a finished setup" until setup runs once more;
it carries on where it stopped, so it is quick. `model-maker setup --skip-models` run after a full setup rewrites
the record as "without the models", and the window then asks for setup again (a full setup puts it right).

## Not in this version

The studio link (pairing, "Use my GPU for the studio", Pause), starting with the system, and a tray icon.
