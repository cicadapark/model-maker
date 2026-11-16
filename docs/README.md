# docs

model-maker's documentation: how to set it up and use it, what it makes and how well, and how the pipeline works,
with its numbers and what we learned making it.

Using it:

* [`setup.md`](setup.md): what a PC needs; setting it up with `setup.sh`, what it installs and checks, a 12 GB card,
  the times on a clean PC, your own signing certificate, warnings you can ignore.
* [`what-you-get.md`](what-you-get.md): what it makes (with examples), what you get for each object, what is in this
  repository.
* [`results.md`](results.md): the pass rate and every result by eye, the known limits, and what is not here yet.
* [`for-agents.md`](for-agents.md): for AI assistants and agents: when to use it, the commands, the exit codes, and how to
  read what it makes.
* [`names-the-picture-model-misses.md`](names-the-picture-model-misses.md): the living list of object names the
  picture model paints as something else, each measured, with a description that paints it right (to write into your
  prompt until a later version says it for you); how to add one.
* [`problems.md`](problems.md): how to report a problem (the bug report file), and how a claim is checked by
  remaking it; contact.
* [`licences.md`](licences.md): the licence, the models' terms, and content credentials.

How it works:

* [`how-it-works.md`](how-it-works.md): the whole process in one diagram, and each step in words.
* [`objects.md`](objects.md): 3-D objects from a prompt, every check and its limit, the marks, thin parts.
* [`metrics.md`](metrics.md): the job ledger, what each line holds, and the summary made from it.
* [`trials/`](trials/): the notes of the release's first two trials, and of the diagnosis of 0.2.0's sealed run: what
  failed, why, and what was decided (the other trials are described in `objects.md`; their notes are not published).
