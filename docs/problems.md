# How to report a problem

Open an issue on GitHub with the **Bug report** form, and attach the bug report file for the job:

```
./model-maker bug-report made/chair-ec4d6029
./model-maker bug-report made/chair-ec4d6029 --with-model    # the .glb too
```

It writes one file (`bug-report-<name>-<id>.zip`; give `--out FILE.tar.gz` for a .tar.gz) and prints its size and
what it left out. It holds the job's record and sidecar, its report and pictures, the picture it was made from and
its cut-out, the job's lines from the ledger, and the versions it was made with: the code, the AI models, Python,
torch, the graphics card and its driver. The prompt is sent whole, as the remake needs it: leave out of a prompt
anything you would not post. It leaves out the large in-between files, your environment and any token, and it
replaces your home folder, your user name, your machine's name and other folder paths in every text it copies.
`README.txt` inside says what it holds. Drag the file into the form's box (GitHub takes files up to 25 MB). For a
question, use the **Question** form; for a security problem, see [`SECURITY.md`](../SECURITY.md).

We claim no rights in what you make; the models' terms apply ([`NOTICE`](../NOTICE)). As the code that made an object
is on your machine too, we cannot take the object or its report as proof of anything: we can check a claim only by
remaking it. So a maintainer makes the same prompt with the same seeds, settings and version on a card of their own
(`./model-maker reproduce FILE --out DIR`) and compares the picture, every check's value and the verdict with yours.

## Contact

developer@cicadapark.com. Security problems: see [`SECURITY.md`](../SECURITY.md).
