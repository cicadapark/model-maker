# registry

`models.toml`: the AI models the pipeline uses, each with its licence, its hub revision and the
sha256 of every file that is loaded. Before a model is loaded its files are checked against these
sums, and a file in its folder that runs code (a model's own Python, a pickle) and is not listed is
refused. MV-Adapter's code, checked out from its repository (`source`), is checked to be at the
commit listed, with nothing changed or added, each time before it is imported. Only models whose
licence allows commercial use without a fee are listed.
