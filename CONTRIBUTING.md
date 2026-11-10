# Contributing

Thank you for helping. A few rules keep the project safe to use and to share:

1. **Licences.** Code in this repository, and the libraries and models it uses, only under MIT, Apache-2.0, BSD,
   ISC, Zlib or a licence as free. No GPL, AGPL, LGPL or non-commercial code, models or assets. What the installed
   packages' own wheels bundle is listed in NOTICE and checked library by library: NVIDIA's CUDA libraries (free to
   use and ship, not open source), GCC's runtime libraries (GPL with the GCC Runtime Library Exception, which allows
   shipping them with any program), and one recorded exception, libquadmath (LGPL), which numpy's and scipy's wheels
   load at run time. `cargo deny check` and `python/model_maker/check_licences.py` check this.
2. **Your own work.** Send only what you made yourself or may share under Apache-2.0.
3. **No secrets.** No tokens, keys, passwords, addresses or machine names in code, docs or files.
   A check looks for them before anything is merged.
4. **Tests.** `cargo test`, `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` pass
   before a pull request.
5. **Plain words.** Short sentences and descriptive names in code and docs.
6. **AI help is welcome; say which AI.** If an AI assistant or agent helped write a commit, end the commit message
   with a `Co-Authored-By:` line that names the AI and its model and version, for example
   `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. One line for each AI that helped. Naming the model is
   required: we want to know which AI helped with what. You stay responsible for what you send, AI-written or not.

By sending a pull request you agree that your work is licensed under Apache-2.0.
