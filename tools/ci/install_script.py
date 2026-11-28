#!/usr/bin/env python3
"""install.ps1 for one Windows release: the template tools/ci/install.ps1 with the version, the archive's address and
sha256, and the address of the documentation page's list of sums written in.

    python3 -I tools/ci/install_script.py VERSION ARCHIVE ARCHIVE_URL SUMS_URL OUT [--test]

Both addresses must be https; --test also allows http on this machine (127.0.0.1), for CI's own run of the script.
The standard library only.
"""

import argparse
import hashlib
import os
import re
import sys

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "install.ps1")
LOCAL_TEST_ADDRESS = "http://127.0.0.1:"


def filled(template, version, archive_name, sha256, archive_url, sums_url, test=False):
    """The template with the four values in; ValueError for a value the script could not use safely."""
    for url in (archive_url, sums_url):
        if not (url.startswith("https://") or (test and url.startswith(LOCAL_TEST_ADDRESS))):
            raise ValueError(f"{url}: not an https address")
        # The values sit in single-quoted PowerShell strings.
        if "'" in url or any(c.isspace() for c in url):
            raise ValueError(f"{url}: a quote or a space")
    if archive_url.rsplit("/", 1)[-1] != archive_name:
        raise ValueError(f"{archive_url} does not end with the archive's name {archive_name}")
    if not re.fullmatch(r"[0-9A-Za-z.+-]+", version) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError(f"version {version!r} or sum {sha256!r} is not one")
    values = {"@VERSION@": version, "@ARCHIVE_URL@": archive_url, "@ARCHIVE_SHA256@": sha256,
              "@SUMS_URL@": sums_url}
    for placeholder, value in values.items():
        template = template.replace(placeholder, value)
    left = re.findall(r"@[A-Z0-9_]+@", template)
    if left:
        raise ValueError(f"not filled in: {', '.join(left)}")
    return template


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("version", "archive", "archive_url", "sums_url", "out"):
        parser.add_argument(name)
    parser.add_argument("--test", action="store_true")
    options = parser.parse_args(argv)
    with open(options.archive, "rb") as file:
        sha256 = hashlib.sha256(file.read()).hexdigest()
    with open(TEMPLATE, encoding="utf-8") as file:
        template = file.read()
    try:
        text = filled(template, options.version, os.path.basename(options.archive), sha256, options.archive_url,
                      options.sums_url, options.test)
    except ValueError as refused:
        print(f"install_script: {refused}", file=sys.stderr)
        return 1
    with open(options.out, "w", encoding="utf-8", newline="\r\n") as file:
        file.write(text)
    print(f"{options.out}: {options.version}, {sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
