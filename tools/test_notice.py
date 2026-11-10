"""NOTICE names every model registry/models.toml holds, and every extra hub repository a model's files come from.

    python3 -I -m unittest discover -s tools -p 'test_*.py'
"""

import pathlib
import tomllib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent


class Notice(unittest.TestCase):
    def test_every_registered_model_is_in_the_notice(self):
        notice = (ROOT / "NOTICE").read_text().lower()
        registry = tomllib.loads((ROOT / "registry" / "models.toml").read_text())
        for model in registry["model"]:
            first_word = model["name"].split()[0]
            self.assertIn(first_word.lower(), notice, model["id"])
            for repository in model.get("folders", {}):
                self.assertIn(repository.split("/")[-1].lower(), notice, f"{model['id']}: {repository}")


if __name__ == "__main__":
    unittest.main()
