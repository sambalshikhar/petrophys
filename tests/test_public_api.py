"""Every public entry point accepts its documented inputs and refuses bad ones.

The cases live in ``api_cases.py``; each group below reports its failing cases
by name. Run one group from a shell with
``python -m unittest test_public_api.PublicApiTests.test_plot`` (inside tests/).
"""

import unittest

from _support import HAS_SKLEARN
import api_cases


class PublicApiTests(unittest.TestCase):
    results = None

    @classmethod
    def setUpClass(cls):
        cls.results = api_cases.run_all()

    def check_group(self, group):
        cases = [row for row in self.results if row[0] == group]
        self.assertTrue(cases, f"no cases ran in group {group!r}")
        problems = [f"{status} {name}: {detail}" for _, name, status, detail in cases
                    if status in {"FAIL", "UNEXPECTED-OK"}]
        if problems:
            self.fail(f"{len(problems)} of {len(cases)} {group} cases:\n  " + "\n  ".join(problems))

    def test_samples(self):
        self.check_group("samples")

    def test_read(self):
        self.check_group("read")

    def test_read_formations(self):
        self.check_group("read_formations")

    def test_well(self):
        self.check_group("Well")

    def test_config(self):
        self.check_group("config")

    def test_interpret(self):
        self.check_group("interpret")

    def test_result(self):
        self.check_group("result")

    def test_plot(self):
        self.check_group("plot")

    @unittest.skipUnless(HAS_SKLEARN, "pp.impute needs scikit-learn (pip install petrophys[ml])")
    def test_impute(self):
        self.check_group("impute")

    def test_export_context(self):
        self.check_group("context")

    def test_every_bundled_well(self):
        self.check_group("sweep")


if __name__ == "__main__":
    unittest.main()
