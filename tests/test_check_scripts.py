"""The repository's check scripts pass: input validation, diagnostic plots,
declared function parameters and saved notebook outputs."""

import os
import subprocess
import sys
import unittest

from _support import ROOT, SRC

SCRIPTS = [
    "check_input_validation.py",
    "check_diagnostic_plots.py",
    "check_function_schema.py",
    "check_notebook_outputs.py",
]


class CheckScriptTests(unittest.TestCase):
    def test_scripts_pass(self):
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC), env.get("PYTHONPATH")]))
        for script in SCRIPTS:
            with self.subTest(script=script):
                done = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, env=env,
                                      capture_output=True, text=True, encoding="utf-8", errors="replace")
                output = (done.stdout + done.stderr).strip().splitlines()
                self.assertEqual(done.returncode, 0, f"{script} failed:\n" + "\n".join(output[-40:]))


if __name__ == "__main__":
    unittest.main()
