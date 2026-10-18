# Release verification evidence

The release audit is in progress. Dataset/model redistribution checks, Taranaki
and Teapot verification, the release version, and publishing setup were deferred
by the maintainer on 16 October 2026. Those items cannot be counted as passed
release requirements. No TestPyPI or PyPI upload is authorized until the required
verification and permissions are resolved.

Generate the review workbook from the repository root:

```bash
python scripts/release_inventory.py
```

Open `outputs/release_audit/Petrophys_Release_Audit.xlsx`. Each module, callable,
extracted numerical expression, declared public export, workflow setting,
dataset, script, template, notebook and saved figure has a review row.
The workbook preserves the `Reviewer`, `Review decision` and `Review notes`
columns when regenerated. Imported test results are a separate evidence sheet;
test coverage or an execution pass does not approve a scientific equation.
Extracted expressions are implementation text, not independent references.
Reports are ordered by execution time; the evidence sheet retains earlier
failures and identifies the latest recorded run. Source hashes invalidate a
previous review decision when its callable or expression changes.

Run independent spot checks and record expected/actual values:

```bash
python scripts/release_science_checks.py
```

The initial checks cover clean-formation Archie, mass-balance density porosity,
Wyllie travel-time mixing, effective porosity volume balance, basic unit
conversions, SI checks on capillary relations, analytic pressure intersections,
NMR logarithmic means and cutoff volume partitions, a constructed isotropic
elastic medium, and scalar/array and missing-input contracts. Passing these checks approves
only those stated cases. The full scientific review remains required.

Run all notebooks, including specialist/ML/MCP lessons, in an environment with
the complete declared extras:

```bash
python scripts/execute_tutorials.py --all --output-dir outputs/release_audit/notebooks-full --report outputs/release_audit/evidence/notebooks-full.json
```

The report is updated after each notebook. A skipped dependency or failed cell
is not a passing specialist check.

Evidence lives under `outputs/release_audit/evidence/`. Source and artifact
hashes matter: a source change invalidates affected verification results, and
artifacts must be rebuilt after fixes. Install wheel and sdist in separate clean
environments, run from outside the source tree, and exercise them with:

```bash
python /path/to/scripts/verify_installed_release.py --artifact /path/to/dist/petrophys-VERSION.whl --report /path/to/evidence/checks-installed.json
```

Use the sdist path when testing a clean sdist installation. The script refuses
to validate an import from `src/` or an editable checkout. It checks installed
resources, version consistency, input preservation, clastic/carbonate
interpretation, general context and CLI export. Full tests, optional feature
checks, documented examples, figure review, and the platform matrix are
additional required evidence.

Check all declared public exports in the installed optional-dependency
environment with `scripts/verify_public_api.py --report REPORT.json`.
Use `scripts/verify_context_package.py PACKAGE --reference REFERENCE --report
REPORT.json` for strict JSON parsing and exact reference comparisons. It reports
every changed field and missing filename without tolerances or ignored keys.
`scripts/summarize_release_checks.py` consolidates completed local checks and
verifies that the ten Windows reports identify the final candidate hashes.
It selects the most recently completed full-suite, installed-optional and
documentation reports; earlier results remain in the evidence directory.

Extract figures from the latest execution copies with:

```bash
python scripts/release_figures.py --notebooks-dir outputs/release_audit/notebooks-refreshed
```

Figure-review notes survive extraction only when the image hash is unchanged.
Changed images require renewed review. Image decoding or an execution pass is
not scientific approval of a plot.

The current execution results and unresolved scientific/data limitations are
recorded in [FINDINGS.md](FINDINGS.md). Passing execution does not resolve missing
laboratory units, calibrated depth ties or acquisition provenance.

The manually triggered `release-verification.yml` workflow builds one candidate
and installs both artifacts on Windows, Linux and macOS with Python 3.9–3.13.
It has no publishing step. Local Windows results and a prepared workflow are
not evidence that the Linux/macOS jobs have passed.

`release_checks/approval.json` starts with pending/deferred requirements.
`python scripts/check_release_gate.py --dist DIST_DIRECTORY` refuses publication
unless every required category has PASS evidence, evidence hashes match, the
release-source fingerprint and version match, the checkout is clean, and both
artifact hashes match. The approval file is excluded from its own source
fingerprint to allow an approval record to be committed without a circular
commit-hash dependency. Generating an inventory never approves a release.

References consulted for the verification approach and initial checks:

- [Archie (1942), original paper](https://personal.ems.psu.edu/~radovic/EME590_Archie_1942.pdf).
- [Wyllie, Gregory and Gardner (1956)](https://doi.org/10.1190/1.1438217).
- [NIST unit conversion factors](https://www.nist.gov/pml/us-surveyfoot/revised-unit-conversion-factors).
- [PyPA TestPyPI workflow](https://packaging.python.org/en/latest/guides/using-testpypi/).
- [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

On publication, use the exact verified artifact hashes. A successful TestPyPI
upload alone does not verify that installation works or that the wheel has
passed scientific, security or redistribution review.
