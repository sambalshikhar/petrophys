"""Write the reference signatures the reference test compares against.

Run it after a change that is meant to move the numbers, check the JSON diff
in ``tests/reference/``, and commit the files with the change::

    python tests/make_reference.py            # every scenario
    python tests/make_reference.py 30_6-5     # named scenarios only
"""

import json
import sys

from reference_cases import REFERENCE_DIR, SCENARIOS, reference_path, run


def main(names):
    unknown = sorted(set(names) - set(SCENARIOS))
    if unknown:
        raise SystemExit(f"Unknown scenarios {unknown}; choose from {sorted(SCENARIOS)}")
    REFERENCE_DIR.mkdir(exist_ok=True)
    for name in names or SCENARIOS:
        path = reference_path(name)
        path.write_text(json.dumps(run(name), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(REFERENCE_DIR.parent.parent)}")


if __name__ == "__main__":
    main(sys.argv[1:])
