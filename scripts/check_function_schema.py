"""Check the declared parameters of the public analysis functions.

Every public function of :mod:`petrophys.interpretation`,
:mod:`petrophys.evaluation` and :mod:`petrophys.preprocessing` must declare
every parameter, by a known kind, with standard curve names
(:func:`petrophys.function_schema.check`). Prints each problem and exits 1
when there is any.

    python scripts/check_function_schema.py
"""

import sys

from petrophys.function_schema import check, public_functions


def main() -> int:
    problems = check()
    for problem in problems:
        print(problem)
    n = len(public_functions())
    if problems:
        print(f"{len(problems)} problem(s) in the declarations of {n} functions")
        return 1
    print(f"the declarations of {n} public functions match their signatures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
