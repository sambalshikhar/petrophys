"""Compatibility namespace for preprocessing imputation."""

from petrophys._compat import reexport_package

reexport_package(
    globals(),
    "petrophys.preprocessing.imputation",
    {
        "curve_imputer": "petrophys.preprocessing.imputation.curve_imputer",
        "pretrained": "petrophys.preprocessing.imputation.pretrained",
        "training": "petrophys.preprocessing.imputation.training",
    },
)
