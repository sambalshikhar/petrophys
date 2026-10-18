"""Version information for petrophys."""

import re

__version__ = "3.0.0"
# Release numbers only: "3.0.0rc1" and "3.0.0.post1" give (3, 0, 0).
__version_info__ = tuple(
    int(x) for x in re.match(r"\d+(?:\.\d+)*", __version__).group(0).split(".")
)
