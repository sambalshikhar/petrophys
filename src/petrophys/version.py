"""Version information for petrophys."""

import re

__version__ = "3.0.0"
# Release numbers only: "3.0.0rc1" and "3.0.0.post1" give (3, 0, 0).
_release_match = re.match(r"\d+(?:\.\d+)*", __version__)
if _release_match is None:
    raise ValueError(f"Invalid release version: {__version__!r}")
__version_info__ = tuple(int(x) for x in _release_match.group(0).split("."))
