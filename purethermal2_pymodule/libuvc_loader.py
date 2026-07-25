"""Loads the libuvc shared library exactly once for the whole package.

Both ``uvctypes.py`` and ``pt2_api.py`` import the same ``libuvc`` object
from this module instead of each calling ``cdll.LoadLibrary`` themselves.
"""

import platform
from ctypes import cdll

from purethermal2_pymodule.exceptions import LibUVCNotFoundError


def _candidate_names(system: str) -> list[str]:
    """Shared-library names to try, in order, for the given platform.system()."""
    if system == "Darwin":
        return ["libuvc.dylib"]
    elif system == "Linux":
        # "libuvc.so" (the unversioned symlink) only exists when libuvc-dev is
        # installed. Runtime-only installs (the libuvc0 package) only ship the
        # versioned "libuvc.so.0", so that must be tried as a fallback.
        return ["libuvc.so", "libuvc.so.0"]
    else:
        return ["libuvc"]


def _load_libuvc(names: list[str]):
    errors = []
    for name in names:
        try:
            return cdll.LoadLibrary(name)
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    raise LibUVCNotFoundError(
        "could not find/load libuvc shared library; tried: " + "; ".join(errors)
    )


libuvc = _load_libuvc(_candidate_names(platform.system()))
