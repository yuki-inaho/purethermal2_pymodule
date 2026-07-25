"""purethermal2_pymodule: a ctypes/libuvc binding for the PureThermal2/FLIR
Lepton USB thermal camera.

The most commonly needed names are re-exported here so callers can do::

    from purethermal2_pymodule import PyPureThermal2, ColorMapType

instead of reaching into the individual submodules.
"""

from purethermal2_pymodule.color_map import ColorMapType
from purethermal2_pymodule.exceptions import (
    DeviceNotFoundError,
    DeviceOpenError,
    LibUVCNotFoundError,
    PureThermal2Error,
    StreamingError,
    UnsupportedFormatError,
)
from purethermal2_pymodule.pt2_api import PyPureThermal2

__all__ = [
    "ColorMapType",
    "DeviceNotFoundError",
    "DeviceOpenError",
    "LibUVCNotFoundError",
    "PureThermal2Error",
    "PyPureThermal2",
    "StreamingError",
    "UnsupportedFormatError",
]
