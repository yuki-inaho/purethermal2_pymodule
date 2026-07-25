"""Confirm the package root re-exports the intended public API.

Before this, __init__.py was empty and callers were forced to reach into
purethermal2_pymodule.pt2_api / .color_map / .exceptions directly.
"""

import purethermal2_pymodule as pt2
from purethermal2_pymodule import (
    ColorMapType,
    DeviceNotFoundError,
    DeviceOpenError,
    LibUVCNotFoundError,
    PureThermal2Error,
    PyPureThermal2,
    StreamingError,
    UnsupportedFormatError,
)


def test_all_advertised_names_are_actually_importable_from_package_root():
    for name in pt2.__all__:
        assert hasattr(pt2, name), f"{name} listed in __all__ but not importable"


def test_reexported_names_are_the_same_objects_as_their_submodule_originals():
    from purethermal2_pymodule.color_map import ColorMapType as SubColorMapType
    from purethermal2_pymodule.exceptions import (
        DeviceNotFoundError as SubDeviceNotFoundError,
    )
    from purethermal2_pymodule.pt2_api import PyPureThermal2 as SubPyPureThermal2

    assert ColorMapType is SubColorMapType
    assert DeviceNotFoundError is SubDeviceNotFoundError
    assert PyPureThermal2 is SubPyPureThermal2


def test_exception_types_reexported_at_root_still_derive_from_package_base():
    for exc_type in (
        LibUVCNotFoundError,
        DeviceNotFoundError,
        DeviceOpenError,
        UnsupportedFormatError,
        StreamingError,
    ):
        assert issubclass(exc_type, PureThermal2Error)
