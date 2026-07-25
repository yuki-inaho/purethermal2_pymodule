import pytest

from purethermal2_pymodule.exceptions import (
    DeviceNotFoundError,
    DeviceOpenError,
    LibUVCNotFoundError,
    PureThermal2Error,
    StreamingError,
    UnsupportedFormatError,
)

ALL_EXCEPTION_TYPES = [
    LibUVCNotFoundError,
    DeviceNotFoundError,
    DeviceOpenError,
    UnsupportedFormatError,
    StreamingError,
]


@pytest.mark.parametrize("exc_type", ALL_EXCEPTION_TYPES)
def test_all_exception_types_derive_from_package_base(exc_type):
    assert issubclass(exc_type, PureThermal2Error)
    assert issubclass(PureThermal2Error, Exception)


@pytest.mark.parametrize("exc_type", ALL_EXCEPTION_TYPES)
def test_exception_types_are_constructible_with_no_arguments(exc_type):
    # Every exception has a sensible default message so call sites can raise
    # them tersely (e.g. `raise UnsupportedFormatError()`).
    err = exc_type()
    assert isinstance(err, PureThermal2Error)
    assert str(err)


@pytest.mark.parametrize(
    "exc_type", [DeviceNotFoundError, DeviceOpenError, StreamingError]
)
def test_exceptions_with_relevant_libuvc_code_store_and_report_it(exc_type):
    err = exc_type("something failed", code=-51)
    assert err.code == -51
    assert "-51" in str(err)


@pytest.mark.parametrize(
    "exc_type", [DeviceNotFoundError, DeviceOpenError, StreamingError]
)
def test_exceptions_without_code_have_code_none_and_no_code_in_message(exc_type):
    err = exc_type("something failed")
    assert err.code is None
    assert "code" not in str(err)


def test_catching_base_exception_catches_all_subclasses():
    for exc_type in ALL_EXCEPTION_TYPES:
        with pytest.raises(PureThermal2Error):
            raise exc_type()
