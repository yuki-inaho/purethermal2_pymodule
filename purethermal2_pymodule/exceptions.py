"""Exception hierarchy for purethermal2_pymodule.

All exceptions raised deliberately by this package derive from
:class:`PureThermal2Error`, so callers can catch that single type to handle
any camera/libuvc related failure.

Where a libuvc return code is relevant to the failure, it is stored on the
exception instance as ``.code`` and folded into the message.
"""


class PureThermal2Error(Exception):
    """Base class for all errors raised by purethermal2_pymodule."""

    def __init__(self, message: str, code: int | None = None):
        self.code = code
        if code is not None:
            message = f"{message} (libuvc code {code})"
        super().__init__(message)


class LibUVCNotFoundError(PureThermal2Error):
    """The libuvc shared library could not be loaded."""

    def __init__(self, message: str = "could not find/load libuvc shared library"):
        super().__init__(message)


class DeviceNotFoundError(PureThermal2Error):
    """No PureThermal device was found on the USB bus."""

    def __init__(
        self, message: str = "no PureThermal device found", code: int | None = None
    ):
        super().__init__(message, code)


class DeviceOpenError(PureThermal2Error):
    """uvc_open() failed to open the device handle."""

    def __init__(
        self,
        message: str = "failed to open PureThermal device",
        code: int | None = None,
    ):
        super().__init__(message, code)


class UnsupportedFormatError(PureThermal2Error):
    """The device does not advertise the Y16 stream format."""

    def __init__(self, message: str = "device does not support Y16 format"):
        super().__init__(message)


class StreamingError(PureThermal2Error):
    """Stream ctrl negotiation or start_streaming failed."""

    def __init__(
        self,
        message: str = "failed to negotiate/start streaming",
        code: int | None = None,
    ):
        super().__init__(message, code)
