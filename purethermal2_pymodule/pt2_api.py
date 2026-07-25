import warnings
from collections.abc import Callable
from ctypes import CFUNCTYPE, POINTER, byref, c_uint16, c_void_p, cast
from queue import Empty, Queue

import cv2
import numpy as np

from purethermal2_pymodule.color_map import ColorMapType, generate_color_map
from purethermal2_pymodule.exceptions import (
    DeviceNotFoundError,
    DeviceOpenError,
    StreamingError,
    UnsupportedFormatError,
)
from purethermal2_pymodule.libuvc_loader import libuvc
from purethermal2_pymodule.utils import get_logger_with_stdout
from purethermal2_pymodule.uvctypes import (
    PT_USB_PID,
    PT_USB_VID,
    UVC_FRAME_FORMAT_Y16,
    VS_FMT_GUID_Y16,
    print_device_formats,
    print_device_info,
    uvc_context,
    uvc_device,
    uvc_device_handle,
    uvc_frame,
    uvc_get_frame_formats_by_guid,
    uvc_stream_ctrl,
)

logger = get_logger_with_stdout("PureThermal2")

FrameCallback = Callable[[POINTER(uvc_frame), c_void_p], None]


def _make_frame_callback(target_queue: Queue) -> FrameCallback:
    """Build a libuvc frame callback bound to one instance's queue.

    Each :class:`PyPureThermal2` instance gets its own callback (closing
    over that instance's ``target_queue``) instead of every instance
    fighting over one module-global queue.

    The callback also closes over ``seen_complete_frame``, a per-callback
    (i.e. per-instance - a fresh callback/closure is built for every
    :class:`PyPureThermal2`) flag tracking whether a full, correctly-sized
    frame has been seen yet. Immediately after ``uvc_start_streaming()``,
    libuvc reliably delivers a handful of *partial* frames while the
    isochronous USB transfer ramps up - empirically, clean fractions of a
    full frame (e.g. 1/40, 1/4, 1/2 of the expected byte count) arriving in
    the first ~0.5s. That is normal stream-startup behaviour, not a fault,
    so those are logged at DEBUG. Once a complete frame has arrived, the
    stream is established, and any short frame after that point is a real
    anomaly (e.g. a USB dropout mid-stream) that must stay visible at
    WARNING. Because the flag lives in this closure rather than on
    ``self`` or a module global, a camera that is closed and reopened (a
    new :class:`PyPureThermal2` instance) automatically gets its own fresh
    grace period.
    """
    seen_complete_frame = False

    def _frame_callback(frame, userptr):
        nonlocal seen_complete_frame
        contents = frame.contents
        expected_bytes = 2 * contents.width * contents.height
        # Validate *before* building any view over the buffer: data_bytes
        # can be short/corrupt (e.g. a truncated USB transfer), and casting
        # + reshaping to the full width*height extent first would build a
        # view that reads past the end of a too-small buffer.
        if contents.data_bytes != expected_bytes:
            if seen_complete_frame:
                # A complete frame already arrived, so the stream was
                # established - a short frame now is a genuine fault, not
                # startup noise.
                logger.warning(
                    "dropping frame: expected %d bytes for %dx%d Y16 frame, got %d",
                    expected_bytes,
                    contents.width,
                    contents.height,
                    contents.data_bytes,
                )
            else:
                # No complete frame yet: almost certainly the normal
                # isochronous stream ramp-up transient, so keep it quiet at
                # DEBUG instead of alarming callers on every open().
                logger.debug(
                    "dropping short frame during stream startup: expected "
                    "%d bytes for %dx%d Y16 frame, got %d (normal while the "
                    "isochronous stream ramps up)",
                    expected_bytes,
                    contents.width,
                    contents.height,
                    contents.data_bytes,
                )
            return

        seen_complete_frame = True

        array_pointer = cast(
            contents.data,
            POINTER(c_uint16 * (contents.width * contents.height)),
        )
        # np.frombuffer() only ever creates a VIEW over libuvc's internal
        # frame buffer. libuvc reuses/overwrites that same buffer for the
        # *next* captured frame as soon as this callback returns, so a
        # queued view would be silently torn/overwritten later (verified
        # empirically). `.copy()` forces an independent allocation right
        # now, while the data is still valid, so nothing libuvc does
        # afterwards can mutate what we hand off to the queue/consumer.
        data = (
            np.frombuffer(array_pointer.contents, dtype=np.uint16)
            .reshape(contents.height, contents.width)
            .copy()
        )
        if not target_queue.full():
            target_queue.put(data)

    return _frame_callback


class PyPureThermal2:
    """ctypes/libuvc binding for a FLIR Lepton / PureThermal2 USB camera.

    Note on startup: right after the device is opened, libuvc's isochronous
    USB stream needs a brief moment (empirically well under a second) to
    ramp up, during which it delivers a handful of undersized/partial
    frames that are silently dropped (see :func:`_make_frame_callback`).
    As a result, the first several calls to :meth:`update` after opening -
    including via the ``with PyPureThermal2() as ...`` context manager -
    can legitimately return ``False`` even though nothing is wrong.
    Callers should loop on :meth:`update` (as :mod:`example.main` does)
    rather than treat a single ``False`` as a failure.
    """

    # Small bound on in-flight frames: the callback drops new frames rather
    # than blocking libuvc's capture thread once this many are queued.
    _QUEUE_MAX_SIZE = 2

    def __init__(self):
        self._ctx = POINTER(uvc_context)()
        self._dev = POINTER(uvc_device)()
        self._devh = POINTER(uvc_device_handle)()
        self._ctrl = uvc_stream_ctrl()

        # Resource-acquisition flags used by close() to figure out what needs
        # to be torn down. Set as each libuvc resource is successfully
        # acquired so close() stays correct even if __init__ raises partway
        # through _open().
        self._ctx_initialized = False
        self._dev_found = False
        self._devh_opened = False
        self._streaming = False
        self._closed = False

        self._thermal_image_raw: np.ndarray | None = None
        self._thermal_image_colorized: np.ndarray | None = None
        self._thermal_image_celsius: np.ndarray | None = None

        # Each instance gets its own queue (no more module-global queue
        # shared - and fought over - by every PyPureThermal2 instance), and
        # its own callback closure bound to that queue.
        self._queue: Queue = Queue(self._QUEUE_MAX_SIZE)

        # The C side (libuvc) only ever sees a raw function pointer, not a
        # Python reference, so nothing stops the CFUNCTYPE trampoline object
        # from being garbage-collected once __init__ returns *unless* we
        # keep our own reference to it. If that happened while libuvc still
        # holds the pointer (i.e. any time between uvc_start_streaming and
        # uvc_stop_streaming), the next captured frame would call through a
        # dangling trampoline and segfault the whole process. Keeping it on
        # `self` ties its lifetime to the camera instance's.
        self._frame_callback = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(
            _make_frame_callback(self._queue)
        )

        self._open()

    def _open(self):
        try:
            res = libuvc.uvc_init(byref(self._ctx), 0)
            if res < 0:
                raise DeviceNotFoundError("uvc_init failed", code=res)
            self._ctx_initialized = True

            res = libuvc.uvc_find_device(
                self._ctx, byref(self._dev), PT_USB_VID, PT_USB_PID, 0
            )
            if res < 0:
                raise DeviceNotFoundError("no PureThermal device found", code=res)
            self._dev_found = True

            res = libuvc.uvc_open(self._dev, byref(self._devh))
            if res < 0:
                raise DeviceOpenError("uvc_open failed", code=res)
            self._devh_opened = True
            logger.info("device opened!")

            frame_formats = uvc_get_frame_formats_by_guid(self._devh, VS_FMT_GUID_Y16)
            if len(frame_formats) == 0:
                raise UnsupportedFormatError()

            res = libuvc.uvc_get_stream_ctrl_format_size(
                self._devh,
                byref(self._ctrl),
                UVC_FRAME_FORMAT_Y16,
                frame_formats[0].wWidth,
                frame_formats[0].wHeight,
                int(1e7 / frame_formats[0].dwDefaultFrameInterval),
            )
            if res < 0:
                raise StreamingError("uvc_get_stream_ctrl_format_size failed", code=res)

            res = libuvc.uvc_start_streaming(
                self._devh, byref(self._ctrl), self._frame_callback, None, 0
            )
            if res < 0:
                raise StreamingError("uvc_start_streaming failed", code=res)
            self._streaming = True
            logger.info("done starting stream")
        except Exception:
            # Don't leak whatever we already acquired if a later step fails.
            self.close()
            raise

    def close(self):
        """Tear down whatever libuvc resources were acquired.

        Safe to call on a partially-constructed instance (e.g. when _open()
        fails partway through) and safe to call more than once.
        """
        if getattr(self, "_closed", False):
            return

        if getattr(self, "_streaming", False) and getattr(self, "_devh_opened", False):
            libuvc.uvc_stop_streaming(self._devh)
        self._streaming = False

        if getattr(self, "_devh_opened", False):
            libuvc.uvc_close(self._devh)
        self._devh_opened = False

        if getattr(self, "_dev_found", False):
            libuvc.uvc_unref_device(self._dev)
        self._dev_found = False

        if getattr(self, "_ctx_initialized", False):
            libuvc.uvc_exit(self._ctx)
        self._ctx_initialized = False

        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def show_stream_info(self):
        print_device_info(self._devh)
        print_device_formats(self._devh)

    def _colorize_thermal_image(
        self, data: np.ndarray, colour_map_type: ColorMapType = ColorMapType.IRONBLACK
    ) -> np.ndarray:
        # Only one scratch buffer is needed: cv2.normalize() reads `data` and
        # writes into `data_processed`, it never writes back into its source,
        # so `data` itself can be passed directly as the source instead of
        # copying it first. `data_processed` is allocated with
        # np.empty_like() rather than data.copy() since its initial contents
        # are irrelevant - cv2.normalize() overwrites all of it.
        data_processed = np.empty_like(data)
        cv2.normalize(data, data_processed, 0, 65535, cv2.NORM_MINMAX)
        np.right_shift(data_processed, 8, data_processed)
        image_colorized = cv2.LUT(
            cv2.cvtColor(np.uint8(data_processed), cv2.COLOR_GRAY2RGB),
            generate_color_map(colour_map_type),
        )
        return image_colorized

    def _get_frame(self, timeout_s: float = 0.5) -> np.ndarray | None:
        """Pop the most recently queued frame, waiting up to ``timeout_s``.

        ``timeout_s`` is a timeout in *seconds* (``Queue.get()``'s native
        unit - it used to be passed a value of 500 intending milliseconds,
        which actually meant an 8-minute-plus wait). If no frame becomes
        available within ``timeout_s``, ``None`` is returned rather than
        letting ``queue.Empty`` escape, so callers (namely ``update()``) can
        treat "no frame yet" as a normal, checkable condition.
        """
        try:
            return self._queue.get(True, timeout_s)
        except Empty:
            return None

    def _cvt_ktoc_ndarray(self, data: np.ndarray) -> np.ndarray:
        return (data.astype(np.float32) - 27315.0) / 100.0

    def update(self) -> bool:
        """Pull the latest frame off the queue and refresh derived images.

        Returns True if a new frame was available and the thermal_image*
        properties were updated, or False if no frame arrived within
        _get_frame()'s timeout (in which case the properties keep whatever
        they already held).

        Immediately after opening the device, the isochronous USB stream is
        still ramping up and only delivers partial frames for a brief
        moment, so it is normal for the first several calls to return
        False. Callers should keep looping rather than treat one False as
        an error - see the class docstring.
        """
        frame = self._get_frame()
        if frame is None:
            return False
        self._thermal_image_raw = frame
        self._thermal_image_colorized = self._colorize_thermal_image(frame)
        self._thermal_image_celsius = self._cvt_ktoc_ndarray(frame)
        return True

    @property
    def thermal_image(self) -> np.ndarray | None:
        return self._thermal_image_raw

    @property
    def thermal_image_colorized(self) -> np.ndarray | None:
        return self._thermal_image_colorized

    @property
    def thermal_image_celsius(self) -> np.ndarray | None:
        return self._thermal_image_celsius

    @property
    def thermal_image_cercius(self) -> np.ndarray | None:
        """Deprecated misspelled alias for :attr:`thermal_image_celsius`.

        Kept for backwards compatibility with existing callers; new code
        should use :attr:`thermal_image_celsius` instead.
        """
        warnings.warn(
            "thermal_image_cercius is deprecated and will be removed in a "
            "future release; use thermal_image_celsius instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.thermal_image_celsius
