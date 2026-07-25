from ctypes import POINTER, byref, CFUNCTYPE, cast
from ctypes import c_void_p, c_uint16

from purethermal2_pymodule.uvctypes import (
    uvc_context,
    uvc_device,
    uvc_device_handle,
    uvc_stream_ctrl,
    uvc_frame,
    print_device_info,
    print_device_formats,
    uvc_get_frame_formats_by_guid,
)

from purethermal2_pymodule.uvctypes import (
    PT_USB_PID,
    PT_USB_VID,
    VS_FMT_GUID_Y16,
    UVC_FRAME_FORMAT_Y16,
)
from purethermal2_pymodule.color_map import generate_color_map, ColorMapType
from purethermal2_pymodule.exceptions import (
    DeviceNotFoundError,
    DeviceOpenError,
    StreamingError,
    UnsupportedFormatError,
)
from purethermal2_pymodule.libuvc_loader import libuvc
from purethermal2_pymodule.utils import get_logger_with_stdout
from queue import Empty, Queue

import cv2
import numpy as np
from collections.abc import Callable
from typing import Optional

logger = get_logger_with_stdout("PureThermal2")

FrameCallback = Callable[[POINTER(uvc_frame), c_void_p], None]


def _make_frame_callback(target_queue: Queue) -> FrameCallback:
    """Build a libuvc frame callback bound to one instance's queue.

    Each :class:`PyPureThermal2` instance gets its own callback (closing
    over that instance's ``target_queue``) instead of every instance
    fighting over one module-global queue.
    """

    def _frame_callback(frame, userptr):
        contents = frame.contents
        expected_bytes = 2 * contents.width * contents.height
        # Validate *before* building any view over the buffer: data_bytes
        # can be short/corrupt (e.g. a truncated USB transfer), and casting
        # + reshaping to the full width*height extent first would build a
        # view that reads past the end of a too-small buffer.
        if contents.data_bytes != expected_bytes:
            logger.warning(
                "dropping frame: expected %d bytes for %dx%d Y16 frame, got %d",
                expected_bytes,
                contents.width,
                contents.height,
                contents.data_bytes,
            )
            return

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

        self._thermal_image_raw: Optional[np.ndarray] = None
        self._thermal_image_colorized: Optional[np.ndarray] = None
        self._thermal_image_cercius: Optional[np.ndarray] = None

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
        data_copy = data.copy()
        data_processed = data.copy()
        cv2.normalize(data_copy, data_processed, 0, 65535, cv2.NORM_MINMAX)
        np.right_shift(data_processed, 8, data_processed)
        image_colorized = cv2.LUT(
            cv2.cvtColor(np.uint8(data_processed), cv2.COLOR_GRAY2RGB),
            generate_color_map(colour_map_type),
        )
        return image_colorized

    def _get_frame(self, timeout_s: float = 0.5) -> Optional[np.ndarray]:
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
        """
        frame = self._get_frame()
        if frame is None:
            return False
        self._thermal_image_raw = frame
        self._thermal_image_colorized = self._colorize_thermal_image(frame)
        self._thermal_image_cercius = self._cvt_ktoc_ndarray(frame)
        return True

    @property
    def thermal_image(self) -> Optional[np.ndarray]:
        return self._thermal_image_raw

    @property
    def thermal_image_colorized(self) -> Optional[np.ndarray]:
        return self._thermal_image_colorized

    @property
    def thermal_image_cercius(self) -> Optional[np.ndarray]:
        return self._thermal_image_cercius
