import platform
from ctypes import POINTER, byref, cdll, CFUNCTYPE, cast
from ctypes import c_void_p, c_uint16

from purethermal2_pymodule.uvctypes import (
    uvc_context,
    uvc_device,
    uvc_device_handle,
    uvc_stream_ctrl,
    uvc_frame,
    print_device_info,
    print_device_formats,
    uvc_get_frame_formats_by_guid
)

from purethermal2_pymodule.uvctypes import PT_USB_PID, PT_USB_VID, VS_FMT_GUID_Y16, UVC_FRAME_FORMAT_Y16
from purethermal2_pymodule.color_map import generate_color_map, ColorMapType
from queue import Queue

import cv2
import numpy as np
from typing import Optional

# TODO: combine logger
# from purethermal2_pymodule.utils import get_logger_with_stdout
# logger = get_logger_with_stdout("PureThermal2")

try:
    if platform.system() == "Darwin":
        libuvc = cdll.LoadLibrary("libuvc.dylib")
    elif platform.system() == "Linux":
        libuvc = cdll.LoadLibrary("libuvc.so")
    else:
        libuvc = cdll.LoadLibrary("libuvc")
except OSError:
    print("Error: could not find libuvc!")
    exit(1)


def py_frame_callback(frame, userptr):
    array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    data = np.frombuffer(array_pointer.contents, dtype=np.dtype(np.uint16)).reshape(
        frame.contents.height, frame.contents.width
    )
    if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
        return
    if not q.full():
        q.put(data)


BUF_SIZE = 2
q = Queue(BUF_SIZE)
PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)


class PyPureThermal2:
    def __init__(self):
        self._ctx = POINTER(uvc_context)()
        self._dev = POINTER(uvc_device)()
        self._devh = POINTER(uvc_device_handle)()
        self._ctrl = uvc_stream_ctrl()
        self._open()

        self._thermal_image_colorized: Optional[np.ndarray]
        self._thermal_image_cercius: Optional[np.ndarray]

    def _open(self):
        try:
            res = libuvc.uvc_init(byref(self._ctx), 0)
            if res < 0:
                print("uvc_init error")
                exit(1)

            res = libuvc.uvc_find_device(self._ctx, byref(self._dev), PT_USB_VID, PT_USB_PID, 0)
            if res < 0:
                print("uvc_init error")
                exit(1)

            try:
                res = libuvc.uvc_open(self._dev, byref(self._devh))
                if res < 0:
                    ("uvc_open error")
                    exit(1)
                print("device opened!")

                frame_formats = uvc_get_frame_formats_by_guid(self._devh, VS_FMT_GUID_Y16)
                if len(frame_formats) == 0:
                    print("device does not support Y16")
                    exit(1)

                libuvc.uvc_get_stream_ctrl_format_size(
                    self._devh,
                    byref(self._ctrl),
                    UVC_FRAME_FORMAT_Y16,
                    frame_formats[0].wWidth,
                    frame_formats[0].wHeight,
                    int(1e7 / frame_formats[0].dwDefaultFrameInterval),
                )

                res = libuvc.uvc_start_streaming(self._devh, byref(self._ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
                if res < 0:
                    print("uvc_start_streaming failed: {0}".format(res))
                    exit(1)
                print("done starting stream")

            except SystemError:
                print("Failed to Open Device")

        except SystemError:
            print("Failed to Find Device")
            exit(1)

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
            cv2.cvtColor(np.uint8(data_processed), cv2.COLOR_GRAY2RGB), generate_color_map(colour_map_type)
        )
        return image_colorized

    def _get_frame(self, timeout=500) -> Optional[np.ndarray]:
        data = q.get(True, timeout)
        if data is None:
            return
        else:
            return data

    def _cvt_ktoc_ndarray(self, data: np.ndarray) -> np.ndarray:
        return (data.astype(np.float32) - 27315.0) / 100.0

    def update(self):
        data = self._get_frame().copy()
        status = data is not None
        if status:
            self._thermal_image_colorized = self._colorize_thermal_image(data)
            self._thermal_image_cercius = self._cvt_ktoc_ndarray(data)
        return status

    @property
    def thermal_image_colorized(self) -> Optional[np.ndarray]:
        return self._thermal_image_colorized

    @property
    def thermal_image_cercius(self) -> Optional[np.ndarray]:
        return self._thermal_image_cercius
