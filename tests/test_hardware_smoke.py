from ctypes import POINTER, byref, c_void_p, cast

import pytest

import purethermal2_pymodule.uvctypes as uvctypes
from purethermal2_pymodule.uvctypes import (
    PT_USB_PID,
    PT_USB_VID,
    UVC_FRAME_FORMAT_Y16,
    VS_FMT_GUID_Y16,
    uvc_context,
    uvc_device,
    uvc_device_handle,
    uvc_get_frame_formats_by_guid,
    uvc_stream_ctrl,
)

libuvc = uvctypes.libuvc


def _purethermal_connected():
    ctx = POINTER(uvc_context)()
    if libuvc.uvc_init(byref(ctx), 0) != 0:
        return False
    try:
        dev = POINTER(uvc_device)()
        return libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0) == 0
    finally:
        libuvc.uvc_exit(ctx)


@pytest.mark.skipif(
    not _purethermal_connected(),
    reason="PureThermal/FLIR Lepton not connected over USB",
)
def test_start_streaming_succeeds_on_connected_device():
    ctx = POINTER(uvc_context)()
    dev = POINTER(uvc_device)()
    devh = POINTER(uvc_device_handle)()
    ctrl = uvc_stream_ctrl()

    assert libuvc.uvc_init(byref(ctx), 0) == 0
    try:
        assert libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0) == 0
        assert libuvc.uvc_open(dev, byref(devh)) == 0
        try:
            frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
            assert len(frame_formats) > 0
            fmt = frame_formats[0]

            res = libuvc.uvc_get_stream_ctrl_format_size(
                devh,
                byref(ctrl),
                UVC_FRAME_FORMAT_Y16,
                fmt.wWidth,
                fmt.wHeight,
                int(1e7 / fmt.dwDefaultFrameInterval),
            )
            assert res == 0
            assert ctrl.dwMaxVideoFrameSize == fmt.wWidth * fmt.wHeight * 2

            res = libuvc.uvc_start_streaming(
                devh, byref(ctrl), cast(None, c_void_p), None, 0
            )
            assert res == 0
            libuvc.uvc_stop_streaming(devh)
        finally:
            libuvc.uvc_close(devh)
            libuvc.uvc_unref_device(dev)
    finally:
        libuvc.uvc_exit(ctx)


@pytest.mark.skipif(
    not _purethermal_connected(),
    reason="PureThermal/FLIR Lepton not connected over USB",
)
def test_pypurethermal2_context_manager_opens_streams_and_releases_device():
    # End-to-end smoke test against real hardware: PyPureThermal2 must open
    # the device, start streaming, and fully release the device on __exit__
    # so a second instance can immediately reopen it (i.e. close() does not
    # leak the USB device handle / uvc context).
    from purethermal2_pymodule.pt2_api import PyPureThermal2

    with PyPureThermal2() as cam1:
        assert cam1._streaming is True
        assert cam1._closed is False

    assert cam1._closed is True
    assert cam1._streaming is False

    # Idempotent: closing an already-closed camera must not raise.
    cam1.close()

    # The device must be free again after close(), proving no resources
    # (context/device ref/handle) were leaked by the first instance.
    with PyPureThermal2() as cam2:
        assert cam2._streaming is True
