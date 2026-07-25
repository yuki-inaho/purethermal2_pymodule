from purethermal2_pymodule.uvctypes import (
    PT_USB_PID,
    PT_USB_VID,
    UVC_FRAME_FORMAT_BGR,
    UVC_FRAME_FORMAT_I420,
    UVC_FRAME_FORMAT_RGB,
    UVC_FRAME_FORMAT_UYVY,
    UVC_FRAME_FORMAT_Y16,
)


def test_pt_usb_ids_match_purethermal_device():
    assert PT_USB_VID == 0x1E4E
    assert PT_USB_PID == 0x0100


def test_frame_format_constants_match_upstream_libuvc_enum():
    # These must match the `uvc_frame_format` enum ordering of upstream libuvc
    # (the libuvc0/libuvc-dev packages), not the old groupgets/libuvc fork.
    # A mismatch makes uvc_get_stream_ctrl_format_size()/uvc_start_streaming()
    # fail (observed as a bogus -51 return) even though the device opens fine.
    # See https://github.com/groupgets/purethermal1-uvc-capture/pull/34
    assert UVC_FRAME_FORMAT_UYVY == 4
    assert UVC_FRAME_FORMAT_RGB == 5
    assert UVC_FRAME_FORMAT_BGR == 6
    assert UVC_FRAME_FORMAT_Y16 == 10
    assert UVC_FRAME_FORMAT_I420 == 16
