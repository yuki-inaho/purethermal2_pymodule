import numpy as np
import pytest

from purethermal2_pymodule import pt2_api
from purethermal2_pymodule.exceptions import (
    DeviceNotFoundError,
    DeviceOpenError,
    StreamingError,
    UnsupportedFormatError,
)
from purethermal2_pymodule.pt2_api import PyPureThermal2


@pytest.fixture
def camera():
    # Bypass __init__ (and therefore _open()'s real USB/libuvc device access)
    # so the pure image-processing logic can be tested without hardware.
    return object.__new__(PyPureThermal2)


class _FakeFrameFormat:
    wWidth = 80
    wHeight = 60
    dwDefaultFrameInterval = 100000  # -> 100 fps, value is unimportant here


class FakeLibUVC:
    """Records calls and lets each libuvc entry point's return code be
    configured, so _open()'s error-handling/cleanup can be exercised without
    real hardware."""

    def __init__(self, init_res=0, find_res=0, open_res=0, ctrl_res=0, start_res=0):
        self.init_res = init_res
        self.find_res = find_res
        self.open_res = open_res
        self.ctrl_res = ctrl_res
        self.start_res = start_res
        self.calls = []

    def uvc_init(self, ctx_ref, usb_ctx):
        self.calls.append("init")
        return self.init_res

    def uvc_find_device(self, ctx, dev_ref, vid, pid, serial):
        self.calls.append("find_device")
        return self.find_res

    def uvc_open(self, dev, devh_ref):
        self.calls.append("open")
        return self.open_res

    def uvc_get_stream_ctrl_format_size(self, devh, ctrl_ref, fmt, w, h, interval):
        self.calls.append("ctrl_format_size")
        return self.ctrl_res

    def uvc_start_streaming(self, devh, ctrl_ref, cb, user_ptr, flags):
        self.calls.append("start_streaming")
        return self.start_res

    def uvc_stop_streaming(self, devh):
        self.calls.append("stop_streaming")

    def uvc_close(self, devh):
        self.calls.append("close")

    def uvc_unref_device(self, dev):
        self.calls.append("unref_device")

    def uvc_exit(self, ctx):
        self.calls.append("exit")


@pytest.fixture
def fake_frame_formats(monkeypatch):
    """Make _open() see a device that advertises one Y16 format."""
    monkeypatch.setattr(
        pt2_api,
        "uvc_get_frame_formats_by_guid",
        lambda devh, guid: [_FakeFrameFormat()],
    )


def test_open_raises_devicenotfounderror_when_uvc_init_fails(monkeypatch):
    fake = FakeLibUVC(init_res=-1)
    monkeypatch.setattr(pt2_api, "libuvc", fake)

    with pytest.raises(DeviceNotFoundError):
        PyPureThermal2()

    # uvc_init itself failed, so nothing was acquired and there is nothing to
    # unwind but the (never-initialized) context.
    assert fake.calls == ["init"]


def test_open_raises_devicenotfounderror_when_uvc_find_device_fails(monkeypatch):
    fake = FakeLibUVC(find_res=-1)
    monkeypatch.setattr(pt2_api, "libuvc", fake)

    with pytest.raises(DeviceNotFoundError):
        PyPureThermal2()

    assert fake.calls == ["init", "find_device", "exit"]


def test_open_raises_deviceopenerror_when_uvc_open_fails_and_cleans_up(monkeypatch):
    fake = FakeLibUVC(open_res=-1)
    monkeypatch.setattr(pt2_api, "libuvc", fake)

    with pytest.raises(DeviceOpenError):
        PyPureThermal2()

    assert fake.calls == ["init", "find_device", "open", "unref_device", "exit"]


def test_open_raises_unsupportedformaterror_when_device_has_no_y16(monkeypatch):
    fake = FakeLibUVC()
    monkeypatch.setattr(pt2_api, "libuvc", fake)
    monkeypatch.setattr(pt2_api, "uvc_get_frame_formats_by_guid", lambda devh, guid: [])

    with pytest.raises(UnsupportedFormatError):
        PyPureThermal2()

    assert fake.calls == [
        "init",
        "find_device",
        "open",
        "close",
        "unref_device",
        "exit",
    ]


def test_open_raises_streamingerror_when_stream_ctrl_format_size_fails(
    monkeypatch, fake_frame_formats
):
    # Regression coverage: uvc_get_stream_ctrl_format_size()'s return code
    # used to be silently ignored, which made bogus stream negotiations hard
    # to diagnose. It must now be checked and raise StreamingError.
    fake = FakeLibUVC(ctrl_res=-51)
    monkeypatch.setattr(pt2_api, "libuvc", fake)

    with pytest.raises(StreamingError) as excinfo:
        PyPureThermal2()

    assert excinfo.value.code == -51
    assert fake.calls == [
        "init",
        "find_device",
        "open",
        "ctrl_format_size",
        "close",
        "unref_device",
        "exit",
    ]


def test_open_raises_streamingerror_when_start_streaming_fails(
    monkeypatch, fake_frame_formats
):
    fake = FakeLibUVC(start_res=-1)
    monkeypatch.setattr(pt2_api, "libuvc", fake)

    with pytest.raises(StreamingError):
        PyPureThermal2()

    assert fake.calls == [
        "init",
        "find_device",
        "open",
        "ctrl_format_size",
        "start_streaming",
        "close",
        "unref_device",
        "exit",
    ]


def test_successful_open_close_and_context_manager_release_resources_in_order(
    monkeypatch, fake_frame_formats
):
    fake = FakeLibUVC()
    monkeypatch.setattr(pt2_api, "libuvc", fake)

    with PyPureThermal2() as cam:
        assert cam._streaming is True
        assert cam._closed is False
        assert fake.calls == [
            "init",
            "find_device",
            "open",
            "ctrl_format_size",
            "start_streaming",
        ]

    assert cam._closed is True
    assert cam._streaming is False
    assert fake.calls[-4:] == ["stop_streaming", "close", "unref_device", "exit"]


def test_close_is_idempotent_and_does_not_double_free(monkeypatch, fake_frame_formats):
    fake = FakeLibUVC()
    monkeypatch.setattr(pt2_api, "libuvc", fake)

    cam = PyPureThermal2()
    cam.close()
    calls_after_first_close = list(fake.calls)

    cam.close()
    cam.close()

    assert fake.calls == calls_after_first_close


def test_close_on_never_constructed_instance_is_a_safe_no_op(camera):
    # `camera` bypasses __init__ entirely (no resource-tracking attributes
    # exist at all), so close() must be guarded well enough not to raise, and
    # calling it repeatedly must still be a no-op.
    camera.close()
    camera.close()


def test_cvt_ktoc_ndarray_converts_kelvin_centikelvin_to_celsius(camera):
    raw = np.array([[27315, 30315]], dtype=np.uint16)  # 0.00C, 30.00C
    celsius = camera._cvt_ktoc_ndarray(raw)
    np.testing.assert_allclose(celsius, [[0.0, 30.0]])


def test_colorize_thermal_image_shape_and_dtype(camera):
    raw = np.arange(120 * 160, dtype=np.uint16).reshape(120, 160)
    colorized = camera._colorize_thermal_image(raw)
    assert colorized.shape == (120, 160, 3)
    assert colorized.dtype == np.uint8


def test_colorize_thermal_image_does_not_mutate_input(camera):
    raw = np.arange(120 * 160, dtype=np.uint16).reshape(120, 160)
    original = raw.copy()
    camera._colorize_thermal_image(raw)
    np.testing.assert_array_equal(raw, original)


def test_update_pulls_frame_from_queue_and_populates_properties(camera):
    raw_frame = np.full((120, 160), 30315, dtype=np.uint16)  # constant 30.00C
    pt2_api.q.put(raw_frame)
    try:
        status = camera.update()
    finally:
        # avoid leaking the frame into other tests sharing the module-level queue
        while not pt2_api.q.empty():
            pt2_api.q.get_nowait()

    assert status is True
    np.testing.assert_array_equal(camera.thermal_image, raw_frame)
    assert camera.thermal_image_colorized.shape == (120, 160, 3)
    np.testing.assert_allclose(camera.thermal_image_cercius, np.full((120, 160), 30.0))
