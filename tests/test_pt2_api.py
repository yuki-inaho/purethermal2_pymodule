from ctypes import POINTER, c_uint8, c_uint16, cast
from queue import Queue

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
    # Each instance owns its queue now (no more module-level pt2_api.q to
    # manually drain between tests): give this bare instance its own.
    camera._queue = Queue(pt2_api.PyPureThermal2._QUEUE_MAX_SIZE)
    raw_frame = np.full((120, 160), 30315, dtype=np.uint16)  # constant 30.00C
    camera._queue.put(raw_frame)

    status = camera.update()

    assert status is True
    np.testing.assert_array_equal(camera.thermal_image, raw_frame)
    assert camera.thermal_image_colorized.shape == (120, 160, 3)
    np.testing.assert_allclose(camera.thermal_image_celsius, np.full((120, 160), 30.0))


def test_thermal_image_cercius_is_a_deprecated_alias_for_celsius(camera):
    # `thermal_image_cercius` was a misspelling of `thermal_image_celsius`.
    # It must keep working (existing callers depend on it) but must warn so
    # they know to migrate.
    camera._queue = Queue(pt2_api.PyPureThermal2._QUEUE_MAX_SIZE)
    raw_frame = np.full((120, 160), 30315, dtype=np.uint16)  # constant 30.00C
    camera._queue.put(raw_frame)
    camera.update()

    with pytest.deprecated_call():
        aliased = camera.thermal_image_cercius

    np.testing.assert_array_equal(aliased, camera.thermal_image_celsius)


def test_get_frame_returns_none_on_timeout_instead_of_500s_block(camera):
    # Regression coverage: Queue.get(block, timeout) takes seconds, so the
    # old default of 500 was an ~8.3 minute wait, not 500ms. A short,
    # explicit timeout here must return quickly with None rather than
    # blocking or raising queue.Empty.
    camera._queue = Queue(2)
    assert camera._get_frame(timeout_s=0.01) is None


def test_update_returns_false_and_leaves_properties_untouched_when_no_frame(
    camera, monkeypatch
):
    # Regression coverage: update() used to do
    # `data = self._get_frame().copy()` and only afterwards check
    # `data is not None` - if _get_frame() ever returned None, `.copy()`
    # would already have raised AttributeError, so the None branch was dead
    # code. Force _get_frame() to report "no frame" and confirm update()
    # now handles that for real instead of blowing up.
    camera._queue = Queue(2)
    monkeypatch.setattr(camera, "_get_frame", lambda: None)
    sentinel = np.zeros((1, 1), dtype=np.uint16)
    camera._thermal_image_raw = sentinel

    status = camera.update()

    assert status is False
    assert camera._thermal_image_raw is sentinel


class _FakeFrameContents:
    """Stand-in for a real ``uvc_frame``'s ``.contents``.

    ``buf`` is the ctypes array backing the frame's pixel data - by keeping
    a handle to it separately from ``data``, tests can mutate it after the
    callback runs to simulate libuvc reusing/overwriting its internal frame
    buffer for the next capture.
    """

    def __init__(self, buf, width, height, data_bytes=None):
        self.buf = buf
        self.width = width
        self.height = height
        self.data = cast(buf, POINTER(c_uint8))
        self.data_bytes = (width * height * 2) if data_bytes is None else data_bytes


class _FakeFrame:
    def __init__(self, contents):
        self.contents = contents


def test_frame_callback_copies_data_so_reused_libuvc_buffer_cannot_corrupt_it():
    # Regression test for the use-after-free/aliasing bug: np.frombuffer()
    # over libuvc's frame buffer is only a VIEW. If the callback queues that
    # view instead of a copy, mutating the underlying buffer afterwards
    # (exactly what libuvc does when it reuses the buffer for the next
    # frame) silently corrupts whatever is sitting in the queue. This test
    # would fail if the callback stopped copying before queueing.
    width, height = 4, 3
    original_values = list(range(1, width * height + 1))
    buf = (c_uint16 * (width * height))(*original_values)
    frame = _FakeFrame(_FakeFrameContents(buf, width, height))

    target_queue = Queue(2)
    callback = pt2_api._make_frame_callback(target_queue)
    callback(frame, None)

    queued = target_queue.get_nowait()
    expected = np.array(original_values, dtype=np.uint16).reshape(height, width)
    np.testing.assert_array_equal(queued, expected)

    # Simulate libuvc reusing/overwriting its internal buffer for the next
    # captured frame, which happens for real as soon as the callback
    # returns control to libuvc.
    for i in range(len(buf)):
        buf[i] = 0xDEAD

    # The queued/consumed frame must still hold the original values - it
    # must not alias libuvc's (now-overwritten) memory.
    np.testing.assert_array_equal(queued, expected)


def test_frame_callback_rejects_short_data_bytes_and_queues_nothing():
    # Regression coverage: the data_bytes sanity check used to run *after*
    # the array was already cast/reshaped, so a short/corrupt frame still
    # paid for (and risked) building a bogus, possibly out-of-bounds view
    # before being discarded. It must now be rejected before any view is
    # built, and nothing should land in the queue.
    width, height = 4, 3
    buf = (c_uint16 * (width * height))(*range(width * height))
    short_data_bytes = width * height * 2 - 2  # one pixel short
    frame = _FakeFrame(_FakeFrameContents(buf, width, height, short_data_bytes))

    target_queue = Queue(2)
    callback = pt2_api._make_frame_callback(target_queue)
    callback(frame, None)

    assert target_queue.empty()


def test_two_instances_do_not_share_frame_queue(monkeypatch, fake_frame_formats):
    fake = FakeLibUVC()
    monkeypatch.setattr(pt2_api, "libuvc", fake)

    cam1 = PyPureThermal2()
    cam2 = PyPureThermal2()
    try:
        assert cam1._queue is not cam2._queue
        # Each instance's C callback trampoline must also be distinct (and
        # kept alive on the instance - see the comment in __init__).
        assert cam1._frame_callback is not cam2._frame_callback

        cam1._queue.put(np.zeros((2, 2), dtype=np.uint16))

        assert cam1._queue.qsize() == 1
        assert cam2._queue.empty()
    finally:
        cam1.close()
        cam2.close()
