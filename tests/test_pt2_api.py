import numpy as np
import pytest

from purethermal2_pymodule import pt2_api
from purethermal2_pymodule.pt2_api import PyPureThermal2


@pytest.fixture
def camera():
    # Bypass __init__ (and therefore _open()'s real USB/libuvc device access)
    # so the pure image-processing logic can be tested without hardware.
    return object.__new__(PyPureThermal2)


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
