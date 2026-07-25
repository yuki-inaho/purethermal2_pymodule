import numpy as np
import pytest

from purethermal2_pymodule.color_map import ColorMapType, generate_color_map


@pytest.mark.parametrize("color_map_type", list(ColorMapType))
def test_generate_color_map_shape_and_dtype(color_map_type):
    color_lut = generate_color_map(color_map_type)
    assert color_lut.shape == (256, 1, 3)
    assert color_lut.dtype == np.uint8


def test_generate_color_map_default_is_ironblack():
    default_lut = generate_color_map()
    ironblack_lut = generate_color_map(ColorMapType.IRONBLACK)
    np.testing.assert_array_equal(default_lut, ironblack_lut)


def test_generate_color_map_unknown_type_falls_back_to_ironblack():
    fallback_lut = generate_color_map(color_map_type=object())
    ironblack_lut = generate_color_map(ColorMapType.IRONBLACK)
    np.testing.assert_array_equal(fallback_lut, ironblack_lut)


def test_generate_color_map_converts_rgb_to_bgr():
    # rainbow's first RGB triplet is [1, 3, 74]; the LUT must store it reversed as BGR.
    color_lut = generate_color_map(ColorMapType.RAINBOW)
    np.testing.assert_array_equal(color_lut[0, 0], [74, 3, 1])
