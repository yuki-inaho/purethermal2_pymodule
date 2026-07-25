import numpy as np
import pytest

from purethermal2_pymodule.color_map import (
    ColorMapType,
    _generate_color_map_cached,
    generate_color_map,
)


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


def test_generate_color_map_repeated_calls_return_equal_but_independent_luts():
    first = generate_color_map(ColorMapType.GRAYSCALE)
    second = generate_color_map(ColorMapType.GRAYSCALE)
    np.testing.assert_array_equal(first, second)
    # Each call must hand back its own array, not the same object, so that
    # mutating one caller's copy can never be observed by another caller.
    assert first is not second


def test_generate_color_map_underlying_generation_happens_once_per_type():
    # The LUT-building step (list comprehension over 768 elements + numpy
    # conversions) is the expensive part this cache exists to avoid re-doing
    # on every frame. Warm the cache for this type first, then confirm
    # further calls are pure cache hits (no new misses) regardless of what
    # other tests already touched the shared, module-level cache.
    generate_color_map(ColorMapType.IRONBLACK)
    misses_before = _generate_color_map_cached.cache_info().misses

    generate_color_map(ColorMapType.IRONBLACK)
    generate_color_map(ColorMapType.IRONBLACK)
    generate_color_map(ColorMapType.IRONBLACK)

    assert _generate_color_map_cached.cache_info().misses == misses_before


def test_generate_color_map_cache_cannot_be_poisoned_by_caller_mutation():
    lut = generate_color_map(ColorMapType.RAINBOW)
    lut[:] = 0  # attempt to corrupt whatever generate_color_map() hands out

    fresh = generate_color_map(ColorMapType.RAINBOW)

    assert not np.all(fresh == 0)
    np.testing.assert_array_equal(fresh, generate_color_map(ColorMapType.RAINBOW))


def test_generate_color_map_cached_backing_array_is_read_only():
    cached = _generate_color_map_cached(ColorMapType.IRONBLACK)
    with pytest.raises(ValueError):
        cached[0, 0, 0] = 123
