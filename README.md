# purethermal2_pymodule
![a colorized thermal image](https://github.com/yuki-inaho/purethermal2_pymodule/blob/main/thermal.png)

## Usage

This package requires the system `libuvc` shared library. On Debian/Ubuntu:

```bash
# Runtime only (ships libuvc.so.0):
sudo apt install libuvc0

# If you need the unversioned libuvc.so symlink (e.g. for linking against
# libuvc from other tools), also install the dev package:
sudo apt install libuvc-dev
```

`PyPureThermal2` is a context manager: it opens the device and starts
streaming on `__enter__`, and releases all libuvc resources on `__exit__`
(so it's safe to reopen a fresh instance right after).

```python
from purethermal2_pymodule import PyPureThermal2

with PyPureThermal2() as camera:
    for _ in range(10):
        if camera.update():
            print("max temperature (C):", camera.thermal_image_celsius.max())
```

`camera.thermal_image` holds the raw Y16 frame, `camera.thermal_image_colorized`
the false-colour (BGR) image, and `camera.thermal_image_celsius` the
per-pixel temperature in degrees Celsius.

`camera.update()` can occasionally return `False` on an early call even on
healthy hardware, which is why the example loops instead of checking it
once - but on known-healthy hardware this is rare (measured at 19/20 and
20/20 successful calls across two open/close sessions), not a long startup
delay to design around.

## Troubleshooting

**`update()` keeps returning `False`, or short/undersized frames are
logged persistently, even though the device opened successfully:** this is
the signature of the PureThermal board itself having wedged - its
isochronous (video) USB transfer stops while enumeration and control
transfers keep working, so `uvc_open`/`uvc_start_streaming` report success
with nothing actually wrong from libuvc's point of view. A USB-level reset
(`USBDEVFS_RESET`) does **not** clear this. Physically unplug and replug
the board (an actual power cycle) to recover it.
