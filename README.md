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
