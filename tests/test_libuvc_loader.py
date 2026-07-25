from ctypes import cdll

import pytest

from purethermal2_pymodule import libuvc_loader
from purethermal2_pymodule.exceptions import LibUVCNotFoundError, PureThermal2Error


def test_candidate_names_for_linux_tries_so_then_versioned_so():
    # "libuvc.so" only exists when libuvc-dev is installed; runtime-only
    # installs (libuvc0) only ship "libuvc.so.0", so both must be tried,
    # unversioned first.
    assert libuvc_loader._candidate_names("Linux") == ["libuvc.so", "libuvc.so.0"]


def test_candidate_names_for_darwin_is_dylib():
    assert libuvc_loader._candidate_names("Darwin") == ["libuvc.dylib"]


def test_candidate_names_for_other_platforms_falls_back_to_bare_name():
    assert libuvc_loader._candidate_names("Windows") == ["libuvc"]


def test_module_level_libuvc_was_loaded_successfully():
    # This machine has libuvc installed, so importing the package must have
    # succeeded and bound a usable handle rather than raising/exiting.
    assert libuvc_loader.libuvc is not None


def test_load_libuvc_tries_candidates_in_order_and_returns_first_success(monkeypatch):
    attempted = []

    def fake_load_library(name):
        attempted.append(name)
        if name == "libuvc.so":
            raise OSError("no unversioned symlink")
        return f"handle-for-{name}"

    monkeypatch.setattr(cdll, "LoadLibrary", fake_load_library)

    result = libuvc_loader._load_libuvc(["libuvc.so", "libuvc.so.0"])

    assert attempted == ["libuvc.so", "libuvc.so.0"]
    assert result == "handle-for-libuvc.so.0"


def test_load_libuvc_raises_libuvcnotfounderror_when_all_candidates_fail(monkeypatch):
    def always_fail(name):
        raise OSError(f"cannot load {name}")

    monkeypatch.setattr(cdll, "LoadLibrary", always_fail)

    with pytest.raises(LibUVCNotFoundError) as excinfo:
        libuvc_loader._load_libuvc(["libuvc.so", "libuvc.so.0"])

    # It must not print-and-exit(); it must raise our exception type, which
    # derives from the package base exception.
    assert isinstance(excinfo.value, PureThermal2Error)
    assert "libuvc.so" in str(excinfo.value)
    assert "libuvc.so.0" in str(excinfo.value)
