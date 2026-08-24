"""ctypes loader for the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
LIBRARY = ROOT / "dist" / "libmojo-embree.so"

I = ctypes.c_ssize_t

_SIGNATURES = {
    "me_build_bvh": ([I] * 11, I),
    "me_intersect_stream": ([I] * 16, None),
    "me_occluded_stream": ([I] * 12, None),
    "me_stack_size": ([], I),
}


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_EMBREE_MOJO")
    if override:
        return override.split()
    executable = shutil.which("mojo")
    if executable:
        return [executable]
    pixi = shutil.which("pixi")
    if pixi:
        return [pixi, "run", "--manifest-path", str(ROOT / "pixi.toml"), "mojo"]
    raise BuildError("Mojo was not found; run `pixi run build` first")


def build(force: bool = False) -> Path:
    sources = list(SRC.glob("*.mojo"))
    stale = not LIBRARY.exists() or any(
        source.stat().st_mtime > LIBRARY.stat().st_mtime for source in sources
    )
    if not force and not stale:
        return LIBRARY
    LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    command = _mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        "-I",
        str(SRC),
        str(SRC / "embree.mojo"),
        "-o",
        str(LIBRARY),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode or not LIBRARY.exists():
        raise BuildError((result.stderr or result.stdout).strip()[:6000])
    return LIBRARY


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(str(build()))
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library
