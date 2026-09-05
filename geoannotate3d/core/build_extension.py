"""
core/build_extension.py — Compila _fastcore como extensión Python+NumPy.
Funciona en Windows (MSVC/MinGW), Linux y macOS.

Uso:
    python core/build_extension.py
"""
import os, sys, subprocess, shutil
from pathlib import Path

HERE = Path(__file__).parent


def build_fastcore(force=False):
    src = HERE / "_fastcore.c"
    if not src.exists():
        print("[fastcore] _fastcore.c no encontrado")
        return False

    import sysconfig
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = HERE / f"_fastcore{suffix}"

    if not force and out.exists():
        if out.stat().st_mtime >= src.stat().st_mtime:
            print(f"[fastcore] Ya compilado: {out.name}")
            return True

    print("[fastcore] Compilando _fastcore con NumPy C API…", flush=True)

    # Method 1: setuptools
    try:
        _build_setuptools(src)
        print("[fastcore] Compilado OK ✓")
        return True
    except Exception as e1:
        print(f"[fastcore] setuptools: {e1}")

    # Method 2: gcc directo
    try:
        _build_gcc(src, HERE / "_fastcore.so")
        print("[fastcore] gcc OK ✓")
        return True
    except Exception as e2:
        print(f"[fastcore] gcc: {e2}")

    print("[fastcore] Falló — usando numpy fallback (más lento)")
    return False


def _build_setuptools(src):
    import numpy as np
    setup_py = HERE / "_tmp_setup.py"
    setup_py.write_text(f"""
import sys
sys.argv = ['setup', 'build_ext', '--inplace']
from setuptools import setup, Extension
import numpy as np
ext = Extension(
    '_fastcore',
    sources=[r'{src}'],
    include_dirs=[np.get_include()],
    extra_compile_args=['/O2'] if sys.platform=='win32' else ['-O3','-ffast-math','-march=native','-std=c99'],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    language='c',
)
setup(name='_fastcore', ext_modules=[ext],
      script_args=['build_ext','--inplace','--build-lib',r'{HERE}'])
""")
    try:
        r = subprocess.run([sys.executable, str(setup_py)], cwd=str(HERE),
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-1000:])
    finally:
        setup_py.unlink(missing_ok=True)
        for d in ["build", "__pycache__", "_fastcore.egg-info"]:
            p = HERE / d
            if p.exists(): shutil.rmtree(p, True)


def _build_gcc(src, out):
    import sysconfig, numpy as np
    inc = sysconfig.get_path("include")
    np_inc = np.get_include()
    cmd = ["gcc", "-O3", "-march=native", "-ffast-math", "-std=c99", "-fPIC", "-shared",
           f"-I{inc}", f"-I{np_inc}",
           "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
           str(src), "-o", str(out), "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)


if __name__ == "__main__":
    sys.exit(0 if build_fastcore(force=True) else 1)


def build_if_needed():
    """Call from main.py to compile extension if .so is missing."""
    import importlib.util, os
    from pathlib import Path
    src = Path(__file__).parent / "_fastcore.c"
    # Check if .so exists and is newer than .c
    pattern = "_fastcore*.so"
    import glob
    so_files = glob.glob(str(Path(__file__).parent / pattern))
    if not so_files:
        main()
    else:
        so_time = max(os.path.getmtime(f) for f in so_files)
        if src.exists() and os.path.getmtime(src) > so_time:
            main()
