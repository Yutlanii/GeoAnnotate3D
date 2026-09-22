═══════════════════════════════════════════════════════════════════
 GeoAnnotate3D — Instructions for building the executable
═══════════════════════════════════════════════════════════════════

PREREQUISITES (install once):
─────────────────────────────────────────
1. Python 3.11 or 3.12 (https://python.org/downloads)
   ✓ Check "Add Python to PATH" during install

2. Visual Studio Build Tools 2022 (free C++ compiler)
   https://visualstudio.microsoft.com/visual-cpp-build-tools/
   ✓ Select: "Desktop development with C++"

3. Inno Setup 6 (https://jrsoftware.org/isinfo.php) — needed to produce
   the installer, not just the raw standalone folder.

4. Build from a CLEAN, DEDICATED virtual environment — not a conda env
   or venv you use for other projects. Confirmed the hard way: building
   from a "daily driver" environment let unrelated packages (TensorFlow,
   boto3, opencv, dash, grpc...) leak in via Windows' per-user
   site-packages, and Nuitka tried to compile all of it — thousands of
   extra C files, slower, and one more thing to break for no reason:

     python -m venv build_env
     build_env\Scripts\activate
     pip install nuitka ordered-set zstandard imageio
     pip install -r requirements.txt
     pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

   (adjust the torch line for the CUDA/CPU build you actually want —
   see https://pytorch.org/get-started/locally/)

BUILDING:
─────────────────────────────────────────
1. Open CMD (not PowerShell — the script is .bat) with build_env active:
   build_env\Scripts\activate.bat

2. Navigate to the project folder:
   cd C:\path\to\GeoAnnotate3D\geoannotate3d

3. Run:
   build_exe.bat

4. Wait 20-50 minutes (first time only; later builds reuse Nuitka's
   compiler cache and are faster).

5. The standalone app will be at:
   dist\main.dist\GeoAnnotate3D.exe
   (double-click it directly to test before packaging the installer)

6. If Inno Setup 6 is installed, the script also produces:
   installer\GeoAnnotate3D_Setup_v1.0.0.exe

WHY THERE'S NO SINGLE-FILE .exe:
─────────────────────────────────────────
Nuitka can build a self-extracting single .exe (--onefile), but this
app's compressed payload (PyTorch + CUDA runtime + VTK) is about
2.3 GB — over the 2 GB limit the MSVC linker imposes on that mode
(fails with "LINK : fatal error LNK1248: image size ... exceeds the
maximum allowed"). This isn't a flag to tune around, it's a hard
format limit, confirmed in practice. Instead, build_exe.bat produces a
--standalone folder (dist\main.dist\), and Inno Setup packages that
whole folder into ONE installer .exe for distribution — end users still
only ever see and run a single file, they just never see that it's a
folder underneath.

WHAT GETS GENERATED:
─────────────────────────────────────────
• dist\main.dist\ — the full standalone app folder, ~6-8 GB uncompressed
  (for local testing; don't distribute this folder directly)
• installer\GeoAnnotate3D_Setup_v1.0.0.exe — the one file to distribute
• No Python installation required on the target machine
• Compatible with Windows 10/11 (64-bit)
• Source code is protected (compiled to native C)

DISTRIBUTION:
─────────────────────────────────────────
Distribute only the installer (installer\GeoAnnotate3D_Setup_v1.0.0.exe).
It registers the app in Windows "Add/Remove Programs" with a proper
uninstaller and Start Menu/desktop shortcuts, which is what public users
expect from a real installer — and it's a fraction of the 6-8 GB
unpacked size thanks to Inno Setup's own LZMA compression.

NOTES:
─────────────────────────────────────────
• If Nuitka errors with "missing module", add:
  --include-package=MODULE_NAME to build_exe.bat

• If Windows Defender flags/blocks the build or the installer while
  testing (a known false-positive pattern for Nuitka/PyInstaller
  binaries — check Windows Security → Protection history), add the
  project folder and the built .exe names as Defender exclusions before
  rebuilding.

• Windows only for now — there is no macOS/Linux build script yet,
  even though the application code itself is cross-platform
  (PyQt5/VTK/numpy).

• The first build takes a long time. Later builds are faster because
  Nuitka caches compiled objects (clcache) — build_exe.bat redirects
  that cache, and the system TEMP used during the build, to
  build_cache\ inside this project folder instead of C:\, so a large
  build doesn't fill up the system drive.

═══════════════════════════════════════════════════════════════════
