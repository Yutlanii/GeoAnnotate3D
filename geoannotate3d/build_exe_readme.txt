═══════════════════════════════════════════════════════════════════
 GeoAnnotate3D — Instructions for building the executable
═══════════════════════════════════════════════════════════════════

PREREQUISITES (install once):
─────────────────────────────────────────
1. Python 3.11 (https://python.org/downloads)
   ✓ Check "Add Python to PATH" during install

2. Visual Studio Build Tools 2022 (free C++ compiler)
   https://visualstudio.microsoft.com/visual-cpp-build-tools/
   ✓ Select: "Desktop development with C++"

3. Install Python dependencies:
   pip install nuitka ordered-set zstandard
   pip install -r requirements.txt

BUILDING THE EXECUTABLE:
─────────────────────────────────────────
1. Open CMD or PowerShell as administrator
2. Navigate to the project folder:
   cd C:\path\to\GeoAnnotate3D\geoannotate3d

3. Run:
   build_exe.bat

4. Wait 20-40 minutes (first time only)
5. The executable will be at:
   dist\GeoAnnotate3D.exe

If Inno Setup 6 is also installed, the script additionally produces
an installer at:
   installer\GeoAnnotate3D_Setup_v1.0.0.exe

WHAT GETS GENERATED:
─────────────────────────────────────────
• A single .exe file (~300-600MB depending on what VTK pulls in)
• No Python installation required on the target machine
• Nothing else to install
• Compatible with Windows 10/11 (64-bit)
• Source code is protected (compiled to native C)

DISTRIBUTION:
─────────────────────────────────────────
Distribute the installer (installer\GeoAnnotate3D_Setup_v1.0.0.exe),
not the raw standalone .exe — it registers the app in Windows
"Add/Remove Programs" with a proper uninstaller and Start Menu/desktop
shortcuts, which is what public users expect from a real installer.

NOTES:
─────────────────────────────────────────
• If Nuitka errors with "missing module", add:
  --include-package=MODULE_NAME to build_exe.bat

• Windows only for now — there is no macOS/Linux build script yet,
  even though the application code itself is cross-platform
  (PyQt5/VTK/numpy).

• The first build takes a long time. Later builds are faster because
  Nuitka caches the results.

═══════════════════════════════════════════════════════════════════
