@echo off
setlocal EnableDelayedExpansion
REM ═══════════════════════════════════════════════════════════════════════════
REM  GeoAnnotate3D — Build Script v2.0 (Nuitka + Inno Setup)
REM
REM  Genera:
REM    dist\GeoAnnotate3D.exe        (ejecutable standalone onefile)
REM    installer\GeoAnnotate3D_Setup.exe  (instalador Windows)
REM
REM  Requisitos:
REM    - Python 3.11 64-bit  (https://python.org)
REM    - Visual Studio Build Tools 2022 con "Desktop development with C++"
REM    - Inno Setup 6  (https://jrsoftware.org/isinfo.php)  [para el instalador]
REM    - pip install nuitka ordered-set zstandard imageio
REM    - pip install -r requirements.txt
REM ═══════════════════════════════════════════════════════════════════════════

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║        GeoAnnotate3D — Build System v2.0                    ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM ── Verificar Python ─────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado. Instala Python 3.11 64-bit.
    echo         https://python.org/downloads
    pause
    exit /b 1
)

REM ── Verificar Nuitka ──────────────────────────────────────────────────────
python -m nuitka --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Instalando Nuitka...
    pip install nuitka ordered-set zstandard
)

REM ── Compilar extensión C (_fastcore) ─────────────────────────────────────
echo [1/4] Compilando motor C (_fastcore)...
python core\build_extension.py
if errorlevel 1 (
    echo [WARN] No se pudo compilar _fastcore.c — el software funciona sin él, pero más lento.
    echo        Instala Visual Studio Build Tools si quieres el motor C.
)
echo.

REM ── Detectar CUDA ─────────────────────────────────────────────────────────
set CUDA_FLAGS=
set CUDA_MSG=sin CUDA
python -c "import torch; print(torch.cuda.is_available())" 2>nul | findstr /i "True" >nul
if not errorlevel 1 (
    set CUDA_MSG=con CUDA
    echo [INFO] CUDA detectado — el ejecutable incluira soporte GPU.
) else (
    echo [INFO] CUDA no detectado — ejecutable solo CPU ^(PyTorch CPU^).
    echo        Para version GPU: instala CUDA Toolkit y PyTorch CUDA.
)
echo.

REM ── Detectar rutas de PyTorch ────────────────────────────────────────────
for /f "delims=" %%i in ('python -c "import torch, os; print(os.path.dirname(torch.__file__))" 2^>nul') do set TORCH_DIR=%%i
for /f "delims=" %%i in ('python -c "import site; print(site.getsitepackages()[0])" 2^>nul') do set SITE_PACKAGES=%%i

echo [INFO] PyTorch en: !TORCH_DIR!
echo [INFO] site-packages: !SITE_PACKAGES!
echo.

REM ── Crear carpetas de salida ─────────────────────────────────────────────
if not exist "dist" mkdir dist
if not exist "installer" mkdir installer
if not exist "build_cache" mkdir build_cache

echo [2/4] Compilando con Nuitka (puede tardar 20-50 min primera vez)...
echo       Las compilaciones siguientes son mas rapidas por el cache.
echo.

REM ── Nuitka: compilar a standalone ────────────────────────────────────────
python -m nuitka ^
  --standalone ^
  --onefile ^
  --windows-disable-console ^
  --windows-product-name="GeoAnnotate3D" ^
  --windows-product-version="1.0.0.0" ^
  --windows-file-version="1.0.0.0" ^
  --windows-file-description="LiDAR Point Cloud Annotation & GeoAI Platform" ^
  --windows-company-name="GeoAnnotate3D" ^
  --windows-icon-from-ico="ui\icon.ico" ^
  --enable-plugin=pyqt5 ^
  --enable-plugin=numpy ^
  --enable-plugin=torch ^
  --include-package=torch ^
  --include-package=torch.nn ^
  --include-package=torch.optim ^
  --include-package=torch.utils ^
  --include-package=torch.utils.data ^
  --include-package=torch.cuda ^
  --include-package=torch.backends ^
  --include-package=torch.backends.cudnn ^
  --include-package=vtk ^
  --include-package=vtkmodules ^
  --include-package=laspy ^
  --include-package=pye57 ^
  --include-package=scipy ^
  --include-package=scipy.spatial ^
  --include-package=scipy.interpolate ^
  --include-package=scipy.ndimage ^
  --include-package=sklearn ^
  --include-package=sklearn.neighbors ^
  --include-package=matplotlib ^
  --include-package=matplotlib.cm ^
  --include-package=matplotlib.colors ^
  --include-package=h5py ^
  --include-package=psutil ^
  --include-package=colorsys ^
  --include-package=importlib ^
  --include-package=importlib.util ^
  --include-package=core ^
  --include-package=render ^
  --include-package=annotation ^
  --include-package=ui ^
  --include-data-files=core\_fastcore*.pyd=core\ ^
  --include-data-dir=core=core ^
  --noinclude-unittest-mode=nofollow ^
  --python-flag=no_site ^
  --python-flag=no_warnings ^
  --assume-yes-for-downloads ^
  --output-dir=dist ^
  --output-filename=GeoAnnotate3D ^
  --jobs=4 ^
  main.py

if errorlevel 1 (
    echo.
    echo [ERROR] La compilacion fallo.
    echo         Revisa el log de Nuitka arriba para detalles.
    echo.
    echo Causas comunes:
    echo   - Modulo no incluido: agrega --include-package=NOMBRE
    echo   - Error en el codigo fuente: corre python main.py para verificar
    echo   - Falta Visual C++ Build Tools
    pause
    exit /b 1
)

echo.
echo [3/4] Ejecutable generado: dist\GeoAnnotate3D.exe
echo       Tamaño aproximado: 400-800 MB segun las librerias incluidas.
echo.

REM ── Generar instalador con Inno Setup ─────────────────────────────────────
where iscc >nul 2>&1
if errorlevel 1 (
    REM Buscar Inno Setup en rutas comunes
    if exist "C:\Program Files (x86)\Inno Setup 6\iscc.exe" (
        set ISCC="C:\Program Files (x86)\Inno Setup 6\iscc.exe"
    ) else if exist "C:\Program Files\Inno Setup 6\iscc.exe" (
        set ISCC="C:\Program Files\Inno Setup 6\iscc.exe"
    ) else (
        echo [INFO] Inno Setup no encontrado. Solo se genera el .exe standalone.
        echo        Para generar el instalador: instala Inno Setup 6 y corre build_installer.bat
        goto :done
    )
) else (
    set ISCC=iscc
)

echo [4/4] Generando instalador con Inno Setup...
python build_installer_script.py
!ISCC! installer_setup.iss
if errorlevel 1 (
    echo [WARN] No se pudo generar el instalador.
    goto :done
)
echo       Instalador: installer\GeoAnnotate3D_Setup.exe

:done
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║  BUILD COMPLETADO                                           ║
echo ║                                                             ║
echo ║  Ejecutable standalone:                                     ║
echo ║    dist\GeoAnnotate3D.exe                                  ║
echo ║                                                             ║
echo ║  El usuario solo necesita este .exe para correr            ║
echo ║  el software. No requiere Python ni nada adicional.        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
pause
