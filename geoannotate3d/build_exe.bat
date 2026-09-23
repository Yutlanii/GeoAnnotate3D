@echo off
setlocal EnableDelayedExpansion
REM ═══════════════════════════════════════════════════════════════════════════
REM  GeoAnnotate3D — Build Script v2.0 (Nuitka + Inno Setup)
REM
REM  Genera:
REM    dist\main.dist\               (carpeta standalone completa, ~6-8 GB;
REM                                   GeoAnnotate3D.exe vive adentro)
REM    installer\GeoAnnotate3D_Setup_v1.0.0.exe  (instalador Windows — este
REM                                   es el único archivo que se distribuye)
REM
REM  NO se usa --onefile: el payload comprimido de esta app (PyTorch+CUDA+
REM  VTK, ~2.28 GB) supera el límite de 2 GB del linker de MSVC para ese
REM  modo (LNK1248). Ver el comentario junto al comando de Nuitka abajo.
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
if not exist "build_cache\nuitka_cache" mkdir build_cache\nuitka_cache
if not exist "build_cache\tmp" mkdir build_cache\tmp

REM ── Redirigir cachés y TEMP a esta unidad (no a C:) ──────────────────────
REM  Por defecto Nuitka guarda su caché de compilación (clcache: objetos
REM  .obj de los ~15000 archivos C generados) en %LOCALAPPDATA%, es decir
REM  en C:. Igual que el TEMP del sistema, que Nuitka también usa para
REM  parte del staging al armar el .exe onefile. En una compilación con
REM  PyTorch+CUDA+VTK eso son varios GB — confirmado en la práctica: un
REM  build falló con "OSError: [Errno 28] No space left on device" copiando
REM  DLLs de CUDA porque C: se quedó sin espacio, aunque el proyecto vive
REM  en otra unidad con más espacio libre. Redirigir ambos aquí evita
REM  llenar C: sin querer.
set NUITKA_CACHE_DIR=%~dp0build_cache\nuitka_cache
set TEMP=%~dp0build_cache\tmp
set TMP=%~dp0build_cache\tmp
echo [INFO] Cache de Nuitka y TEMP redirigidos a: %~dp0build_cache
echo.

REM ── Limpiar restos de un build anterior fallido ──────────────────────────
REM  Si un intento previo se interrumpio (sin espacio, cierre forzado,
REM  etc.), deja carpetas intermedias de varios GB en dist\ que Nuitka
REM  intentaria reutilizar/parchar de forma inconsistente. Empezar limpio
REM  evita tanto binarios corruptos como espacio desperdiciado.
if exist "dist\main.build"          rmdir /s /q "dist\main.build"
if exist "dist\main.dist"           rmdir /s /q "dist\main.dist"
if exist "dist\main.onefile-build"  rmdir /s /q "dist\main.onefile-build"

echo [2/4] Compilando con Nuitka (puede tardar 20-50 min primera vez)...
echo       Las compilaciones siguientes son mas rapidas por el cache.
echo.

REM ── Nuitka: compilar a standalone ────────────────────────────────────────
REM  IMPORTANTE: corre esto desde un venv LIMPIO dedicado a compilar (ver
REM  build_env\, creado con "python -m venv build_env" desde un Python base
REM  sin nada instalado por fuera), no desde un entorno conda/venv que uses
REM  para otros proyectos. Confirmado en la práctica: compilar desde un
REM  entorno "de trabajo" (con paquetes de otros proyectos filtrándose por
REM  el user-site de Windows, %APPDATA%\Python\PythonXXX\site-packages)
REM  hizo que Nuitka intentara compilar TensorFlow, boto3, cv2, dash,
REM  folium, grpc y otros paquetes que GeoAnnotate3D nunca usa — miles de
REM  archivos C de más, más lento y más frágil (un error de toolchain de
REM  MSVC compilando un módulo de TensorFlow tumbó una compilación entera
REM  que no tenía nada que ver con nuestro código).
REM  (sklearn también se quitó de la lista de abajo por lo mismo: no se usa
REM  en ningún archivo del proyecto.)
REM  --onefile SE QUITÓ A PROPÓSITO: el payload comprimido de esta app
REM  (PyTorch+CUDA+VTK) pesa ~2.28 GB, y el linker de MSVC que arma el
REM  bootstrap de --onefile no puede enlazar un bloque de objeto COFF
REM  mayor a 2 GB (0x80000000) — confirmado en la práctica con
REM  "LINK : fatal error LNK1248: el tamaño de imagen ... supera el
REM  tamaño máximo permitido (80000000)". No es una bandera que se pueda
REM  ajustar: es un límite estructural del formato para ese modo de
REM  enlazado. La solución real para una app de este tamaño en Windows
REM  es distribuir la carpeta --standalone completa (dist\main.dist\)
REM  envuelta en el instalador de Inno Setup — que para el usuario final
REM  sigue siendo un solo .exe para descargar/correr, Inno Setup
REM  simplemente empaqueta la carpeta en vez de que Nuitka arme un
REM  bootstrap autoextraíble.
REM
REM  --enable-plugin=numpy y --enable-plugin=torch también se quitaron:
REM  Nuitka avisa "This plugin has been deprecated, do not enable it
REM  anymore" para ambos (ya vienen integrados por defecto).
REM  =disable: sin consola visible para el usuario final (app GUI normal).
REM  Se puso en =force temporalmente para depurar un crash silencioso al
REM  arrancar (=disable no muestra consola en absoluto, así que cualquier
REM  excepción se perdía sin dejar rastro) — ya resuelto (ver los dos
REM  bloques de comentarios de abajo) y confirmado con una compilación
REM  real: la app abre y se ve igual que corriendo desde código fuente.
REM
REM  ui\icons\*.svg y ui\splash_bg.png se cargan del disco en tiempo de
REM  ejecución (ui/icons.py, ui/splash.py) — no son código Python, así que
REM  sin --include-data-dir/--include-data-files para ellos Nuitka nunca
REM  los copiaba al standalone: confirmado en la práctica, el exe
REM  arrancaba con decenas de "[icons] falta ui/icons/X.svg" y sin la
REM  imagen del splash.
REM
REM  --noinclude-unittest-mode=nofollow SE QUITÓ: hacía que Nuitka
REM  descartara todo lo relacionado a 'unittest' asumiendo que solo se usa
REM  en tests, pero alguna dependencia (numpy/matplotlib/h5py) importa
REM  unittest.mock de verdad en tiempo de ejecución — confirmado en la
REM  práctica, crasheaba con "IMPORT_HARD_UNITTEST__MOCK: Unexpected
REM  failure of hard import of 'unittest.mock'" apenas arrancaba.
REM
REM  --nofollow-import-to=open3d/dash/plotly/jedi/IPython: open3d es
REM  opcional (core/pointcloud.py lo importa en un try/except ImportError
REM  con un parser manual de respaldo para PLY) pero Nuitka lo seguía y
REM  empaquetaba igual, junto con TODO lo que open3d arrastra (dash,
REM  plotly, jedi/IPython del lado Jupyter) — ~229 MB muertos que nadie
REM  usa. Con GitHub limitando los assets de Release a 2 GiB por archivo
REM  y el instalador dando 2.045 GiB (confirmado, justo por encima),
REM  esto es lo que lo baja a un tamaño publicable sin tocar ninguna
REM  funcionalidad real: sin estas librerías, "import open3d" simplemente
REM  falla en runtime como ya está previsto, y se usa el parser manual.
python -m nuitka ^
  --standalone ^
  --windows-console-mode=disable ^
  --windows-product-name="GeoAnnotate3D" ^
  --windows-product-version="1.0.0.0" ^
  --windows-file-version="1.0.0.0" ^
  --windows-file-description="LiDAR Point Cloud Annotation & GeoAI Platform" ^
  --windows-company-name="GeoAnnotate3D" ^
  --windows-icon-from-ico="ui\icon.ico" ^
  --enable-plugin=pyqt5 ^
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
  --include-data-dir=ui\icons=ui\icons ^
  --include-data-files=ui\splash_bg.png=ui\splash_bg.png ^
  --nofollow-import-to=open3d ^
  --nofollow-import-to=dash ^
  --nofollow-import-to=plotly ^
  --nofollow-import-to=jedi ^
  --nofollow-import-to=IPython ^
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
echo [3/4] Carpeta standalone generada: dist\main.dist\GeoAnnotate3D.exe
echo       Tamaño aproximado: 6-8 GB (PyTorch+CUDA+VTK sin comprimir).
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
echo       Instalador: installer\GeoAnnotate3D_Setup_v1.0.0.exe

:done
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║  BUILD COMPLETADO                                           ║
echo ║                                                             ║
echo ║  Carpeta standalone (para probar sin instalar):             ║
echo ║    dist\main.dist\GeoAnnotate3D.exe                        ║
echo ║                                                             ║
echo ║  Para DISTRIBUIR, usa el instalador (si se genero):         ║
echo ║    installer\GeoAnnotate3D_Setup_v1.0.0.exe                ║
echo ║  Es el UNICO archivo que necesita el usuario final.         ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
pause
