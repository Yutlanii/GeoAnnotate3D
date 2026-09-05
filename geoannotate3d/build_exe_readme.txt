═══════════════════════════════════════════════════════════════════
 GeoAnnotate3D — Instrucciones para crear el ejecutable
═══════════════════════════════════════════════════════════════════

REQUISITOS PREVIOS (instalar una sola vez):
─────────────────────────────────────────
1. Python 3.11 (https://python.org/downloads)
   ✓ Marcar "Add Python to PATH" al instalar

2. Visual Studio Build Tools 2022 (compilador C++, gratuito)
   https://visualstudio.microsoft.com/visual-cpp-build-tools/
   ✓ Seleccionar: "Desktop development with C++"

3. Instalar dependencias Python:
   pip install nuitka ordered-set zstandard
   pip install -r requirements.txt

COMPILAR EL EJECUTABLE:
─────────────────────────────────────────
1. Abrir CMD o PowerShell como administrador
2. Navegar a la carpeta del proyecto:
   cd C:\ruta\a\GeoAnnotate3D\geoannotate3d

3. Ejecutar:
   build_exe.bat

4. Esperar 20-40 minutos (solo la primera vez)
5. El ejecutable estará en:
   dist\GeoAnnotate3D.exe

EL EJECUTABLE GENERADO:
─────────────────────────────────────────
• Un solo archivo .exe (~300-600MB según lo que VTK incluya)
• No requiere Python instalado en la computadora destino
• No requiere instalar nada adicional
• Compatible con Windows 10/11 (64-bit)
• El código fuente está protegido (compilado a C nativo)

DISTRIBUCIÓN:
─────────────────────────────────────────
Solo comparte el archivo: dist\GeoAnnotate3D.exe
El usuario solo hace doble clic y abre el software.

NOTAS:
─────────────────────────────────────────
• Si Nuitka da error "missing module", agrega:
  --include-package=NOMBRE_MODULO al build_exe.bat

• Para versión macOS/Linux, usa build_exe.sh (requiere gcc/clang)

• Primera compilación tarda mucho. Compilaciones siguientes son más
  rápidas porque Nuitka cachea los resultados.

═══════════════════════════════════════════════════════════════════
