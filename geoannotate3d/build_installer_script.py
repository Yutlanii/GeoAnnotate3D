"""
build_installer_script.py

Genera el archivo installer_setup.iss para Inno Setup.
Corre automáticamente desde build_exe.bat antes de llamar a iscc.

Inno Setup 6 crea un instalador profesional con:
  - Pantalla de bienvenida con nombre y descripción
  - Selección de directorio de instalación
  - Acceso directo en el escritorio y menú inicio
  - Desinstalador automático
  - Verificación de Windows 10/11
"""
from pathlib import Path

# DIST_DIR: carpeta --standalone completa (no un .exe --onefile). El
# payload comprimido de esta app (PyTorch+CUDA+VTK, ~2.28 GB) supera el
# límite de 2 GB del linker de MSVC para el modo --onefile de Nuitka
# (LNK1248, confirmado en la práctica), así que el instalador empaqueta
# la carpeta completa en vez de un solo .exe autoextraíble — el usuario
# final sigue descargando y corriendo un único .exe (este instalador),
# Inno Setup solo se encarga de comprimir/descomprimir la carpeta.
DIST_DIR   = r"dist\main.dist"
OUTPUT_DIR = r"installer"
APP_NAME   = "GeoAnnotate3D"
APP_VER    = "1.0.0"
APP_PUBLISHER = "GeoAnnotate3D"
APP_URL    = "https://github.com/Yutlanii/GeoAnnotate3D"
APP_ICON   = r"ui\icon.ico"
APP_EXE    = "GeoAnnotate3D.exe"

# Verificar que la carpeta standalone y su ejecutable existen
if not (Path(DIST_DIR) / APP_EXE).exists():
    print(f"[ERROR] No se encontró {DIST_DIR}\\{APP_EXE}")
    print("        Ejecuta build_exe.bat primero para compilar el standalone.")
    raise SystemExit(1)

iss = f"""
; ── GeoAnnotate3D Installer Script ──────────────────────────────────────────
; Generado automáticamente por build_installer_script.py
; ─────────────────────────────────────────────────────────────────────────────

#define MyAppName      "{APP_NAME}"
#define MyAppVersion   "{APP_VER}"
#define MyAppPublisher "{APP_PUBLISHER}"
#define MyAppURL       "{APP_URL}"
#define MyAppExeName   "{APP_EXE}"

[Setup]
AppId={{{{E2F7A3B1-4C8D-4E6F-9A2B-1D3C5E7F9001}}}}
AppName={{#MyAppName}}
AppVersion={{#MyAppVersion}}
AppPublisher={{#MyAppPublisher}}
AppPublisherURL={{#MyAppURL}}
AppSupportURL={{#MyAppURL}}
AppUpdatesURL={{#MyAppURL}}
DefaultDirName={{autopf}}\\{{#MyAppName}}
DefaultGroupName={{#MyAppName}}
AllowNoIcons=yes
LicenseFile=
OutputDir={OUTPUT_DIR}
OutputBaseFilename=GeoAnnotate3D_Setup_v{APP_VER}
SetupIconFile={APP_ICON}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
WizardResizable=yes
MinVersion=10.0
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={{app}}\\{{#MyAppExeName}}

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Crear acceso directo en el escritorio"; GroupDescription: "Accesos directos:"; Flags: unchecked
Name: "startmenuicon"; Description: "Crear acceso directo en el menú inicio"; GroupDescription: "Accesos directos:"

[Files]
; Carpeta standalone completa (Nuitka --standalone, sin --onefile — ver
; el comentario junto a DIST_DIR arriba). recursesubdirs+createallsubdirs
; copia todo el árbol tal cual; ignoreversion evita que Inno Setup se
; salte archivos por comparación de versión (no todos son .exe/.dll con
; versión embebida).
Source: "{DIST_DIR}\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Menú inicio
Name: "{{group}}\\{{#MyAppName}}"; Filename: "{{app}}\\{{#MyAppExeName}}"
Name: "{{group}}\\Desinstalar {{#MyAppName}}"; Filename: "{{uninstallexe}}"
; Escritorio (opcional)
Name: "{{autodesktop}}\\{{#MyAppName}}"; Filename: "{{app}}\\{{#MyAppExeName}}"; Tasks: desktopicon

[Run]
Filename: "{{app}}\\{{#MyAppExeName}}"; Description: "Abrir {{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[Code]
// Verificar arquitectura x64
function InitializeSetup(): Boolean;
begin
  if not IsWin64 then
  begin
    MsgBox('GeoAnnotate3D requiere Windows 10/11 de 64 bits.', mbError, MB_OK);
    Result := False;
  end
  else
    Result := True;
end;
"""

out = Path("installer_setup.iss")
out.write_text(iss, encoding="utf-8")
print(f"[OK] Generado: {out.resolve()}")
print(f"     Fuente:   {DIST_DIR}\\ (carpeta completa)")
print(f"     Salida:   {OUTPUT_DIR}\\GeoAnnotate3D_Setup_v{APP_VER}.exe")
