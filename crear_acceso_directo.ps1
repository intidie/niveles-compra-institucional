<#
.SYNOPSIS
  Crea el acceso directo "Niveles Institucionales.lnk" en el Escritorio para abrir la app sin consola.

.DESCRIPTION
  El acceso directo ejecuta  <repo>\.venv\Scripts\pythonw.exe app_ui.py  con la carpeta del repo como directorio de
  trabajo y la ventana minimizada (pythonw no abre consola; app_ui.py redirige stdout/stderr a devnull si no existen).
  La carpeta del repo se deduce de la ubicacion de este script ($PSScriptRoot), asi que sirve en cualquier equipo.
  El Escritorio se obtiene con [Environment]::GetFolderPath('Desktop') (respeta OneDrive y redirecciones).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\crear_acceso_directo.ps1
  powershell -ExecutionPolicy Bypass -File .\crear_acceso_directo.ps1 -Carpeta "C:\Temp"   # prueba sin tocar el Escritorio
#>
param(
    [string]$Carpeta = [Environment]::GetFolderPath('Desktop')
)

$ErrorActionPreference = 'Stop'

$repo = $PSScriptRoot
$pythonw = Join-Path $repo '.venv\Scripts\pythonw.exe'
$app = Join-Path $repo 'app_ui.py'

if (-not (Test-Path -LiteralPath $pythonw)) {
    Write-Error "No existe $pythonw. Crea el entorno virtual primero: python -m venv .venv ; .venv\Scripts\pip install -r requirements.txt"
    exit 1
}
if (-not (Test-Path -LiteralPath $app)) {
    Write-Error "No existe $app. Ejecuta este script desde la raiz del repositorio."
    exit 1
}
if (-not (Test-Path -LiteralPath $Carpeta -PathType Container)) {
    Write-Error "La carpeta de destino no existe: $Carpeta"
    exit 1
}

$destino = Join-Path $Carpeta 'Niveles Institucionales.lnk'
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut($destino)
$lnk.TargetPath = $pythonw
$lnk.Arguments = 'app_ui.py'
$lnk.WorkingDirectory = $repo
$lnk.WindowStyle = 7                        # 7 = minimizada (pedido por el usuario). pythonw no abre consola, asi que
                                            # solo afecta a la ventana de la app: con la ventana nativa (pywebview) puede
                                            # arrancar minimizada; si molesta, cambiar a 1 (normal) y volver a ejecutar.
$lnk.IconLocation = "$pythonw,0"            # icono del propio pythonw.exe (no se anade ningun binario al repo)
$lnk.Description = 'Screener de niveles de compra institucionales'
$lnk.Save()

if (-not (Test-Path -LiteralPath $destino)) {
    Write-Error "No se pudo crear $destino"
    exit 1
}
Write-Host "Acceso directo creado: $destino"
Write-Host "  Destino:           $($lnk.TargetPath) $($lnk.Arguments)"
Write-Host "  Directorio inicio: $($lnk.WorkingDirectory)"
