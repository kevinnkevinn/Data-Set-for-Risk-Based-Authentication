Param(
    [string]$DownloadDir = "D:\Unduhan",
    [string]$RootDir    = "D:\Portofolio Data\Data Set for Risk-Based Authentication",
    [string]$ProjectName = "Device_Fingerprinting_Portfolio",
    [switch]$InstallJupyter = $true,
    [switch]$RunNotebook = $false,
    [switch]$RunAPI = $false,
    [switch]$BuildDocker = $false
)

# =========================
# Utility helpers
# =========================
function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
        Write-Host "Created directory: $Path"
    }
}

function Move-IfExists {
    param([string]$Src, [string]$Dst)
    if (Test-Path -LiteralPath $Src) {
        $dstDir = Split-Path -LiteralPath $Dst -Parent
        if (-not (Test-Path -LiteralPath $dstDir)) { Ensure-Dir -Path $dstDir }
        Move-Item -Force -LiteralPath $Src -Destination $Dst
        Write-Host "Moved: $Src -> $Dst"
    } else {
        Write-Host "Skip (not found): $Src"
    }
}

function Rename-IfExists {
    param([string]$Path, [string]$NewName)
    if (Test-Path -LiteralPath $Path) {
        $dir = Split-Path -LiteralPath $Path -Parent
        Rename-Item -LiteralPath $Path -NewName $NewName -Force
        Write-Host "Renamed: $Path -> $(Join-Path $dir $NewName)"
    } else {
        Write-Host "Skip rename (not found): $Path"
    }
}

function Extract-Rar {
    param([string]$RarPath, [string]$DestDir)
    # PowerShell native Expand-Archive doesn't support .rar.
    # Try 7-Zip if installed.
    $sevenZip = "C:\Program Files\7-Zip\7z.exe"
    if (Test-Path -LiteralPath $sevenZip) {
        Ensure-Dir -Path $DestDir
        & $sevenZip x -y "-o$DestDir" $RarPath | Out-Null
        Write-Host "Extracted RAR with 7-Zip: $RarPath -> $DestDir"
    } else {
        Write-Warning "7-Zip not found. Please install 7-Zip or extract the RAR manually: $RarPath"
    }
}

# =========================
# Paths
# =========================
$Proj = Join-Path $RootDir $ProjectName
$NotebooksDir = Join-Path $Proj "notebooks"
$DataDir = Join-Path $Proj "data"
$SrcDir = Join-Path $Proj "src"
$ApiDir = Join-Path $SrcDir "api"
$ModelsDir = Join-Path $SrcDir "models"
$FeaturesDir = Join-Path $SrcDir "features"
$UtilsDir = Join-Path $SrcDir "utils"

# Create directories
Ensure-Dir -Path $Proj
Ensure-Dir -Path $NotebooksDir
Ensure-Dir -Path $DataDir
Ensure-Dir -Path $SrcDir
Ensure-Dir -Path $ApiDir
Ensure-Dir -Path $ModelsDir
Ensure-Dir -Path $FeaturesDir
Ensure-Dir -Path $UtilsDir

# =========================
# Move files from DownloadDir
# =========================
Move-IfExists -Src (Join-Path $DownloadDir "Device_Fingerprinting_Portfolio.ipynb") -Dst (Join-Path $NotebooksDir "Device_Fingerprinting_Portfolio.ipynb")
Move-IfExists -Src (Join-Path $DownloadDir "Device_Fingerprinting_Portfolio (1).ipynb") -Dst (Join-Path $NotebooksDir "Device_Fingerprinting_Portfolio (1).ipynb")
Rename-IfExists -Path (Join-Path $NotebooksDir "Device_Fingerprinting_Portfolio (1).ipynb") -NewName "Device_Fingerprinting_RBA.ipynb"

Move-IfExists -Src (Join-Path $DownloadDir "device_fingerprint.py") -Dst (Join-Path $FeaturesDir "device_fingerprint.py")
Move-IfExists -Src (Join-Path $DownloadDir "device_fingerprint (1).py") -Dst (Join-Path $FeaturesDir "device_fingerprint (1).py")
Rename-IfExists -Path (Join-Path $FeaturesDir "device_fingerprint (1).py") -NewName "device_fingerprint_rba.py"

Move-IfExists -Src (Join-Path $DownloadDir "train.py") -Dst (Join-Path $ModelsDir "train.py")
Move-IfExists -Src (Join-Path $DownloadDir "app.py") -Dst (Join-Path $ApiDir "app.py")
Move-IfExists -Src (Join-Path $DownloadDir "io.py") -Dst (Join-Path $UtilsDir "io.py")
Move-IfExists -Src (Join-Path $DownloadDir "sample_events.csv") -Dst (Join-Path $DataDir "sample_events.csv")
Move-IfExists -Src (Join-Path $DownloadDir "Dockerfile") -Dst (Join-Path $Proj "Dockerfile")
Move-IfExists -Src (Join-Path $DownloadDir "requirements.txt") -Dst (Join-Path $Proj "requirements.txt")
Move-IfExists -Src (Join-Path $DownloadDir "README.md") -Dst (Join-Path $Proj "README.md")

# =========================
# Optional: extract legacy RAR (if exists)
# =========================
$rarPath = Join-Path $RootDir "Data Set for Risk-Based Authentication.rar"
$legacyOut = Join-Path $Proj "_legacy_RBA"
if (Test-Path -LiteralPath $rarPath) {
    Extract-Rar -RarPath $rarPath -DestDir $legacyOut
} else {
    Write-Host "RAR not found (skip): $rarPath"
}

# =========================
# Python venv + install
# =========================
$VenvPath = Join-Path $Proj ".venv"
$PythonExe = "python"

# Create venv if not exists
if (-not (Test-Path -LiteralPath $VenvPath)) {
    & $PythonExe -m venv $VenvPath
    Write-Host "Created venv at: $VenvPath"
} else {
    Write-Host "Using existing venv: $VenvPath"
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "venv python not found: $VenvPython"
}

# Upgrade pip & install requirements
& $VenvPython -m pip install --upgrade pip
$ReqPath = Join-Path $Proj "requirements.txt"
if (Test-Path -LiteralPath $ReqPath) {
    & $VenvPython -m pip install -r $ReqPath
} else {
    Write-Warning "requirements.txt not found at $ReqPath"
}

if ($InstallJupyter) {
    & $VenvPython -m pip install jupyter
}

# =========================
# Optional runs
# =========================
if ($RunNotebook) {
    $NbPath = Join-Path $NotebooksDir "Device_Fingerprinting_RBA.ipynb"
    if (-not (Test-Path -LiteralPath $NbPath)) {
        $NbPath = Join-Path $NotebooksDir "Device_Fingerprinting_Portfolio.ipynb"
    }
    Write-Host "Launching Jupyter Notebook: $NbPath"
    Start-Process -FilePath $VenvPython -ArgumentList "-m","jupyter","notebook","`"$NbPath`"" -WorkingDirectory $Proj
}

if ($RunAPI) {
    # Ensure uvicorn is available (comes from requirements)
    Write-Host "Starting API at http://127.0.0.1:8000 ..."
    Start-Process -FilePath $VenvPython -ArgumentList "-m","uvicorn","src.api.app:app","--reload","--port","8000" -WorkingDirectory $Proj
}

if ($BuildDocker) {
    Write-Host "Building Docker image: device-fingerprinting:latest"
    docker build -t device-fingerprinting:latest $Proj
}

Write-Host "All done. Project is ready at: $Proj"
