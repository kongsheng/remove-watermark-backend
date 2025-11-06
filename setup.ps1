#requires -Version 5.1
param(
  [switch]$StartServer
)

$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "[ERR ] $msg" -ForegroundColor Red }

# 1) Check Python
Write-Info "Checking Python..."
try {
  $pyVer = python -c "import sys; print(sys.version)" 2>$null
  if (-not $pyVer) { throw "Python not found in PATH" }
  Write-Info "Python: $pyVer"
} catch {
  Write-Err "Python 未找到，请安装 Python 3.9+ 并确保加入 PATH。"
  exit 1
}

# 2) Create & activate venv
Write-Info "Creating virtual env (.venv)..."
python -m venv .venv
$venvActivate = Join-Path (Resolve-Path .).Path ".venv/Scripts/Activate.ps1"
if (-not (Test-Path $venvActivate)) { Write-Err "虚拟环境创建失败"; exit 1 }
Write-Info "Activating venv..."
. $venvActivate

# 3) Upgrade pip
Write-Info "Upgrading pip..."
python -m pip install --upgrade pip

# 4) Install base requirements
Write-Info "Installing base requirements..."
pip install -r requirements.txt

# 5) Detect GPU (NVIDIA)
Write-Info "Detecting GPU..."
$hasNvidia = $false
try {
  $gpus = Get-CimInstance Win32_VideoController | Select-Object -Property Name,AdapterRAM,DriverVersion
  foreach ($g in $gpus) {
    if ($g.Name -match "NVIDIA") { $hasNvidia = $true }
  }
  if ($hasNvidia) { Write-Info "Detected NVIDIA GPU" } else { Write-Warn "No NVIDIA GPU detected, will install CPU PyTorch" }
} catch {
  Write-Warn "GPU 检测失败（继续按 CPU）: $($_.Exception.Message)"
}

# 6) Install PyTorch
function Install-TorchCpu {
  Write-Info "Installing PyTorch (CPU)..."
  try { pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu } catch { Write-Err "CPU 版 PyTorch 安装失败: $($_.Exception.Message)" }
}
function Install-TorchCuda118 {
  Write-Info "Installing PyTorch (CUDA 11.8)..."
  try { pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 } catch { throw }
}
function Install-TorchCuda121 {
  Write-Info "Installing PyTorch (CUDA 12.1)..."
  try { pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 } catch { throw }
}

if ($hasNvidia) {
  $torchOk = $false
  try { Install-TorchCuda118; $torchOk = $true } catch { Write-Warn "CUDA 11.8 安装失败，尝试 12.1..." }
  if (-not $torchOk) {
    try { Install-TorchCuda121; $torchOk = $true } catch { Write-Warn "CUDA 12.1 安装失败，回退 CPU..." }
  }
  if (-not $torchOk) { Install-TorchCpu }
} else {
  Install-TorchCpu
}

# 7) Install lama-cleaner
Write-Info "Installing lama-cleaner..."
pip install lama-cleaner

# 8) Verify torch & device
Write-Info "Verifying torch and CUDA..."
$pycode = @'
import torch
print("torch_version=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
print("device_count=", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current_device=", torch.cuda.current_device())
    print("device_name=", torch.cuda.get_device_name(torch.cuda.current_device()))
'@

$pyTmp = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "torch_check_{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
Set-Content -Path $pyTmp -Value $pycode -Encoding UTF8
python $pyTmp
Remove-Item $pyTmp -ErrorAction SilentlyContinue

# 9) Set env and start server optionally
Write-Info "Setting INPAINT_ENGINE=lama (current session)"
$env:INPAINT_ENGINE = "lama"

if ($StartServer) {
  Write-Info "Starting Flask server (LaMa)..."
  python app.py
} else {
  Write-Info "环境准备完成。可手动启动：`python app.py`"
}
