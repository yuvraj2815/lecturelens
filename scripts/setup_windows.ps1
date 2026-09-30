# Run in PowerShell from the repo root on a Snapdragon (Windows on ARM) laptop.
# Use a native ARM64 build of Python 3.11 so the QNN provider can load.
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_models.py
python -m lecturelens providers
Write-Host "If QNNExecutionProvider is listed above, the NPU path is available."
