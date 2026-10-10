<#
Windows convenience commands for this project.
Usage examples:
  .\scripts\run_project.ps1 test
  .\scripts\run_project.ps1 baseline
  .\scripts\run_project.ps1 app
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("data", "baseline", "bilstm", "evaluate", "app", "test")]
    [string]$Task
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

switch ($Task) {
    "data" { python -m src.data_loader --input data/raw/hinglish_sentiment.csv }
    "baseline" { python -m src.baseline }
    "bilstm" { python -m src.bilstm }
    "evaluate" { python -m src.evaluate; if ($LASTEXITCODE -eq 0) { python -m src.error_analysis } }
    "app" { streamlit run app/streamlit_app.py }
    "test" { python -m pytest -q -p no:cacheprovider }
}
