# Forward-test runner — advances the shadow ledger one step and refreshes the
# progress dashboard, appending all output to a timestamped log. This is the
# script the Windows Scheduled Task invokes (see register_forward_task.ps1).
#
# Manual use:  powershell -ExecutionPolicy Bypass -File run_forward_step.ps1

$proj = "C:\Users\noaro\QUANT\Quant-Research\ACTIVE\funding-carry-harvest"
$py   = "C:\DevApps\Python313\python.exe"
$fwd  = Join-Path $proj "scripts\forward_test_bybit.py"
$log  = Join-Path $proj "data_cache\forward_log.txt"
$ts   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# FROZEN 2026-11-17 16:00 UTC: the cross-sectional ledgers (baseline and variant c)
# no longer advance. Their state, ledgers and logs in data_cache/ are kept unchanged.
# Reasons: docs/preregistration_2026-11-17.md, section "Separate decision". To print
# their final reports:  python scripts/forward_test_bybit.py --report [--variant c]

# Live shadow ledger: cash-and-carry (CC L=21 K=7), selected by the
# pre-registered improvement study of 2026-11-07. Separate script, state and log.
$cc     = Join-Path $proj "scripts\forward_test_cc.py"
$logCC  = Join-Path $proj "data_cache\forward_log_cc.txt"
$stepCC = (& $py $cc --step 2>&1 | Out-String)
$dashCC = (& $py $cc --dashboard 2>&1 | Out-String)
Add-Content -Path $logCC -Encoding utf8 -Value "===== $ts (local) =====`n$stepCC$dashCC"
Write-Output "[cash-and-carry] $($stepCC.Trim())"

# Regime-filtered cash-and-carry (tau = 15% a year), selected by the pre-registered
# study of 2026-11-17 (docs/preregistration_2026-11-17.md, rule R5). Own state and log.
$logT  = Join-Path $proj "data_cache\forward_log_cc_tau15.txt"
$stepT = (& $py $cc --step --tau 0.15 2>&1 | Out-String)
$dashT = (& $py $cc --dashboard --tau 0.15 2>&1 | Out-String)
Add-Content -Path $logT -Encoding utf8 -Value "===== $ts (local) =====`n$stepT$dashT"
Write-Output "[cash-and-carry tau 15%] $($stepT.Trim())"
