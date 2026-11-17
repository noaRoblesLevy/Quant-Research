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

$step = (& $py $fwd --step 2>&1 | Out-String)
$dash = (& $py $fwd --dashboard 2>&1 | Out-String)

Add-Content -Path $log -Encoding utf8 -Value "===== $ts (local) =====`n$step$dash"

# Echo the latest equity line to the console too (handy when run manually)
Write-Output $step.Trim()

# Parallel shadow ledger: variant c (inverse-vol + BTC/ETH hedge). Runs after the
# baseline and logs separately, so a failure here never touches the baseline test.
$logC  = Join-Path $proj "data_cache\forward_log_c.txt"
$stepC = (& $py $fwd --step --variant c 2>&1 | Out-String)
$dashC = (& $py $fwd --dashboard --variant c 2>&1 | Out-String)
Add-Content -Path $logC -Encoding utf8 -Value "===== $ts (local) =====`n$stepC$dashC"
Write-Output "[variant c] $($stepC.Trim())"

# Third parallel shadow ledger: cash-and-carry (CC L=21 K=7), selected by the
# pre-registered improvement study of 2026-11-07. Separate script, state and log.
$cc     = Join-Path $proj "scripts\forward_test_cc.py"
$logCC  = Join-Path $proj "data_cache\forward_log_cc.txt"
$stepCC = (& $py $cc --step 2>&1 | Out-String)
$dashCC = (& $py $cc --dashboard 2>&1 | Out-String)
Add-Content -Path $logCC -Encoding utf8 -Value "===== $ts (local) =====`n$stepCC$dashCC"
Write-Output "[cash-and-carry] $($stepCC.Trim())"
