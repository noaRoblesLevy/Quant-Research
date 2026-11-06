# One-time setup: register a Windows Scheduled Task that auto-runs the forward-test
# step every 8 hours (matching the funding cadence). Runs while you are logged on;
# StartWhenAvailable catches up missed runs after sleep/shutdown. No admin needed.
#
#   powershell -ExecutionPolicy Bypass -File register_forward_task.ps1
#
# Manage afterwards:
#   Get-ScheduledTask  FundingCarryForwardTest
#   Start-ScheduledTask FundingCarryForwardTest        # run once now
#   Unregister-ScheduledTask FundingCarryForwardTest -Confirm:$false   # remove

$taskName = "FundingCarryForwardTest"
$proj     = "C:\Users\noaro\QUANT\Quant-Research\ACTIVE\funding-carry-harvest"
$wrapper  = Join-Path $proj "scripts\run_forward_step.ps1"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$wrapper`""

# Base at the next 01:00 local, then repeat every 8h indefinitely.
$base    = (Get-Date).Date.AddHours(1)
if ($base -lt (Get-Date)) { $base = $base.AddHours(8) }
$trigger = New-ScheduledTaskTrigger -Once -At $base `
    -RepetitionInterval (New-TimeSpan -Hours 8) -RepetitionDuration (New-TimeSpan -Days 3650)

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 20) -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Force `
    -Description "Auto-advances the funding-carry forward-test shadow ledger every 8h."

Write-Output "Registered '$taskName'. First run at $base, then every 8h."
Write-Output "Run now with:  Start-ScheduledTask $taskName"
