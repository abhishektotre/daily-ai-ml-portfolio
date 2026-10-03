<#
.SYNOPSIS
    Configures a Windows Scheduled Task to run the daily AI/ML/Data Science automation automatically.
.DESCRIPTION
    Creates a scheduled task named 'DailyGitHubAIProjects' that runs scripts\run_daily.bat every day.
.PARAMETER Time
    The time of day to trigger the automation (default: '09:00AM').
.PARAMETER Remove
    Removes the existing scheduled task.
#>

param (
    [string]$Time = "09:00AM",
    [switch]$Remove
)

$TaskName = "DailyGitHubAIProjects"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BatchFile = Join-Path $ScriptDir "run_daily.bat"

if ($Remove) {
    Write-Host "Unregistering scheduled task '$TaskName'..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Scheduled task '$TaskName' removed successfully." -ForegroundColor Green
    exit 0
}

if (-not (Test-Path $BatchFile)) {
    Write-Error "Could not find batch file at '$BatchFile'"
    exit 1
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Setting up Daily GitHub Automation Windows Scheduled Task" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "• Task Name:    $TaskName"
Write-Host "• Trigger Time: Daily at $Time"
Write-Host "• Script:       $BatchFile"

$Action = New-ScheduledTaskAction -Execute "$BatchFile" -WorkingDirectory (Split-Path -Parent $ScriptDir)
$Trigger = New-ScheduledTaskTrigger -Daily -At $Time
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

try {
    # Check if already exists and unregister first
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Daily AI, ML, Data Science & Analytics GitHub Portfolio Automation" | Out-Null
    Write-Host "`n✅ Scheduled task '$TaskName' registered successfully!" -ForegroundColor Green
    Write-Host "The automation will run automatically every day at $Time." -ForegroundColor Green
    Write-Host "To run manually anytime, open PowerShell and run:"
    Write-Host "  Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Gray
} catch {
    Write-Error "Failed to register scheduled task. Ensure you have appropriate permissions: $_"
}
