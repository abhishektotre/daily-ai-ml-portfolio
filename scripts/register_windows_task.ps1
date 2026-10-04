# Register Daily GitHub Automation in Windows Task Scheduler

$taskName = "DailyGitHubProjectAutomation"
$pythonwPath = "C:\Users\abhis\AppData\Local\Programs\Python\Python313\pythonw.exe"
$scriptPath = "C:\Imp docs\Abhishek documents\Github\scripts\daily_job.py"
$workDir = "C:\Imp docs\Abhishek documents\Github"

# Verify pythonw exists
if (-not (Test-Path $pythonwPath)) {
    $pythonwPath = (Get-Command pythonw.exe -ErrorAction SilentlyContinue).Source
    if (-not $pythonwPath) {
        $pythonwPath = "pythonw.exe"
    }
}

# 1. Define Action (runs completely hidden in background via pythonw)
$action = New-ScheduledTaskAction -Execute $pythonwPath -Argument "`"$scriptPath`"" -WorkingDirectory $workDir

# 2. Define Triggers:
# Trigger A: Daily at 09:00 AM
$triggerDaily = New-ScheduledTaskTrigger -Daily -At "09:00AM"

# Trigger B: At user logon (catches up every day user turns on / logs in to their PC)
$triggerLogon = New-ScheduledTaskTrigger -AtLogOn

# 3. Define Settings:
# StartWhenAvailable ensures it runs if PC was off/asleep during scheduled time
# Works on battery or AC power
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1)

# 4. Register Task
try {
    # Unregister existing if present
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    
    # Register new task for the current user
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger @($triggerDaily, $triggerLogon) -Settings $settings -Description "Autonomous daily AI / ML project generation and deployment to GitHub."
    Write-Host "SUCCESS: Windows Scheduled Task '$taskName' registered successfully!"
} catch {
    Write-Error "Failed to register task: $_"
    exit 1
}
