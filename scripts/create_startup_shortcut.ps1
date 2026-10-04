# Creates a hidden startup shortcut in the user's Windows Startup folder
$WshShell = New-Object -ComObject WScript.Shell
$startupFolder = [System.Environment]::GetFolderPath('Startup')
$shortcutPath = Join-Path $startupFolder "DailyGitHubAutomation.lnk"

$shortcut = $WshShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "powershell.exe"
$shortcut.Arguments = "-WindowStyle Hidden -ExecutionPolicy Bypass -Command `"python 'C:\Imp docs\Abhishek documents\Github\scripts\daily_job.py'`""
$shortcut.WorkingDirectory = "C:\Imp docs\Abhishek documents\Github"
$shortcut.WindowStyle = 7
$shortcut.Description = "Autonomous daily AI / ML project generation and deployment to GitHub."
$shortcut.Save()

Write-Host "SUCCESS: Created silent startup shortcut at $shortcutPath"
