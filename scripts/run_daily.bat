@echo off
REM ============================================================================
REM Daily AI, ML, Data Science & Analytics Project Automation Runner
REM ============================================================================

set SCRIPT_DIR=%~dp0
set REPO_DIR=%SCRIPT_DIR%..
cd /d "%REPO_DIR%"

echo [INFO] Running Daily Project Automation...
python -m automation.runner --generate --execute --deploy --push

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Daily project successfully created, evaluated, and committed!
) else (
    echo [ERROR] Automation encountered an error. Check logs above.
)

REM Pause if run by double-clicking in File Explorer
if "%~1"=="" (
    pause
)
