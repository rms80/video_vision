@echo off
rem Force-restart the segviewer Vite dev server on port 4444.
rem Kills any existing listener on 4444, then starts `npm run dev`.
setlocal enabledelayedexpansion

set PORT=4444
set VIDEOVISION_DIR=%~dp0

echo [run-server] checking port %PORT%...

set FOUND=0
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
  set FOUND=1
  echo [run-server] killing PID %%P on port %PORT%
  taskkill /F /PID %%P >nul 2>&1
)
if "!FOUND!"=="0" echo [run-server] port %PORT% is free

cd /d "%VIDEOVISION_DIR%"
echo [run-server] starting vite dev server in %VIDEOVISION_DIR%
call npm run dev
