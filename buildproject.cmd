@echo off
rem clarautils installieren/aktualisieren (editable, pip install -e .)
rem Ziel-venv: die aktivierte venv, sonst die Repo-.venv. Extra pip-Args werden durchgereicht.
rem Beispiel: buildproject.cmd --upgrade
setlocal
cd /d "%~dp0" || exit /b 1

if defined VIRTUAL_ENV (
    set "PY=%VIRTUAL_ENV%\Scripts\python.exe"
) else (
    set "PY=.venv\Scripts\python.exe"
)

echo Installing clarautils (editable) with: %PY%
"%PY%" -m pip install -e . %*
if errorlevel 1 (
    echo INSTALL FAILED
    exit /b 1
)

"%PY%" -c "import clarautils; from clarautils import Bitty, DefinedBit; print('clarautils', clarautils.__version__, 'OK')"
