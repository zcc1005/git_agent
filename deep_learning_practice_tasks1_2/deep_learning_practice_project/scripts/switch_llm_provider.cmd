@echo off
chcp 65001 >nul
setlocal

if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
    goto run_conda
)

if exist "%USERPROFILE%\anaconda3\envs\dl_practice\python.exe" (
    goto run_dl_practice
)

where python >nul 2>nul && goto run_python

where py >nul 2>nul && goto run_py

echo Python not found. Activate the dl_practice environment and try again.
exit /b 1

:run_conda
"%CONDA_PREFIX%\python.exe" "%~dp0switch_llm_provider.py" %*
exit /b %ERRORLEVEL%

:run_dl_practice
"%USERPROFILE%\anaconda3\envs\dl_practice\python.exe" "%~dp0switch_llm_provider.py" %*
exit /b %ERRORLEVEL%

:run_python
python "%~dp0switch_llm_provider.py" %*
exit /b %ERRORLEVEL%

:run_py
py -3 "%~dp0switch_llm_provider.py" %*
exit /b %ERRORLEVEL%
