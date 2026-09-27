@echo off
cd /d "%~dp0"

echo ==========================================
echo Installing Diabetes Risk Stratifier
echo ==========================================

echo.
echo Project folder:
echo %cd%
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Creating virtual environment...

    where py >nul 2>nul
    if %errorlevel%==0 (
        py -m venv .venv
    ) else (
        python -m venv .venv
    )

    if not exist ".venv\Scripts\python.exe" (
        echo.
        echo ERROR: Failed to create virtual environment.
        echo Make sure Python is installed and added to PATH.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

echo.
echo Activating virtual environment...
call ".venv\Scripts\activate.bat"

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing project dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Installation completed successfully.
echo ==========================================
echo.
pause