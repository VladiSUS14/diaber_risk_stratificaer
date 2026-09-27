@echo off
cd /d "%~dp0"

echo ==========================================
echo Starting Diabetes Risk Stratifier
echo ==========================================

echo.
echo Project folder:
echo %cd%
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found.
    echo Please run install.bat first.
    echo.
    pause
    exit /b 1
)

if not exist "app\streamlit_app.py" (
    echo ERROR: app\streamlit_app.py not found.
    echo Check project structure.
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
call ".venv\Scripts\activate.bat"

echo.
echo Starting Streamlit application...
echo If the browser does not open automatically, copy the Local URL from this window.
echo.

streamlit run app\streamlit_app.py

pause