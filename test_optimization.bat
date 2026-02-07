@echo off
echo ============================================================
echo QUICK OPTIMIZATION TEST
echo ============================================================
echo.
echo This will test the optimization system (5 minutes)
echo Running 2 trials per model to verify everything works.
echo.
pause

cd models
python test_optimization.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo TEST PASSED!
    echo ============================================================
    echo.
    echo The optimization system works correctly.
    echo.
    echo To run full optimization (3-4 hours):
    echo   run_optimization.bat
    echo.
    echo Or manually:
    echo   cd models
    echo   python hyperparameter_optimization.py
    echo.
) else (
    echo.
    echo ============================================================
    echo TEST FAILED
    echo ============================================================
    echo.
    echo Check the error messages above.
    echo.
)

pause
