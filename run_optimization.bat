@echo off
echo ============================================================
echo HYPERPARAMETER OPTIMIZATION - QUICK START
echo ============================================================
echo.
echo This will optimize all models (3-4 hours total)
echo.
echo Models to optimize:
echo   1. Random Forest (30 trials, ~30 min)
echo   2. XGBoost (30 trials, ~45 min)
echo   3. TabTransformer (15 trials, ~2.5 hours)
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo Starting optimization...
echo.

cd models
python hyperparameter_optimization.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo OPTIMIZATION COMPLETE!
    echo ============================================================
    echo.
    echo Updating dashboard...
    python update_dashboard_with_optimized.py
    echo.
    echo ============================================================
    echo DONE!
    echo ============================================================
    echo.
    echo Next steps:
    echo   1. Restart Streamlit: streamlit run app.py
    echo   2. Navigate to Prediction page
    echo   3. See improved models with optimization badge!
    echo.
) else (
    echo.
    echo ============================================================
    echo OPTIMIZATION FAILED
    echo ============================================================
    echo.
    echo Check the error messages above.
    echo.
)

pause
