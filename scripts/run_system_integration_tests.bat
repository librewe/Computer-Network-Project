@echo off
chcp 65001 > nul
setlocal

cd /d "%~dp0.."

echo ============================================================
echo 第三部分：系统集成与前端测试
echo ============================================================
echo.

python --version
if errorlevel 1 (
    echo Python 不可用，请先安装或配置 PATH。
    exit /b 1
)

echo [1/2] 运行系统集成测试脚本...
set PYTHONDONTWRITEBYTECODE=1
python scripts\test_system_integration.py
if errorlevel 1 (
    echo.
    echo 测试未通过，请根据上面的失败信息检查代码。
    exit /b 1
)

echo.
echo [2/2] 测试完成。
echo 如果你要联调演示，可继续手动运行：
echo   cd src-trained
echo   streamlit run dashboard.py --server.port 8501
echo.
echo ============================================================
echo 所有测试执行结束
echo ============================================================

endlocal

pause
