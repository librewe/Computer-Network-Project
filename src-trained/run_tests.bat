@echo off
chcp 65001 > nul
echo ============================================================
echo 代理底层开发与流量特征提取模块 - 快速测试
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/3] 检查Python环境...
python --version
if errorlevel 1 (
    echo 错误: Python未安装或未添加到PATH
    pause
    exit /b 1
)
echo.

echo [2/3] 检查依赖包...
python -c "import psutil; import numpy; import torch" 2>nul
if errorlevel 1 (
    echo 警告: 缺少依赖包，正在安装...
    pip install psutil numpy torch -q
)
echo.

echo [3/3] 运行单元测试...
echo.
python test_validation.py

echo.
echo ============================================================
echo 单元测试完成
echo ============================================================
echo.
echo 继续运行集成测试? (Y/N)
set /p choice=
if /i "%choice%"=="Y" goto run_integration
goto end

:run_integration
echo.
echo 运行集成测试和压力测试...
python test_integration_stress.py

echo.
echo ============================================================
echo 所有测试完成
echo ============================================================
echo.
echo 查看详细测试计划: TEST_PLAN.md
echo.

:end
pause
