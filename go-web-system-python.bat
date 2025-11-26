@echo off
echo ================================================
echo RVC WebUI 启动脚本 (使用系统 Python)
echo ================================================
echo.

REM 检查 Python 是否可用
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python！
    echo.
    echo 请确保：
    echo 1. 已安装 Python 3.8-3.11
    echo 2. 安装时勾选了 "Add Python to PATH"
    echo.
    echo 下载地址: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [信息] 检测到 Python 版本:
python --version
echo.

REM 检查 infer-web.py 是否存在
if not exist "infer-web.py" (
    echo [错误] 未找到 infer-web.py 文件！
    echo 请确保在项目根目录运行此脚本。
    echo.
    pause
    exit /b 1
)

echo [启动] 正在启动 RVC WebUI...
echo [信息] 端口: 7897
echo [信息] 启动后请在浏览器中访问: http://localhost:7897
echo.

python infer-web.py --pycmd python --port 7897

if %errorlevel% neq 0 (
    echo.
    echo [错误] 启动失败！
    echo.
    echo 可能的原因:
    echo 1. 缺少依赖包 - 请运行: python -m pip install -r requirements.txt
    echo 2. 缺少预训练模型 - 请下载 assets 文件夹中的必要文件
    echo.
)

pause
