@echo off
echo ================================================
echo RVC WebUI 依赖安装脚本
echo ================================================
echo.

REM 检查 Python 是否可用
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python！
    echo.
    echo 请先安装 Python 3.8-3.11
    echo 下载地址: https://www.python.org/downloads/
    echo 安装时请勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo [信息] 检测到 Python 版本:
python --version
echo.

echo [提示] 请选择您的显卡类型:
echo.
echo 1. NVIDIA 显卡 (推荐)
echo 2. AMD/Intel 显卡 (使用 DirectML)
echo 3. 仅 CPU (不推荐,速度较慢)
echo.
set /p choice="请输入选项 (1/2/3): "

if "%choice%"=="1" goto nvidia
if "%choice%"=="2" goto amd_intel
if "%choice%"=="3" goto cpu_only
echo [错误] 无效选项！
pause
exit /b 1

:nvidia
echo.
echo [步骤 1/2] 安装 PyTorch (NVIDIA CUDA 版本)...
echo.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
if %errorlevel% neq 0 (
    echo [错误] PyTorch 安装失败！
    pause
    exit /b 1
)

echo.
echo [步骤 2/2] 安装项目依赖...
echo.
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [错误] 依赖安装失败！
    pause
    exit /b 1
)
goto success

:amd_intel
echo.
echo [步骤 1/2] 安装 PyTorch (CPU 版本)...
echo.
python -m pip install torch torchvision torchaudio
if %errorlevel% neq 0 (
    echo [错误] PyTorch 安装失败！
    pause
    exit /b 1
)

echo.
echo [步骤 2/2] 安装项目依赖 (DirectML)...
echo.
python -m pip install -r requirements-dml.txt
if %errorlevel% neq 0 (
    echo [错误] 依赖安装失败！
    pause
    exit /b 1
)
goto success

:cpu_only
echo.
echo [步骤 1/2] 安装 PyTorch (CPU 版本)...
echo.
python -m pip install torch torchvision torchaudio
if %errorlevel% neq 0 (
    echo [错误] PyTorch 安装失败！
    pause
    exit /b 1
)

echo.
echo [步骤 2/2] 安装项目依赖...
echo.
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [错误] 依赖安装失败！
    pause
    exit /b 1
)
goto success

:success
echo.
echo ================================================
echo [成功] 所有依赖已安装完成！
echo ================================================
echo.
echo 接下来:
echo 1. 下载必要的预训练模型文件 (见 README.md)
echo 2. 运行 go-web-system-python.bat 启动程序
echo.
pause
