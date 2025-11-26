# RVC-WebUI 常见运行问题解决指南

## 问题诊断: pip 命令无法识别

### 错误信息
```
pip : 无法将"pip"项识别为 cmdlet、函数、脚本文件或可运行程序的名称。
```

### 问题原因
这个错误表示以下几种可能情况之一:
1. Python 没有安装
2. Python 已安装但没有添加到系统 PATH 环境变量
3. 在 PowerShell 中需要使用不同的命令格式

---

## 解决方案

### 方案一: 使用项目自带的启动脚本 (推荐)

根据项目结构分析,这个项目可能已经包含了必要的运行环境。**直接使用项目提供的批处理文件启动**:

#### 启动 Web 训练推理界面
```cmd
双击运行: go-web.bat
```

#### 启动实时变声界面
```cmd
双击运行: go-realtime-gui.bat
```

#### 如果您使用 AMD/Intel 显卡
```cmd
双击运行: go-web-dml.bat
```

### 方案二: 检查 Python 安装

#### 步骤 1: 检查 Python 是否已安装
在 PowerShell 中运行:
```powershell
python --version
```
或
```powershell
python3 --version
```
或
```powershell
py --version
```

#### 步骤 2: 如果 Python 未安装
1. 前往 Python 官网下载: https://www.python.org/downloads/
2. **重要**: 安装时勾选 "Add Python to PATH" 选项
3. 推荐安装 Python 3.8 - 3.11 版本(项目要求 > 3.8)

#### 步骤 3: 如果 Python 已安装但 pip 无法使用
尝试使用以下命令代替 `pip`:
```powershell
python -m pip install torch torchvision torchaudio
```

### 方案三: 检查项目内置的 Python 运行时

查看项目目录下是否存在 `runtime` 文件夹:

```powershell
# 检查 runtime 文件夹
dir runtime
```

如果存在 `runtime\python.exe`,说明项目自带了 Python 运行时,可以直接使用:

```powershell
# 使用项目自带的 Python 安装依赖
.\runtime\python.exe -m pip install torch torchvision torchaudio
```

---

## 完整安装步骤 (从零开始)

### 1. 安装 Python (如果系统中没有)
- 下载 Python 3.8-3.11: https://www.python.org/downloads/
- 安装时**必须勾选** "Add Python to PATH"

### 2. 安装 PyTorch
在命令提示符(CMD)或 PowerShell 中运行:

**如果您有 NVIDIA 显卡 (推荐)**:
```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

**如果您只有 CPU 或 AMD/Intel 显卡**:
```powershell
python -m pip install torch torchvision torchaudio
```

### 3. 安装项目依赖

根据您的硬件选择对应的 requirements 文件:

**NVIDIA 显卡用户**:
```powershell
python -m pip install -r requirements.txt
```

**AMD/Intel 显卡用户**:
```powershell
python -m pip install -r requirements-dml.txt
```

### 4. 下载必要的预训练模型

项目需要一些预训练模型才能运行。可以:
- 访问 [Hugging Face](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/) 手动下载
- 或运行下载脚本 (如果有的话)

需要下载的文件:
- `assets/hubert/hubert_base.pt`
- `assets/pretrained/` 目录下的文件
- `assets/uvr5_weights/` 目录下的文件
- `ffmpeg.exe` 和 `ffprobe.exe` (放在项目根目录)
- `rmvpe.pt` (可选,用于音高提取)

### 5. 启动项目

**方法一: 使用批处理文件 (最简单)**
```cmd
双击: go-web.bat
```

**方法二: 使用 Python 命令**
```powershell
python infer-web.py
```

---

## 常见问题

### Q1: PowerShell 执行策略错误
如果运行脚本时提示执行策略错误,运行:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q2: 模块导入错误
确保所有依赖都已正确安装:
```powershell
python -m pip install -r requirements.txt --upgrade
```

### Q3: CUDA 相关错误
- 确认您的 NVIDIA 显卡驱动已更新
- 安装对应的 CUDA 版本的 PyTorch

### Q4: 缺少 assets 文件夹中的文件
从 Hugging Face 下载所需的预训练模型文件

---

## 推荐的完整运行流程

1. **不要手动运行** `pip install torch torchvision torchaudio`
2. **检查项目根目录**是否有 `runtime` 文件夹(自带 Python)
3. **直接双击** `go-web.bat` 启动项目
4. 如果启动失败,查看错误信息:
   - 缺少依赖? → 使用 `python -m pip install -r requirements.txt`
   - 缺少模型文件? → 从 Hugging Face 下载
   - Python 版本问题? → 安装 Python 3.8-3.11

---

## 获取更多帮助

- 官方 FAQ: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/常见问题解答
- Discord 社区: https://discord.gg/HcsmBBGyVk
- 项目 Issues: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues
