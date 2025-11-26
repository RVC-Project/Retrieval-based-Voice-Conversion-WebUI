# Python 安装指南（Windows 系统）

## 当前问题诊断

您运行 `python --version` 时没有任何输出，这说明：
- ❌ Python **未安装**
- ❌ 或 Python 已安装但**未添加到 PATH 环境变量**

这是运行 RVC-WebUI 项目的**必要前提条件**。

---

## 📥 步骤 1: 下载 Python

### 推荐版本
- ✅ **Python 3.10.11**（最稳定，推荐）
- ✅ Python 3.9.x 或 3.11.x（也可以）
- ⚠️ 不要使用 Python 3.12+（可能有兼容性问题）
- ⚠️ 不要使用 Python 3.7 或更早版本（项目不支持）

### 下载地址

**方法 1: 官网下载（推荐）**
1. 访问: https://www.python.org/downloads/
2. 点击 "Download Python 3.10.11" 或 "Download Python 3.11.x"
3. 下载完成后运行安装程序

**方法 2: 直接下载链接**
- Python 3.10.11 (64位): https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
- Python 3.11.9 (64位): https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

⚠️ **重要**: 请下载 64 位版本（amd64），不要下载 32 位版本！

---

## 🔧 步骤 2: 安装 Python（关键步骤）

### ⚠️ 超级重要！必须勾选 "Add Python to PATH"

运行下载的 `.exe` 安装程序后：

```
┌─────────────────────────────────────────────────┐
│  Install Python 3.10.11                         │
├─────────────────────────────────────────────────┤
│                                                 │
│  ☑ Install launcher for all users (recommended)│
│                                                 │
│  ☑ Add Python 3.10 to PATH    ← 必须勾选这个！  │
│                                                 │
│  [ Install Now ]                                │
│  [ Customize installation ]                     │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 安装选项说明

**必须勾选的选项**:
- ✅ **Add Python to PATH** ← 这是最重要的！
- ✅ Install launcher for all users

**推荐的安装方式**:
1. 勾选 ✅ "Add Python to PATH"
2. 点击 "Customize installation"（自定义安装）
3. 在 "Optional Features" 页面，确保勾选：
   - ✅ pip
   - ✅ tcl/tk and IDLE
   - ✅ Python test suite
   - ✅ py launcher
4. 在 "Advanced Options" 页面，确保勾选：
   - ✅ Install for all users
   - ✅ Add Python to environment variables ← 再次确认
   - ✅ Precompile standard library
5. 点击 "Install" 开始安装

### 为什么 "Add Python to PATH" 如此重要？

如果不勾选这个选项：
- ❌ PowerShell 无法识别 `python` 命令
- ❌ 无法使用 `pip` 命令
- ❌ 启动脚本无法找到 Python
- ❌ 需要手动配置环境变量（复杂且容易出错）

---

## ✅ 步骤 3: 验证安装

### 3.1 重启 PowerShell

**重要**: 安装完成后必须**关闭并重新打开** PowerShell 窗口！

旧的 PowerShell 窗口不会加载新的环境变量。

### 3.2 验证 Python 安装

打开**新的** PowerShell 窗口，运行：

```powershell
python --version
```

**正确的输出**应该类似：
```
Python 3.10.11
```

或
```
Python 3.11.9
```

### 3.3 验证 pip 安装

```powershell
python -m pip --version
```

**正确的输出**应该类似：
```
pip 23.0.1 from C:\Users\...\Python\Python310\lib\site-packages\pip (python 3.10)
```

### 3.4 如果仍然没有输出

如果关闭 PowerShell 重新打开后还是没有输出，可能是以下原因：

#### 原因 1: 安装时没有勾选 "Add Python to PATH"

**解决方法**:
1. 卸载 Python（控制面板 → 程序和功能 → 卸载 Python）
2. 重新安装，**务必勾选** "Add Python to PATH"

#### 原因 2: 环境变量没有刷新

**解决方法 A**: 重启电脑（最简单）

**解决方法 B**: 手动刷新环境变量
1. 关闭所有 PowerShell 窗口
2. 按 `Win + R`，输入 `cmd`，回车
3. 在命令提示符中运行 `python --version`
4. 如果 CMD 中可以运行，说明需要重启 PowerShell

#### 原因 3: 使用了 Python 3.12+ 或从 Microsoft Store 安装

**解决方法**:
1. 卸载当前的 Python
2. 从 python.org 下载 Python 3.10.11
3. 按照上面的步骤重新安装

---

## 🔍 故障排查: 手动检查 Python 安装路径

### 检查 Python 是否真的安装了

打开文件资源管理器，检查以下路径是否存在：

**方式 1: 安装在用户目录**
```
C:\Users\你的用户名\AppData\Local\Programs\Python\Python310\
```

**方式 2: 安装在系统目录**
```
C:\Program Files\Python310\
```

在这个文件夹中应该能看到 `python.exe` 文件。

### 如果 Python 已安装但找不到命令

您可以使用**完整路径**来运行 Python：

```powershell
# 假设 Python 安装在用户目录
C:\Users\你的用户名\AppData\Local\Programs\Python\Python310\python.exe --version
```

但这**不是长久之计**，最好的方法是重新安装并勾选 "Add Python to PATH"。

---

## 🎯 步骤 4: 安装 RVC-WebUI 项目依赖

Python 安装成功后，回到项目目录，双击运行：

```
install-requirements.bat
```

这个脚本会：
1. ✅ 自动检测您的显卡类型
2. ✅ 安装 PyTorch（深度学习框架）
3. ✅ 安装所有项目依赖包

安装过程可能需要 **5-15 分钟**，请耐心等待。

---

## 📋 完整操作检查清单

在开始运行项目之前，请确认：

- [ ] Python 3.8-3.11 已安装
- [ ] 安装时勾选了 "Add Python to PATH"
- [ ] 已重启 PowerShell
- [ ] `python --version` 显示正确的版本号
- [ ] `python -m pip --version` 显示正确的版本号
- [ ] 已运行 `install-requirements.bat` 安装依赖
- [ ] 依赖安装成功，没有错误信息
- [ ] 已下载必要的预训练模型文件

完成以上所有步骤后，就可以运行 `go-web-system-python.bat` 启动项目了！

---

## 🆘 常见错误和解决方案

### 错误 1: "python 不是内部或外部命令"
**原因**: Python 没有添加到 PATH
**解决**: 重新安装 Python 并勾选 "Add Python to PATH"

### 错误 2: "pip 不是内部或外部命令"
**原因**: pip 没有随 Python 一起安装
**解决**: 使用 `python -m pip` 代替 `pip` 命令

### 错误 3: "'python' 意外终止"
**原因**: Python 安装损坏
**解决**: 卸载并重新安装 Python

### 错误 4: SSL 证书错误
**原因**: 网络问题或 Python 安装问题
**解决**:
```powershell
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org torch
```

### 错误 5: 权限被拒绝
**原因**: 没有管理员权限
**解决**: 右键 PowerShell，选择"以管理员身份运行"

---

## 📞 获取帮助

如果按照本指南操作后仍有问题：

1. 截图您的错误信息
2. 记录您的 Python 版本: `python --version`
3. 记录您的 pip 版本: `python -m pip --version`
4. 查看项目的 `TROUBLESHOOTING_CN.md` 文件
5. 或访问官方社区寻求帮助

---

## 🎉 成功安装后的下一步

Python 安装成功后：
1. ✅ 运行 `install-requirements.bat` 安装依赖
2. ✅ 下载预训练模型文件
3. ✅ 运行 `go-web-system-python.bat` 启动项目
4. ✅ 在浏览器中打开 http://localhost:7897
5. ✅ 开始使用 RVC 语音转换！

祝您使用愉快！🎤
