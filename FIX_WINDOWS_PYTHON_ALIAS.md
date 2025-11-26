# 修复 Windows Python 应用别名问题

## 问题诊断

如果您运行 `where python` 显示：
```
C:\Users\user\AppData\Local\Microsoft\WindowsApps\python.exe
```

而 `python --version` 没有任何输出，说明您遇到了 **Windows 应用别名占位符**问题。

这个 `python.exe` 不是真正的 Python，只是 Windows 10/11 的一个快捷方式，用于引导用户从 Microsoft Store 安装 Python。

---

## 🛠️ 解决方法

### 方法 1: 禁用 Windows 应用别名（推荐）

#### 步骤 1: 打开 Windows 设置

1. 按 `Win + I` 打开"设置"
2. 或点击"开始菜单" → 点击齿轮图标 ⚙️

#### 步骤 2: 进入应用执行别名设置

**Windows 11**:
```
设置 → 应用 → 高级应用设置 → 应用执行别名
```

**Windows 10**:
```
设置 → 应用 → 应用和功能 → 应用执行别名（在右侧）
```

或者直接搜索："应用执行别名" / "App execution aliases"

#### 步骤 3: 关闭 Python 别名

在"应用执行别名"列表中，找到并**关闭**以下两个选项：

```
应用安装程序: python.exe    [开启] → [关闭]
应用安装程序: python3.exe   [开启] → [关闭]
```

将它们从**开启**切换到**关闭**。

#### 步骤 4: 验证别名已禁用

打开 PowerShell，运行：
```powershell
where python
```

如果显示 `信息: 用给定的模式找不到文件。` 或没有输出，说明别名已成功禁用。

#### 步骤 5: 安装真正的 Python

现在可以从 python.org 安装真正的 Python 了：

1. 下载 Python 3.10.11:
   ```
   https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
   ```

2. 运行安装程序，**务必勾选**：
   ```
   ☑ Add Python 3.10 to PATH
   ```

3. 完成安装后，关闭并重新打开 PowerShell

4. 验证安装：
   ```powershell
   python --version
   # 应该显示: Python 3.10.11
   ```

---

### 方法 2: 直接安装 Python（覆盖别名）

如果您不想修改 Windows 设置，也可以直接安装 Python，它会自动优先于 Windows 别名。

#### 步骤 1: 下载并安装 Python

1. 下载: https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe

2. 运行安装程序时，选择 **"Customize installation"**

3. 在 "Advanced Options" 页面，**务必勾选**：
   ```
   ☑ Add Python to environment variables
   ☑ Precompile standard library
   ```

4. 完成安装

#### 步骤 2: 验证安装

关闭并重新打开 PowerShell，运行：

```powershell
where python
```

**正确的输出**应该类似：
```
C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe
C:\Users\user\AppData\Local\Microsoft\WindowsApps\python.exe
```

或
```
C:\Program Files\Python310\python.exe
C:\Users\user\AppData\Local\Microsoft\WindowsApps\python.exe
```

注意：**第一个路径**应该是真正的 Python 安装路径，而不是 WindowsApps。

#### 步骤 3: 测试 Python

```powershell
python --version
# 应该显示: Python 3.10.11
```

如果仍然没有输出，说明 WindowsApps 的别名仍然优先，需要使用**方法 1**禁用别名。

---

## 🔍 故障排查

### 问题: 安装后 `python --version` 仍然没有输出

**原因**: Windows 应用别名的优先级高于您安装的 Python

**解决**:
1. 按照**方法 1**禁用 Windows 应用别名
2. 或使用完整路径运行 Python:
   ```powershell
   C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe --version
   ```

### 问题: 找不到"应用执行别名"选项

**原因**: Windows 版本较旧或系统语言不同

**解决**:
- 在 Windows 设置中搜索 "alias" 或 "别名"
- 或在 PowerShell 中直接禁用（管理员权限）:
  ```powershell
  Remove-Item "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe"
  Remove-Item "$env:LOCALAPPDATA\Microsoft\WindowsApps\python3.exe"
  ```
  **警告**: 这会直接删除别名文件，请谨慎操作

### 问题: 权限被拒绝

**解决**:
1. 右键点击 PowerShell
2. 选择"以管理员身份运行"
3. 重新执行安装

---

## ✅ 验证 Python 正确安装的方法

运行以下命令并检查输出：

```powershell
# 1. 检查 Python 版本
python --version
# 期望输出: Python 3.10.11

# 2. 检查 Python 路径
where python
# 第一个路径应该不在 WindowsApps 目录下

# 3. 检查 pip 是否可用
python -m pip --version
# 期望输出: pip 23.x.x from ...

# 4. 测试 Python 是否能运行
python -c "print('Hello, Python!')"
# 期望输出: Hello, Python!
```

如果以上所有命令都能正常工作，说明 Python 已正确安装！

---

## 📋 完整操作流程

1. ✅ 禁用 Windows 应用别名（设置 → 应用 → 应用执行别名）
2. ✅ 下载 Python 3.10.11 安装程序
3. ✅ 安装时勾选 "Add Python to PATH"
4. ✅ 关闭并重新打开 PowerShell
5. ✅ 验证 `python --version` 显示正确版本
6. ✅ 运行 `install-requirements.bat` 安装项目依赖
7. ✅ 运行 `go-web-system-python.bat` 启动项目

---

## 🎯 下一步

Python 正确安装后，继续：

1. 安装项目依赖:
   ```cmd
   双击运行: install-requirements.bat
   ```

2. 启动项目:
   ```cmd
   双击运行: go-web-system-python.bat
   ```

3. 在浏览器中打开: http://localhost:7897

祝您使用愉快！🎤
