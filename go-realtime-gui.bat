@echo off

@REM 定义你使用的python，指向你配好的环境中的python路径
@REM     如 C:\Users\yourname\miniconda3\envs\rvc\python.exe
set PYTHON=

@REM 默认配置
if not defined PYTHON (set PYTHON=python)

@REM 实际执行
%PYTHON% gui_v1.py
pause