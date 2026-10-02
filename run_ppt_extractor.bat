@echo off
chcp 65001 >nul 2>&1
title 课件捕获助手 v8.0
echo =============================================
echo          课件捕获助手 v8.0
echo =============================================
echo.
echo   Alt+I = 锁定/解除窗口
echo   Alt+Q = 框选 PPT 范围
echo   Alt+W = 开始/停止手动检测
echo   Alt+E = 保存当前页
echo   Alt+Y = 停止自动任务
echo   Alt+O = 退出
echo   ESC   = 取消选择
echo =============================================
echo.
set "UV_CACHE_DIR=%~dp0.uv-cache"
pushd "%~dp0"
where uv >nul 2>&1
if not errorlevel 1 (
    uv run --project "%~dp0" python "%~dp0ppt_extractor.py"
    if not errorlevel 1 goto :done
    echo.
    echo uv run failed, trying the project virtual environment...
)
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0ppt_extractor.py"
) else (
    python "%~dp0ppt_extractor.py"
)
:done
popd
pause
