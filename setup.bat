@echo off
chcp 65001 >nul 2>&1
title 课件收集助手 - 首次准备
echo =============================================
echo          课件收集助手 - 首次准备
echo =============================================
echo.
echo   这一步会下载程序运行所需文件，首次运行需要联网。
echo   OCR 组件下载较大，请保持窗口打开并稍候。
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 goto :failed
echo.
echo   准备完成！以后双击 run_ppt_extractor.bat 启动助手。
pause
exit /b 0

:failed
echo.
echo   准备未完成。请检查网络后重新运行 setup.bat。
pause
exit /b 1
