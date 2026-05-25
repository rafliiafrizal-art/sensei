@echo off
title GURU - Screening Dokumen
cd /d "%~dp0"
chcp 65001 >nul

cls
echo.
echo  ==========================================
echo    GURU - Document Screening
echo  ==========================================
echo.
echo  Memproses file baru di folder document ...
echo.

python screening.py

echo.
echo  ==========================================
echo    Screening selesai. AI siap digunakan.
echo  ==========================================
echo.
echo  Tekan tombol apapun untuk menutup ...
pause >nul
